"""Load external MCP servers as long-lived child processes and expose their tools.

The loader spawns each MCP server declared in the JSON config exactly
once at server startup, holds the stdio session open via an
``AsyncExitStack``, and returns a flat list of ``Tool`` objects -- one
per advertised remote tool. The agent loop never sees the difference
between a built-in Python tool and an MCP-routed tool: both wear the
same ``Tool`` shape and dispatch through the same callable.

Why long-lived sessions: per-run subprocess startup cost was
acceptable in the pydantic-ai era because the framework owned the
lifecycle. Owning it ourselves, we can spawn once and reuse across
every concurrent run -- the MCP protocol is designed for it (each
request carries its own ID, the transport multiplexes safely).

Why ``AsyncExitStack``: ``stdio_client`` and ``ClientSession`` are
both async context managers and need their teardown sequenced
correctly. The exit stack is the standard library's idiom for
"hold N nested async contexts open across an arbitrary scope" and
unwinds them in LIFO order on shutdown.

JSON config shape (matches the existing
``.subagent-mcp.servers.json`` file format)::

    {
      "mcpServers": {
        "<name>": {
          "command": "uv",
          "args": ["run", "srclight", "serve", "--transport", "stdio"],
          "env": {"PATH": "${PATH}"},
          "cwd": "."  // optional
        },
        ...
      }
    }

Environment variable expansion uses ``${VAR}`` syntax via
``os.path.expandvars`` so callers can forward selected vars without
leaking the entire parent environment to a child process.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .agent import Tool

logger = logging.getLogger("subagent-mcp.mcp_loader")


class MCPToolLoader:
    """Owns the long-lived child MCP sessions and the Tool list they expose.

    Lifecycle: ``await loader.load()`` once at server startup; the
    sessions stay open for the lifetime of the process. ``await
    loader.aclose()`` on shutdown unwinds the exit stack in LIFO
    order, terminating each child cleanly.

    The loader is intentionally a single object (not a free function
    that returns tools + a teardown handle) because the tools' bound
    sessions and the exit stack must share a lifetime. Splitting them
    would invite use-after-close bugs.
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: list[Tool] = []
        self._loaded = False

    @property
    def tools(self) -> list[Tool]:
        """The tools advertised by every loaded MCP server, flattened."""
        return list(self._tools)

    async def load(self, config_path: Path) -> None:
        """Spawn every MCP server in the config and collect its tools.

        Missing or unparseable config is logged and treated as empty
        so a typo cannot prevent the parent server from booting --
        subagents simply run with built-in tools only. A child that
        fails to start or list its tools is logged and skipped: one
        bad server does not poison the rest.

        Idempotent: a second ``load`` call is a no-op so callers do
        not have to guard against double initialization.
        """
        if self._loaded:
            return
        self._loaded = True

        if not config_path.exists():
            logger.info(
                "No MCP servers config at %s; subagents run with built-in tools only",
                config_path,
            )
            return

        try:
            data = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.exception(
                "Failed to read MCP servers config %s; running with built-in tools only",
                config_path,
            )
            return

        servers = (data or {}).get("mcpServers") or {}
        if not isinstance(servers, dict) or not servers:
            logger.info(
                "MCP servers config %s has no entries; running with built-in tools only",
                config_path,
            )
            return

        for name, spec in servers.items():
            try:
                await self._spawn_one(name, spec)
            except Exception:
                logger.exception(
                    "Failed to spawn MCP server %r; skipping", name
                )

        logger.info(
            "MCP loader: %d server(s) loaded, %d tool(s) total",
            len(self._sessions),
            len(self._tools),
        )

    async def _spawn_one(self, name: str, spec: dict[str, Any]) -> None:
        """Spawn one child MCP server and pull its tool list into the registry.

        ``spec`` follows the standard MCP servers config shape:
        ``{command, args, env, cwd}``. Environment values are
        ``${VAR}``-expanded so config files can forward selected
        parent vars without hardcoding paths.
        """
        command = spec.get("command")
        if not command:
            logger.warning("MCP server %r missing command; skipping", name)
            return
        args = list(spec.get("args") or [])
        cwd = spec.get("cwd")
        raw_env = spec.get("env") or {}
        env: dict[str, str] = {
            k: os.path.expandvars(str(v)) for k, v in raw_env.items()
        }
        # Forward PATH if not explicitly listed -- spawning a child
        # without PATH typically means the command itself cannot be
        # resolved on the parent's behalf.
        if "PATH" not in env and "PATH" in os.environ:
            env["PATH"] = os.environ["PATH"]

        params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=cwd,
        )

        # Enter both async context managers via the exit stack so the
        # session and its underlying subprocess survive past this
        # function and tear down in LIFO order on aclose().
        read, write = await self._stack.enter_async_context(
            stdio_client(params)
        )
        session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()

        listing = await session.list_tools()
        self._sessions[name] = session

        added = 0
        for remote in listing.tools:
            tool = self._wrap_remote_tool(name, session, remote)
            self._tools.append(tool)
            added += 1
        logger.info(
            "MCP server %r initialized: %d tool(s)", name, added
        )

    def _wrap_remote_tool(
        self,
        server_name: str,
        session: ClientSession,
        remote: mcp_types.Tool,
    ) -> Tool:
        """Wrap one remote MCP tool as a local ``Tool`` the agent can call.

        Naming policy: when more than one MCP server is loaded, tools
        are prefixed with ``{server_name}__`` to disambiguate
        collisions (different servers can both advertise e.g.
        ``read_file``). With a single server, the bare tool name is
        kept so prompts stay short and readable.
        """
        # Prefix only when needed -- with one server the bare name is
        # cleaner. _spawn_one is called sequentially so we can decide
        # the policy after the fact, but doing it here is simpler:
        # always prefix when there is already at least one other
        # server registered.
        prefixed = (
            f"{server_name}__{remote.name}"
            if len(self._sessions) > 1
            else remote.name
        )

        async def call_remote(args: dict[str, Any]) -> str:
            result = await session.call_tool(remote.name, args)
            return _flatten_tool_result(result)

        return Tool(
            name=prefixed,
            description=remote.description or "",
            parameters=remote.inputSchema or {"type": "object", "properties": {}},
            fn=call_remote,
        )

    async def aclose(self) -> None:
        """Tear down every loaded MCP child session in LIFO order.

        Best-effort: any exception during teardown is logged but does
        not prevent the rest of the stack from unwinding. Callers
        should not need to handle exceptions from shutdown.
        """
        try:
            await self._stack.aclose()
        except Exception:
            logger.exception("error during MCP loader shutdown")
        finally:
            self._sessions.clear()
            self._tools.clear()
            self._loaded = False


def _flatten_tool_result(result: mcp_types.CallToolResult) -> str:
    """Render a CallToolResult as a single string for the model.

    The MCP protocol allows tool results to carry mixed content:
    text, images, audio, embedded resources. Ollama's tool-result
    message takes a single string ``content`` field, so we have to
    flatten. Policy:

    * Text content is concatenated in order.
    * Image / audio / blob content is replaced with a one-line
      placeholder including the mime type so the model knows
      something binary came back without us trying to inline it.
    * If ``isError`` is set, the result is prefixed with ``ERROR:``
      so the model can tell the call failed (the wire-level error
      flag is otherwise lost in the flattening).
    """
    parts: list[str] = []
    for item in result.content:
        if isinstance(item, mcp_types.TextContent):
            parts.append(item.text)
        elif isinstance(item, mcp_types.ImageContent):
            parts.append(f"[image: {item.mimeType}, {len(item.data)} b64 chars]")
        elif isinstance(item, mcp_types.AudioContent):
            parts.append(f"[audio: {item.mimeType}, {len(item.data)} b64 chars]")
        else:
            # Embedded resources, etc. -- best-effort serialize.
            parts.append(f"[{type(item).__name__}]")
    text = "\n".join(p for p in parts if p) or "(empty result)"
    if result.isError:
        return f"ERROR: {text}"
    return text
