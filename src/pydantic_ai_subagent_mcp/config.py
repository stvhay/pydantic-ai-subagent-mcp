"""Configuration for the subagent MCP server."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ServerConfig:
    """Configuration loaded from environment and optional config file."""

    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "gemma4:26b"
    session_dir: str = ".subagent-sessions"
    inbox_dir: str = ".subagent-inbox"
    # Path to a JSON file declaring external MCP servers whose tools
    # should be exposed to every subagent run. The file uses the
    # standard `{"mcpServers": {<name>: {command, args, env, cwd}}}`
    # shape. If the file is missing the server still boots -- subagents
    # simply have no external MCP tools, only the BUILTIN_TOOLS.
    mcp_servers_config: str = ".subagent-mcp.servers.json"
    # Server-wide cap on the number of in-flight skill turns across
    # all sessions. The session worker acquires a slot from the
    # SessionStore's semaphore before running its turn and releases
    # it on exit, so a swarm of background launches cannot saturate
    # the Ollama backend. Foreground and background callers alike
    # wait on the semaphore inside the worker; there is no admission-
    # time rejection (which would be inherently racy against the
    # execution-time acquire).
    max_concurrent_runs: int = 4
    # Per-session cap on the number of items the mailbox will hold.
    # Pushes that would exceed the cap are rejected with
    # status="mailbox_full" -- regardless of foreground/background --
    # so a runaway producer cannot grow a session's queue without
    # bound, and a foreground caller cannot bypass the cap by
    # omitting run_in_background.
    mailbox_max_depth: int = 16

    @classmethod
    def load(cls, config_path: Path | None = None) -> ServerConfig:
        """Load config from file and environment overrides."""
        data: dict[str, Any] = {}

        # Load from config file if provided or default location exists
        if config_path is None:
            config_path = Path(".subagent-mcp.json")
        if config_path.exists():
            data = json.loads(config_path.read_text())

        # Backpressure knobs: env overrides config file overrides class default.
        # Both must be positive ints; non-positive or unparseable values fall
        # back to the class default rather than crashing the server at boot.
        max_concurrent = _positive_int_env(
            "SUBAGENT_MCP_MAX_CONCURRENT_RUNS",
            data.get("max_concurrent_runs", cls.max_concurrent_runs),
            cls.max_concurrent_runs,
        )
        mailbox_max = _positive_int_env(
            "SUBAGENT_MCP_MAILBOX_MAX_DEPTH",
            data.get("mailbox_max_depth", cls.mailbox_max_depth),
            cls.mailbox_max_depth,
        )

        # Environment overrides take precedence
        return cls(
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL", data.get("ollama_base_url", cls.ollama_base_url)
            ),
            default_model=os.environ.get(
                "SUBAGENT_MCP_DEFAULT_MODEL",
                data.get("default_model", cls.default_model),
            ),
            session_dir=data.get("session_dir", cls.session_dir),
            inbox_dir=data.get("inbox_dir", cls.inbox_dir),
            max_concurrent_runs=max_concurrent,
            mailbox_max_depth=mailbox_max,
            mcp_servers_config=data.get(
                "mcp_servers_config", cls.mcp_servers_config
            ),
        )


def _positive_int_env(env_var: str, file_value: Any, default: int) -> int:
    """Resolve a positive-int knob from env / config file / default.

    Precedence: ``$env_var`` > ``file_value`` > ``default``. The result
    must be a positive integer; any non-positive or unparseable value
    silently falls through to the next layer. Backpressure caps must
    never be zero (would deadlock the server) and the server must
    never crash at boot because of a bad config value.
    """
    raw = os.environ.get(env_var)
    if raw is not None:
        try:
            parsed = int(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    try:
        parsed = int(file_value)
        if parsed > 0:
            return parsed
    except (TypeError, ValueError):
        pass
    return default
