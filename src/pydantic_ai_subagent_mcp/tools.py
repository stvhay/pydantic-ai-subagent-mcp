"""Built-in tools available to every subagent run.

Each tool is a small ``Tool`` instance: a name, a JSON-Schema for its
parameters, and an async callable that takes a parsed argument dict
and returns a string. The agent loop dispatches MCP-routed tools and
these built-ins through the same code path -- nothing in the loop
distinguishes them.

Why hand-rolled JSON Schemas: every tool has at most three parameters
and the schema is the description that gets shipped to the model.
Generating it from type hints would add a layer of indirection in
exchange for nothing -- the schema would still need to live somewhere
and the wire shape is small enough to read at a glance.

Why these specific tools: the agent needs read access to the project
(``read_file``, ``list_files``, ``search_files``), occasional write
access for skills that produce artifacts (``write_file``), and a
shell escape hatch (``shell_exec``) so a skill can run uncommon
commands without us having to wrap every CLI on the system. The
shell escape is by design -- subagents are middleware running with
the user's permissions, not a sandbox.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .agent import Tool


async def _read_file(args: dict[str, Any]) -> str:
    path = str(args.get("path") or "")
    offset = int(args.get("offset", 0) or 0)
    limit = int(args.get("limit", 500) or 500)
    try:
        p = Path(path).resolve()
        lines = p.read_text().splitlines()
        selected = lines[offset : offset + limit]
        return "\n".join(
            f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)
        )
    except Exception as e:  # noqa: BLE001 — reported to model
        return f"Error reading {path}: {e}"


async def _write_file(args: dict[str, Any]) -> str:
    path = str(args.get("path") or "")
    content = str(args.get("content") or "")
    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:  # noqa: BLE001 — reported to model
        return f"Error writing {path}: {e}"


async def _list_files(args: dict[str, Any]) -> str:
    pattern = str(args.get("pattern") or "**/*")
    path = str(args.get("path") or ".")
    try:
        base = Path(path).resolve()
        matches = sorted(base.glob(pattern))[:200]
        return "\n".join(
            str(m.relative_to(base)) for m in matches if m.is_file()
        )
    except Exception as e:  # noqa: BLE001 — reported to model
        return f"Error listing files: {e}"


async def _search_files(args: dict[str, Any]) -> str:
    pattern = str(args.get("pattern") or "")
    path = str(args.get("path") or ".")
    glob_pat = str(args.get("glob") or "")
    try:
        try:
            cmd = ["rg", "--no-heading", "--line-number", "--max-count=50",
                   pattern, path]
            if glob_pat:
                cmd.extend(["--glob", glob_pat])
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return result.stdout[:10000]
        except FileNotFoundError:
            pass  # rg not installed; CI won't have it -- fall through to grep
        cmd = ["grep", "-rn", "--max-count=50", pattern, path]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        return result.stdout[:10000] if result.stdout else "No matches found."
    except Exception as e:  # noqa: BLE001 — reported to model
        return f"Error searching: {e}"


async def _shell_exec(args: dict[str, Any]) -> str:
    command = str(args.get("command") or "")
    timeout = float(args.get("timeout", 60.0) or 60.0)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.cwd()),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output[:20000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001 — reported to model
        return f"Error executing command: {e}"


# The flat list of built-in tools advertised to every subagent run.
# Order is preserved into Ollama's tool list -- which means the model
# sees ``read_file`` first, ``shell_exec`` last. That ordering matches
# the typical reach-for sequence for a code-review/exploration skill
# (read → list → search → shell), keeping the most-used tool earliest
# in the prompt.
BUILTIN_TOOLS: list[Tool] = [
    Tool(
        name="read_file",
        description="Read a file from the project. Returns numbered lines.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read."},
                "offset": {
                    "type": "integer",
                    "description": "Line offset to start from.",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read.",
                    "default": 500,
                },
            },
            "required": ["path"],
        },
        fn=_read_file,
    ),
    Tool(
        name="write_file",
        description="Write content to a file. Creates parent directories.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        fn=_write_file,
    ),
    Tool(
        name="list_files",
        description="List files matching a glob pattern.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "default": "**/*"},
                "path": {"type": "string", "default": "."},
            },
            "required": [],
        },
        fn=_list_files,
    ),
    Tool(
        name="search_files",
        description="Search file contents with ripgrep (falls back to grep).",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "glob": {"type": "string", "default": ""},
            },
            "required": ["pattern"],
        },
        fn=_search_files,
    ),
    Tool(
        name="shell_exec",
        description="Execute a shell command and return stdout + stderr.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number", "default": 60.0},
            },
            "required": ["command"],
        },
        fn=_shell_exec,
    ),
]
