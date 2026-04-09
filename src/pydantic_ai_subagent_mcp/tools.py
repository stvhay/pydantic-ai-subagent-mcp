"""Built-in tools available to subagent runs."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic_ai import RunContext


async def read_file(ctx: RunContext[None], path: str, offset: int = 0, limit: int = 500) -> str:
    """Read a file from the project. Returns numbered lines."""
    try:
        p = Path(path).resolve()
        lines = p.read_text().splitlines()
        selected = lines[offset : offset + limit]
        numbered = [f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)]
        return "\n".join(numbered)
    except Exception as e:
        return f"Error reading {path}: {e}"


async def write_file(ctx: RunContext[None], path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


async def list_files(ctx: RunContext[None], pattern: str = "**/*", path: str = ".") -> str:
    """List files matching a glob pattern."""
    try:
        base = Path(path).resolve()
        matches = sorted(base.glob(pattern))[:200]
        return "\n".join(str(m.relative_to(base)) for m in matches if m.is_file())
    except Exception as e:
        return f"Error listing files: {e}"


async def search_files(
    ctx: RunContext[None], pattern: str, path: str = ".", glob: str = ""
) -> str:
    """Search file contents using ripgrep (or fallback to grep)."""
    try:
        try:
            cmd = ["rg", "--no-heading", "--line-number", "--max-count=50", pattern, path]
            if glob:
                cmd.extend(["--glob", glob])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout[:10000]
        except FileNotFoundError:
            pass  # rg not installed, fall through to grep
        cmd = ["grep", "-rn", "--max-count=50", pattern, path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout[:10000] if result.stdout else "No matches found."
    except Exception as e:
        return f"Error searching: {e}"


async def shell_exec(ctx: RunContext[None], command: str, timeout: float = 60.0) -> str:
    """Execute a shell command and return stdout + stderr."""
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
    except Exception as e:
        return f"Error executing command: {e}"


# Registry of all built-in tools
BUILTIN_TOOLS = [
    read_file,
    write_file,
    list_files,
    search_files,
    shell_exec,
]
