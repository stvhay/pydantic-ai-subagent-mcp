"""Tests for built-in subagent tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai import RunContext

from pydantic_ai_subagent_mcp.tools import (
    list_files,
    read_file,
    search_files,
    shell_exec,
    web_search,
    write_file,
)

CTX: RunContext[None] = MagicMock(spec=RunContext)


async def test_read_file(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("line one\nline two\nline three\n")
    result = await read_file(CTX, str(f))
    assert "1\tline one" in result
    assert "2\tline two" in result
    assert "3\tline three" in result


async def test_read_file_with_offset_and_limit(tmp_path: Path) -> None:
    f = tmp_path / "lines.txt"
    f.write_text("\n".join(f"line {i}" for i in range(1, 11)))
    result = await read_file(CTX, str(f), offset=2, limit=3)
    assert "3\tline 3" in result
    assert "4\tline 4" in result
    assert "5\tline 5" in result
    assert "1\tline 1" not in result


async def test_read_file_nonexistent() -> None:
    result = await read_file(CTX, "/nonexistent/path/file.txt")
    assert "Error reading" in result


async def test_write_file(tmp_path: Path) -> None:
    target = tmp_path / "subdir" / "output.txt"
    result = await write_file(CTX, str(target), "hello world")
    assert "Written" in result
    assert target.read_text() == "hello world"


async def test_list_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("c")
    result = await list_files(CTX, "**/*.py", str(tmp_path))
    assert "a.py" in result
    assert "c.py" in result
    assert "b.txt" not in result


async def test_search_files(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("def hello_world():\n    pass\n")
    result = await search_files(CTX, "hello_world", str(tmp_path))
    assert "hello_world" in result


async def test_shell_exec() -> None:
    result = await shell_exec(CTX, "echo hello")
    assert "hello" in result


async def test_shell_exec_failure() -> None:
    result = await shell_exec(CTX, "exit 42")
    assert "[exit code: 42]" in result


async def test_shell_exec_timeout() -> None:
    result = await shell_exec(CTX, "sleep 10", timeout=0.1)
    assert "timed out" in result


async def test_web_search() -> None:
    result = await web_search(CTX, "test query")
    assert "not yet configured" in result
