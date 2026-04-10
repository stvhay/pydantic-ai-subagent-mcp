"""Tests for the MCP server module.

The tests mock at the ``OllamaClient.chat_turn`` boundary so that the
real ``run_agent`` loop, the real ``on_content_delta`` callback path,
and the real streaming-log writes all execute end-to-end. This is the
narrowest meaningful seam between *us* and *Ollama*: anything above it
is our middleware code under test, anything below it is the wire format
we don't control.

A ``_FakeOllamaClient`` is installed per-test on ``server._ollama_client``
because the lifespan async context manager that normally creates the real
client is not entered in unit tests. Each test wires up its own canned
``_TurnSpec`` list; the fake pops one spec per ``chat_turn`` call and
streams the chunks through the real callback before returning a real
``TurnResult``. Timing-sensitive tests (mailbox FIFO, semaphore, mid-stream
tail, stop_session) drive concurrency through ``asyncio.Event`` gates
embedded in the spec.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai_subagent_mcp import server
from pydantic_ai_subagent_mcp.config import ServerConfig
from pydantic_ai_subagent_mcp.ollama import TurnResult
from pydantic_ai_subagent_mcp.skills import Skill
from tests._helpers import yield_until

# -- fakes --


@dataclass
class _TurnSpec:
    """One canned chat_turn outcome.

    ``chunks`` are yielded through ``on_content_delta`` in order, exactly
    as the real client would. ``tool_calls`` is the model's tool-call
    array; if non-empty the agent loop will dispatch them and call
    ``chat_turn`` again, popping the next spec.

    Timing hooks: ``on_enter`` runs at the very top of ``chat_turn``
    (before any chunks) and is the test's way to observe in-flight
    state. ``gate`` is awaited just before returning the ``TurnResult``;
    setting it from outside unblocks the call. ``raise_at_start``
    raises before any chunks; ``raise_exc`` raises *after* every chunk
    has been delivered (so the streaming-log writer has flushed them
    to disk by the time the exception fires).
    """

    chunks: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    content: str | None = None
    gate: asyncio.Event | None = None
    on_enter: Callable[[], Awaitable[None] | None] | None = None
    raise_at_start: BaseException | None = None
    raise_exc: BaseException | None = None


class _FakeOllamaClient:
    """Drop-in replacement for ``OllamaClient`` that returns canned turns.

    The fake honours ``on_content_delta`` so the streaming-log writes go
    through the same flush path as production. ``calls`` records every
    ``chat_turn`` invocation (model + the messages list passed in) so
    tests can assert that resume passes the prior history forward.
    """

    def __init__(self, turns: list[_TurnSpec] | None = None) -> None:
        self.turns: list[_TurnSpec] = list(turns or [])
        self.calls: list[dict[str, Any]] = []

    async def chat_turn(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        num_ctx: int = 32768,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> TurnResult:
        self.calls.append({
            "model": model,
            "messages": [dict(m) for m in messages],
            "tools": list(tools or []),
        })
        if not self.turns:
            msg = (
                f"_FakeOllamaClient ran out of canned turns after "
                f"{len(self.calls)} call(s)"
            )
            raise AssertionError(msg)
        spec = self.turns.pop(0)

        if spec.on_enter is not None:
            res = spec.on_enter()
            if res is not None:
                await res

        if spec.raise_at_start is not None:
            raise spec.raise_at_start

        for chunk in spec.chunks:
            if on_content_delta is not None:
                await on_content_delta(chunk)

        if spec.raise_exc is not None:
            raise spec.raise_exc

        if spec.gate is not None:
            await spec.gate.wait()

        return TurnResult(
            content=spec.content if spec.content is not None else "".join(spec.chunks),
            thinking="",
            tool_calls=list(spec.tool_calls),
            prompt_eval_count=10,
            eval_count=5,
            total_duration_ns=1_000_000,
            done_reason="stop",
        )

    async def aclose(self) -> None:
        return None


def _install_fake(*specs: _TurnSpec) -> _FakeOllamaClient:
    """Install a fresh ``_FakeOllamaClient`` on the server module."""
    fake = _FakeOllamaClient(list(specs))
    server._ollama_client = fake
    return fake


def _simple_turn(content: str = "Hello from the agent!") -> _TurnSpec:
    """A no-frills single-chunk terminal turn."""
    return _TurnSpec(chunks=[content])


# -- fixtures --


@pytest.fixture(autouse=True)
async def _reset_server_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Reset server module globals between tests.

    Async fixture so the teardown can ``await`` the SessionStore's
    shutdown -- per-session worker tasks block on ``mailbox.get()``
    between items, and leaving them dangling would emit "Task was
    destroyed but it is pending!" warnings (and bleed across tests
    that share the same module-level singletons).
    """
    monkeypatch.delenv("SUBAGENT_MCP_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("SUBAGENT_MCP_MAX_CONCURRENT_RUNS", raising=False)
    monkeypatch.delenv("SUBAGENT_MCP_MAILBOX_MAX_DEPTH", raising=False)
    server._config = ServerConfig(
        session_dir=str(tmp_path / "sessions"),
        inbox_dir=str(tmp_path / "inbox"),
        mcp_servers_config=str(tmp_path / "missing-mcp-servers.json"),
    )
    server._session_store = None
    server._inbox = None
    server._skills = []
    server._mcp_loader = None
    server._ollama_client = None
    yield
    if server._session_store is not None:
        await server._session_store.shutdown()
    server._ollama_client = None
    server._mcp_loader = None


def _make_skill(tmp_path: Path, name: str = "test-skill") -> Skill:
    """Create a test skill with a temp markdown file."""
    md = tmp_path / f"{name}.md"
    md.write_text(f"# {name}\n\nA test skill.\n")
    return Skill(name=name, description="A test skill", source_path=md)


def _override_backpressure_config(
    tmp_path: Path,
    *,
    max_concurrent_runs: int = 4,
    mailbox_max_depth: int = 16,
) -> None:
    """Reset server singletons with custom backpressure knobs."""
    server._config = ServerConfig(
        session_dir=str(tmp_path / "sessions"),
        inbox_dir=str(tmp_path / "inbox"),
        max_concurrent_runs=max_concurrent_runs,
        mailbox_max_depth=mailbox_max_depth,
    )
    server._session_store = None
    server._inbox = None


# -- _run_skill: happy path, error, resume --


async def test_run_skill_success(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path)
    _install_fake(_simple_turn("Hello from the agent!"))

    result = await server._run_skill(skill, "say hello")

    assert result["response"] == "Hello from the agent!"
    assert result["skill"] == "test-skill"
    assert result["status"] == "idle"
    assert "session_id" in result
    assert "error" not in result


async def test_run_skill_passes_message_history(tmp_path: Path) -> None:
    """Resuming a session feeds the prior messages back into chat_turn."""
    skill = _make_skill(tmp_path)
    fake = _install_fake(
        _simple_turn("Response 1"),
        _simple_turn("Response 2"),
    )

    result1 = await server._run_skill(skill, "first")
    sid = result1["session_id"]
    result2 = await server._run_skill(skill, "second", session_id=sid)

    assert result2["session_id"] == sid
    assert result2["response"] == "Response 2"

    # Second chat_turn was called with the first turn's full history.
    second_messages = fake.calls[1]["messages"]
    roles = [m["role"] for m in second_messages]
    # system + (user + assistant) from turn 1 + new user from turn 2
    assert roles == ["system", "user", "assistant", "user"]
    assert second_messages[1]["content"] == "first"
    assert second_messages[2]["content"] == "Response 1"
    assert second_messages[3]["content"] == "second"


async def test_run_skill_error(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path)
    _install_fake(_TurnSpec(raise_at_start=RuntimeError("Ollama is down")))

    result = await server._run_skill(skill, "say hello")

    assert "error" in result
    assert "Ollama is down" in result["error"]
    assert result["status"] == "failed"
    assert "session_id" in result


async def test_run_skill_resumes_session(tmp_path: Path) -> None:
    """Two sequential foreground runs share the same session id."""
    skill = _make_skill(tmp_path)
    _install_fake(_simple_turn("Response 1"), _simple_turn("Response 2"))

    result1 = await server._run_skill(skill, "first message")
    sid = result1["session_id"]
    result2 = await server._run_skill(skill, "second message", session_id=sid)

    assert result2["session_id"] == sid
    assert result2["response"] == "Response 2"


# -- streaming log + trailers (always-on; no streaming knob) --


async def test_run_skill_writes_log(tmp_path: Path) -> None:
    """Every run writes a header + the streamed deltas + an ok trailer."""
    skill = _make_skill(tmp_path)
    chunks = ["Hello", " from", " the", " agent!"]
    _install_fake(_TurnSpec(chunks=chunks))

    result = await server._run_skill(skill, "say hello")

    assert result["response"] == "Hello from the agent!"
    assert "error" not in result
    log = server._get_session_store().log_path(result["session_id"]).read_text()
    assert "say hello" in log
    assert "--- response ---" in log
    assert "Hello from the agent!" in log
    assert "--- end ok " in log


async def test_run_skill_multi_turn_appends_log(tmp_path: Path) -> None:
    """Each successive turn appends a fresh header/response/trailer block."""
    skill = _make_skill(tmp_path)
    _install_fake(
        _TurnSpec(chunks=["Response", " one"]),
        _TurnSpec(chunks=["Response", " two"]),
    )

    result1 = await server._run_skill(skill, "first")
    sid = result1["session_id"]
    result2 = await server._run_skill(skill, "second", session_id=sid)

    assert result2["session_id"] == sid
    log = server._get_session_store().log_path(sid).read_text()
    assert log.count("--- response ---") == 2
    assert log.count("--- end ok ") == 2
    assert "Response one" in log
    assert "Response two" in log


async def test_run_skill_writes_ok_trailer(tmp_path: Path) -> None:
    """The success trailer is the last non-empty line of the log."""
    skill = _make_skill(tmp_path)
    _install_fake(_TurnSpec(chunks=["Hello", " world"]))

    result = await server._run_skill(skill, "say hi")

    assert "error" not in result
    log = server._get_session_store().log_path(result["session_id"]).read_text()
    last_line = log.rstrip("\n").rsplit("\n", 1)[-1]
    assert last_line.startswith("--- end ok ")
    assert last_line.endswith(" ---")
    assert "--- end error " not in log


async def test_run_skill_writes_error_trailer_and_returns_error(
    tmp_path: Path,
) -> None:
    """A mid-stream exception is recorded in an error trailer.

    Chunks delivered before the failure remain visible in the log; the
    outer ``_run_skill`` catches the re-raised exception and converts
    it into an error response dict.
    """
    skill = _make_skill(tmp_path)
    boom = RuntimeError("ollama vanished")
    _install_fake(_TurnSpec(chunks=["partial"], raise_exc=boom))

    result = await server._run_skill(skill, "trigger failure")

    assert "error" in result
    assert "ollama vanished" in result["error"]
    log = server._get_session_store().log_path(result["session_id"]).read_text()
    assert "partial" in log
    last_line = log.rstrip("\n").rsplit("\n", 1)[-1]
    assert last_line.startswith("--- end error ")
    assert "RuntimeError: ollama vanished" in last_line
    assert last_line.endswith(" ---")
    assert "--- end ok " not in log


async def test_run_skill_error_trailer_sanitizes_newlines(
    tmp_path: Path,
) -> None:
    """Newlines/CRs in the exception message are flattened to spaces."""
    skill = _make_skill(tmp_path)
    boom = ValueError("line one\nline two\rline three")
    _install_fake(_TurnSpec(raise_at_start=boom))

    await server._run_skill(skill, "go")

    log_files = list((tmp_path / "sessions").glob("*.log"))
    assert len(log_files) == 1
    log = log_files[0].read_text()
    trailer = next(
        line for line in log.splitlines() if line.startswith("--- end error ")
    )
    assert "\n" not in trailer
    assert "\r" not in trailer
    assert "line one line two line three" in trailer


async def test_run_skill_cancelled_writes_cancelled_trailer(
    tmp_path: Path,
) -> None:
    """A mid-stream CancelledError writes a 'cancelled' trailer and propagates."""
    skill = _make_skill(tmp_path)
    cancelled = asyncio.CancelledError()
    _install_fake(_TurnSpec(chunks=["partial"], raise_exc=cancelled))

    with pytest.raises(asyncio.CancelledError):
        await server._run_skill(skill, "trigger cancellation")

    log_files = list((tmp_path / "sessions").glob("*.log"))
    assert len(log_files) == 1
    log = log_files[0].read_text()
    assert "partial" in log
    last_line = log.rstrip("\n").rsplit("\n", 1)[-1]
    assert last_line.startswith("--- end cancelled ")
    assert "CancelledError" in last_line
    assert last_line.endswith(" ---")
    assert "--- end ok " not in log
    assert "--- end error " not in log


async def test_run_skill_resumes_from_disk_and_appends_log(tmp_path: Path) -> None:
    """Resuming a session after dropping the in-memory store rehydrates messages."""
    skill = _make_skill(tmp_path)

    # Turn 1
    _install_fake(_TurnSpec(chunks=["Response", " one"]))
    result1 = await server._run_skill(skill, "first")
    sid = result1["session_id"]
    assert "error" not in result1

    # Simulate restart: drop the in-memory store and the client
    await server._get_session_store().shutdown()
    server._session_store = None

    # Turn 2 against a fresh fake
    fake2 = _install_fake(_TurnSpec(chunks=["Response", " two"]))
    result2 = await server._run_skill(skill, "second", session_id=sid)

    assert result2["session_id"] == sid
    assert "error" not in result2

    # The new store rehydrated history from disk and threaded it into chat_turn.
    second_messages = fake2.calls[0]["messages"]
    roles = [m["role"] for m in second_messages]
    assert roles == ["system", "user", "assistant", "user"]

    log = server._get_session_store().log_path(sid).read_text()
    assert log.count("--- response ---") == 2
    assert log.count("--- end ok ") == 2
    assert "first" in log
    assert "second" in log


# -- tail_session_log: live mid-stream observation --


async def test_tail_session_log_reads_mid_stream(tmp_path: Path) -> None:
    """tail_session_log observes flushed deltas while the stream is in flight.

    Validates the per-chunk ``log.flush()`` in ``_run_skill_streaming``.
    Uses a paired-handshake pattern (input queue + ready queue) so the
    test deterministically waits for the write+flush before tailing.
    """
    skill = _make_skill(tmp_path)
    input_q: asyncio.Queue[str | None] = asyncio.Queue()
    ready_q: asyncio.Queue[None] = asyncio.Queue()

    # Build a turn whose on_enter pulls chunks from input_q one at a time
    # and signals ready_q after each chunk has been delivered through the
    # streaming-log writer. The fake's existing chunk loop is too eager
    # for this test (it would dump everything before yielding control),
    # so we drive the deltas ourselves from inside on_enter and skip the
    # spec's chunks list entirely.
    delta_writer: dict[str, Any] = {}

    async def pump_chunks() -> None:
        # The streaming code in server.py defines on_delta as a closure
        # over the open log file; we can't reach it from here directly,
        # so we capture it via a tiny intercept on_content_delta wrapper
        # installed by chat_turn (see below).
        while True:
            chunk = await input_q.get()
            if chunk is None:
                return
            await delta_writer["on_delta"](chunk)
            await ready_q.put(None)

    class _PumpingClient(_FakeOllamaClient):
        async def chat_turn(
            self,
            *,
            model: str,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            num_ctx: int = 32768,
            temperature: float = 1.0,
            top_p: float = 0.95,
            top_k: int = 64,
            on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        ) -> TurnResult:
            self.calls.append({
                "model": model,
                "messages": [dict(m) for m in messages],
                "tools": list(tools or []),
            })
            assert on_content_delta is not None
            delta_writer["on_delta"] = on_content_delta
            await pump_chunks()
            return TurnResult(
                content="hello world",
                thinking="",
                tool_calls=[],
                prompt_eval_count=10,
                eval_count=5,
                total_duration_ns=1,
                done_reason="stop",
            )

    server._ollama_client = _PumpingClient()

    task = asyncio.create_task(server._run_skill(skill, "say hi"))

    # Push the first chunk and wait for it to land on disk
    await input_q.put("hello ")
    await ready_q.get()

    store = server._get_session_store()
    log_files = list(store.session_dir.glob("*.log"))
    assert len(log_files) == 1
    sid = log_files[0].stem

    result = json.loads(await server.tail_session_log(sid, offset=0))
    assert "hello " in result["text"]
    assert "--- response ---\n" in result["text"]
    assert result["next_offset"] > 0
    first_offset = result["next_offset"]

    await input_q.put("world")
    await ready_q.get()

    result2 = json.loads(await server.tail_session_log(sid, offset=first_offset))
    assert "world" in result2["text"]
    assert "hello " not in result2["text"]

    # End the stream cleanly and let the task finish
    await input_q.put(None)
    final = await task

    assert "error" not in final
    assert final["session_id"] == sid
    final_text = store.log_path(sid).read_text()
    assert "--- end ok " in final_text


async def test_tail_session_log_tool_returns_bytes(tmp_path: Path) -> None:
    store = server._get_session_store()
    session = store.create("test-skill", "gemma4:12b")
    log = store.log_path(session.session_id)
    log.write_text("hello world")

    result = json.loads(
        await server.tail_session_log(session.session_id, offset=0)
    )
    assert result["session_id"] == session.session_id
    assert result["text"] == "hello world"
    assert result["next_offset"] == len("hello world")


async def test_tail_session_log_tool_incremental(tmp_path: Path) -> None:
    store = server._get_session_store()
    session = store.create("test-skill", "gemma4:12b")
    log = store.log_path(session.session_id)
    log.write_text("first")

    first = json.loads(
        await server.tail_session_log(session.session_id, offset=0)
    )
    assert first["text"] == "first"
    assert first["next_offset"] == 5

    with log.open("a") as f:
        f.write(" second")

    second = json.loads(
        await server.tail_session_log(
            session.session_id, offset=first["next_offset"]
        )
    )
    assert second["text"] == " second"


async def test_tail_session_log_tool_nonexistent() -> None:
    result = json.loads(await server.tail_session_log("nonexistent-uuid", offset=0))
    assert result["text"] == ""
    assert result["next_offset"] == 0


# -- MCP tool wiring: list_sessions, transcript, skills, by_name --


async def test_list_sessions_tool(tmp_path: Path) -> None:
    store = server._get_session_store()
    store.create("skill-a", "gemma4:12b")

    result = json.loads(await server.list_sessions())
    assert len(result) == 1
    assert result[0]["skill_name"] == "skill-a"


async def test_get_session_transcript_not_found() -> None:
    result = json.loads(await server.get_session_transcript("nonexistent-uuid"))
    assert "error" in result


async def test_list_available_skills(tmp_path: Path) -> None:
    server._skills = [
        _make_skill(tmp_path, "alpha"),
        _make_skill(tmp_path, "beta"),
    ]
    result = json.loads(await server.list_available_skills())
    assert len(result) == 2
    names = {s["name"] for s in result}
    assert names == {"alpha", "beta"}


async def test_run_skill_by_name_not_found() -> None:
    result = json.loads(await server.run_skill_by_name("nonexistent", "hello"))
    assert "error" in result
    assert "not found" in result["error"]


async def test_list_sessions_tool_exposes_status(tmp_path: Path) -> None:
    store = server._get_session_store()
    s = store.create("skill-a", "gemma4:12b")
    s.status = "running"
    s.last_active = "2026-04-08T10:00:00+00:00"
    store.save(s)

    result = json.loads(await server.list_sessions())
    assert len(result) == 1
    assert result[0]["status"] == "running"
    assert result[0]["last_active"] == "2026-04-08T10:00:00+00:00"


# -- _check_ollama health probe --


async def test_check_ollama_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"models": [{"name": "gemma4:12b"}]}
    mock_response.raise_for_status = MagicMock()

    with patch("pydantic_ai_subagent_mcp.server.httpx") as mock_httpx:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_httpx.AsyncClient.return_value = mock_client

        await server._check_ollama(ServerConfig())
        mock_client.get.assert_called_once()


async def test_check_ollama_failure_logs_warning() -> None:
    """Health check warns but does not raise when Ollama is unreachable."""
    with patch("pydantic_ai_subagent_mcp.server.httpx") as mock_httpx:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
        mock_httpx.AsyncClient.return_value = mock_client

        # Must not raise
        await server._check_ollama(ServerConfig())


# -- Status transitions, background, completion --


async def test_run_skill_status_transitions_idle_to_idle(tmp_path: Path) -> None:
    """A successful foreground run leaves the session idle and bumps last_active."""
    skill = _make_skill(tmp_path)
    _install_fake(_simple_turn())

    result = await server._run_skill(skill, "say hello")

    assert result["status"] == "idle"
    store = server._get_session_store()
    session = store.get(result["session_id"])
    assert session is not None
    assert session.status == "idle"
    assert session.last_active >= session.created_at


async def test_run_skill_status_transitions_to_failed_on_error(
    tmp_path: Path,
) -> None:
    """An exception during the agent call leaves the session in 'failed'."""
    skill = _make_skill(tmp_path)
    _install_fake(_TurnSpec(raise_at_start=RuntimeError("ollama down")))

    result = await server._run_skill(skill, "say hello")

    assert "error" in result
    assert result["status"] == "failed"
    store = server._get_session_store()
    session = store.get(result["session_id"])
    assert session is not None
    assert session.status == "failed"


async def test_run_skill_persists_running_status_during_turn(
    tmp_path: Path,
) -> None:
    """While chat_turn is in flight, the on-disk status reads 'running'."""
    skill = _make_skill(tmp_path)
    store = server._get_session_store()
    observed: dict[str, str] = {}

    async def read_status() -> None:
        sessions = store.list_sessions()
        observed["status"] = sessions[0]["status"]

    _install_fake(_TurnSpec(chunks=["ok"], on_enter=read_status))

    await server._run_skill(skill, "go")

    assert observed["status"] == "running"


async def test_run_skill_background_returns_immediately(tmp_path: Path) -> None:
    """A background launch returns before the agent call resolves."""
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()
    _install_fake(_TurnSpec(chunks=["background done"], gate=can_finish))

    result = await asyncio.wait_for(
        server._run_skill(skill, "go", run_in_background=True),
        timeout=1.0,
    )

    assert result["status"] == "running"
    assert "session_id" in result
    assert "log_path" in result
    assert "mailbox_depth" in result
    assert "response" not in result
    sid = result["session_id"]

    store = server._get_session_store()
    assert store.has_worker(sid) is True

    can_finish.set()
    await asyncio.wait_for(store.drain(sid), timeout=1.0)

    session = store.get(sid)
    assert session is not None
    assert session.status == "idle"
    # Native loop builds: system + user + assistant
    assert len(session.messages) == 3


async def test_run_skill_resume_after_completion_succeeds(tmp_path: Path) -> None:
    """Once the in-flight item completes, the session is resumable again."""
    skill = _make_skill(tmp_path)
    _install_fake(_simple_turn("one"), _simple_turn("two"))

    first = await server._run_skill(skill, "first", run_in_background=True)
    sid = first["session_id"]
    store = server._get_session_store()
    await asyncio.wait_for(store.drain(sid), timeout=1.0)

    second = await server._run_skill(skill, "second", session_id=sid)
    assert second["status"] == "idle"
    assert second["session_id"] == sid
    assert second["response"] == "two"


async def test_run_skill_by_name_background(tmp_path: Path) -> None:
    """The MCP tool wrapper passes run_in_background through correctly."""
    skill = _make_skill(tmp_path)
    server._skills = [skill]
    can_finish = asyncio.Event()
    _install_fake(_TurnSpec(chunks=["done"], gate=can_finish))

    result_json = await asyncio.wait_for(
        server.run_skill_by_name(skill.name, "go", run_in_background=True),
        timeout=1.0,
    )
    result = json.loads(result_json)
    assert result["status"] == "running"

    can_finish.set()
    store = server._get_session_store()
    await asyncio.wait_for(store.drain(result["session_id"]), timeout=1.0)


# -- Inbox notifications --


async def test_run_skill_emits_ok_notification_on_success(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path)
    _install_fake(_simple_turn("agent said hi"))

    result = await server._run_skill(skill, "hello")

    inbox = server._get_inbox()
    notifications, _ = inbox.read(since="")
    assert len(notifications) == 1
    n = notifications[0]
    assert n["session_id"] == result["session_id"]
    assert n["status"] == "ok"
    assert n["skill"] == skill.name
    assert n["summary"] == "agent said hi"


async def test_run_skill_emits_error_notification_on_exception(
    tmp_path: Path,
) -> None:
    skill = _make_skill(tmp_path)
    _install_fake(_TurnSpec(raise_at_start=RuntimeError("boom")))

    result = await server._run_skill(skill, "hello")

    inbox = server._get_inbox()
    notifications, _ = inbox.read(since="")
    assert len(notifications) == 1
    n = notifications[0]
    assert n["session_id"] == result["session_id"]
    assert n["status"] == "error"
    assert "boom" in n["summary"]


async def test_run_skill_emits_cancelled_notification_on_basexception(
    tmp_path: Path,
) -> None:
    """CancelledError writes a 'cancelled' notification before propagating."""
    skill = _make_skill(tmp_path)
    cancelled = asyncio.CancelledError()
    _install_fake(_TurnSpec(chunks=["partial"], raise_exc=cancelled))

    with pytest.raises(asyncio.CancelledError):
        await server._run_skill(skill, "go")

    inbox = server._get_inbox()
    notifications, _ = inbox.read(since="")
    assert len(notifications) == 1
    n = notifications[0]
    assert n["status"] == "cancelled"
    assert "CancelledError" in n["summary"]


async def test_inbox_write_failure_does_not_mask_run_result(
    tmp_path: Path,
) -> None:
    """If the inbox write itself raises, the run still returns its response."""
    skill = _make_skill(tmp_path)
    _install_fake(_simple_turn("agent succeeded"))

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("inbox volume full")

    with patch.object(server._get_inbox(), "write", side_effect=boom):
        result = await server._run_skill(skill, "go")

    assert "error" not in result
    assert result["response"] == "agent succeeded"
    assert result["status"] == "idle"


async def test_run_skill_background_emits_notification_on_completion(
    tmp_path: Path,
) -> None:
    """A background run still writes a notification when its task finishes."""
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()
    _install_fake(_TurnSpec(chunks=["background ok"], gate=can_finish))

    first = await server._run_skill(skill, "go", run_in_background=True)
    sid = first["session_id"]
    assert first["status"] == "running"

    inbox = server._get_inbox()
    notifications, _ = inbox.read(since="")
    assert notifications == []

    can_finish.set()
    store = server._get_session_store()
    await asyncio.wait_for(store.drain(sid), timeout=1.0)

    notifications, _ = inbox.read(since="")
    assert len(notifications) == 1
    assert notifications[0]["status"] == "ok"
    assert notifications[0]["session_id"] == sid
    assert notifications[0]["summary"] == "background ok"


async def test_read_inbox_tool_returns_records_and_head(tmp_path: Path) -> None:
    inbox = server._get_inbox()
    n = inbox.write(
        session_id="s1",
        skill="skill",
        model="m",
        status="ok",
        summary="hello",
    )

    result = json.loads(await server.read_inbox())
    assert "notifications" in result
    assert "head" in result
    assert len(result["notifications"]) == 1
    assert result["notifications"][0]["notification_id"] == n.notification_id
    assert result["head"] == n.notification_id


async def test_read_inbox_tool_advances_with_cursor(tmp_path: Path) -> None:
    inbox = server._get_inbox()
    first = inbox.write(
        session_id="s1", skill="k", model="m", status="ok", summary="one"
    )

    initial = json.loads(await server.read_inbox(since=""))
    assert initial["head"] == first.notification_id

    second = json.loads(await server.read_inbox(since=initial["head"]))
    assert second["notifications"] == []
    assert second["head"] == initial["head"]

    second_n = inbox.write(
        session_id="s2", skill="k", model="m", status="error", summary="boom"
    )
    third = json.loads(await server.read_inbox(since=initial["head"]))
    assert len(third["notifications"]) == 1
    assert third["notifications"][0]["notification_id"] == second_n.notification_id
    assert third["head"] == second_n.notification_id


async def test_read_inbox_tool_empty_inbox(tmp_path: Path) -> None:
    result = json.loads(await server.read_inbox())
    assert result["notifications"] == []
    assert result["head"] == ""


# -- Mailbox FIFO + worker semantics --


async def test_resume_busy_session_enqueues_and_runs_in_order(
    tmp_path: Path,
) -> None:
    """Two background pushes to the same session run FIFO."""
    skill = _make_skill(tmp_path)
    can_finish_first = asyncio.Event()
    call_order: list[str] = []

    async def first_enter() -> None:
        call_order.append("first-start")

    async def second_enter() -> None:
        call_order.append("second-start")

    _install_fake(
        _TurnSpec(chunks=["first-done"], on_enter=first_enter, gate=can_finish_first),
        _TurnSpec(chunks=["second-done"], on_enter=second_enter),
    )

    first = await server._run_skill(skill, "go", run_in_background=True)
    sid = first["session_id"]
    assert first["status"] == "running"

    await yield_until(
        lambda: "first-start" in call_order,
        description="worker enters first agent call",
    )

    second = await server._run_skill(
        skill, "again", session_id=sid, run_in_background=True
    )
    assert second["status"] == "queued"
    assert second["session_id"] == sid
    assert second["mailbox_depth"] == 1

    can_finish_first.set()
    store = server._get_session_store()
    await asyncio.wait_for(store.drain(sid), timeout=1.0)

    assert call_order == ["first-start", "second-start"]
    session = store.get(sid)
    assert session is not None
    assert session.status == "idle"
    assert store.mailbox_depth(sid) == 0


async def test_foreground_resume_blocks_until_its_turn(tmp_path: Path) -> None:
    """A foreground resume on a busy session waits for its enqueued item."""
    skill = _make_skill(tmp_path)
    can_finish_first = asyncio.Event()
    _install_fake(
        _TurnSpec(chunks=["first-done"], gate=can_finish_first),
        _TurnSpec(chunks=["second-done"]),
    )

    first = await server._run_skill(skill, "go", run_in_background=True)
    sid = first["session_id"]

    fg_task = asyncio.create_task(
        server._run_skill(skill, "again", session_id=sid)
    )
    for _ in range(5):
        await asyncio.sleep(0)
    assert not fg_task.done()

    can_finish_first.set()
    result = await asyncio.wait_for(fg_task, timeout=1.0)
    assert result["status"] == "idle"
    assert result["response"] == "second-done"


async def test_first_background_run_returns_running_not_queued(
    tmp_path: Path,
) -> None:
    """A push to an idle session reports status='running'."""
    skill = _make_skill(tmp_path)
    _install_fake(_simple_turn())

    result = await server._run_skill(skill, "go", run_in_background=True)
    assert result["status"] == "running"
    assert result["mailbox_depth"] == 1
    store = server._get_session_store()
    await asyncio.wait_for(store.drain(result["session_id"]), timeout=1.0)


async def test_skill_mismatch_on_resume_returns_error(tmp_path: Path) -> None:
    """Resuming a session with a different skill is rejected."""
    skill_a = _make_skill(tmp_path, name="skill-a")
    skill_b = _make_skill(tmp_path, name="skill-b")
    _install_fake(_simple_turn())

    first = await server._run_skill(skill_a, "go")
    sid = first["session_id"]

    second = await server._run_skill(skill_b, "go", session_id=sid)
    assert second["status"] == "skill_mismatch"
    assert "skill-a" in second["error"]
    assert "skill-b" in second["error"]
    # Resume must NOT have run -- only the original turn's messages exist
    store = server._get_session_store()
    session = store.get(sid)
    assert session is not None
    # system + user + assistant from the first turn only
    assert len(session.messages) == 3


async def test_list_sessions_includes_mailbox_depth(tmp_path: Path) -> None:
    """The list_sessions output exposes the per-session mailbox_depth."""
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()
    _install_fake(
        _TurnSpec(chunks=["done"], gate=can_finish),
        _simple_turn("done2"),
    )

    first = await server._run_skill(skill, "go", run_in_background=True)
    sid = first["session_id"]
    await server._run_skill(
        skill, "again", session_id=sid, run_in_background=True
    )

    fake = server._ollama_client
    assert isinstance(fake, _FakeOllamaClient)
    await yield_until(
        lambda: len(fake.calls) >= 1,
        description="worker enters agent call",
    )

    listing = json.loads(await server.list_sessions())
    entry = next(s for s in listing if s["session_id"] == sid)
    assert entry["mailbox_depth"] == 1
    assert entry["status"] == "running"

    can_finish.set()
    store = server._get_session_store()
    await asyncio.wait_for(store.drain(sid), timeout=1.0)


async def test_worker_lazy_started_on_first_push(tmp_path: Path) -> None:
    """A worker is created on the first push and survives subsequent pushes."""
    skill = _make_skill(tmp_path)
    _install_fake(_simple_turn())

    store = server._get_session_store()
    first = await server._run_skill(skill, "one", run_in_background=True)
    sid = first["session_id"]
    assert store.has_worker(sid)
    await asyncio.wait_for(store.drain(sid), timeout=1.0)
    assert store.has_worker(sid)


async def test_shutdown_drains_workers_no_pending_warnings(
    tmp_path: Path,
) -> None:
    """SessionStore.shutdown() must terminate workers cleanly."""
    skill = _make_skill(tmp_path)
    _install_fake(_simple_turn())

    first = await server._run_skill(skill, "go", run_in_background=True)
    sid = first["session_id"]
    store = server._get_session_store()
    await asyncio.wait_for(store.drain(sid), timeout=1.0)
    assert store.has_worker(sid) is True

    await store.shutdown()
    assert store.has_worker(sid) is False


# -- Backpressure (semaphore + mailbox cap) --


async def test_env_var_max_concurrent_runs_reaches_store_semaphore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SUBAGENT_MCP_MAX_CONCURRENT_RUNS flows env -> config -> SessionStore."""
    monkeypatch.setenv("SUBAGENT_MCP_MAX_CONCURRENT_RUNS", "3")
    server._config = None
    server._session_store = None

    store = server._get_session_store()
    # asyncio.Semaphore exposes its current permit count via _value;
    # on a fresh, unacquired semaphore that equals max_concurrent_runs.
    assert store.run_semaphore._value == 3


async def test_mailbox_full_rejects_background_push(tmp_path: Path) -> None:
    """A background push that would exceed mailbox_max_depth is rejected."""
    _override_backpressure_config(tmp_path, mailbox_max_depth=1)
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()
    _install_fake(
        _TurnSpec(chunks=["one"], gate=can_finish),
        _simple_turn("two"),
    )

    first = await server._run_skill(skill, "one", run_in_background=True)
    sid = first["session_id"]
    fake = server._ollama_client
    assert isinstance(fake, _FakeOllamaClient)
    await yield_until(
        lambda: len(fake.calls) >= 1,
        description="worker picks up first item",
    )

    # Item 2 fits (depth 0 -> 1, cap is 1)
    second = await server._run_skill(
        skill, "two", session_id=sid, run_in_background=True
    )
    assert second["status"] == "queued"

    # Item 3 must be rejected
    third = await server._run_skill(
        skill, "three", session_id=sid, run_in_background=True
    )
    assert third["status"] == "mailbox_full"
    assert third["session_id"] == sid
    assert third["mailbox_depth"] == 1
    assert third["mailbox_max_depth"] == 1
    assert "mailbox is full" in third["error"]

    store = server._get_session_store()
    assert store.mailbox_depth(sid) == 1

    can_finish.set()
    await asyncio.wait_for(store.drain(sid), timeout=1.0)


async def test_mailbox_full_rejects_foreground_push(tmp_path: Path) -> None:
    """Foreground pushes also respect the mailbox cap."""
    _override_backpressure_config(tmp_path, mailbox_max_depth=1)
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()
    _install_fake(
        _TurnSpec(chunks=["one"], gate=can_finish),
        _simple_turn("two"),
    )

    first = await server._run_skill(skill, "one", run_in_background=True)
    sid = first["session_id"]
    fake = server._ollama_client
    assert isinstance(fake, _FakeOllamaClient)
    await yield_until(
        lambda: len(fake.calls) >= 1,
        description="worker picks up first item",
    )

    await server._run_skill(
        skill, "two", session_id=sid, run_in_background=True
    )

    result = await asyncio.wait_for(
        server._run_skill(skill, "three", session_id=sid),
        timeout=1.0,
    )
    assert result["status"] == "mailbox_full"
    assert result["mailbox_depth"] == 1

    can_finish.set()
    store = server._get_session_store()
    await asyncio.wait_for(store.drain(sid), timeout=1.0)


async def test_foreground_blocks_when_semaphore_saturated(
    tmp_path: Path,
) -> None:
    """Foreground launches DO NOT reject on saturation -- they wait."""
    _override_backpressure_config(tmp_path, max_concurrent_runs=1)
    skill = _make_skill(tmp_path)
    can_finish_a = asyncio.Event()
    _install_fake(
        _TurnSpec(chunks=["a-done"], gate=can_finish_a),
        _simple_turn("b-done"),
    )

    a = await server._run_skill(skill, "go", run_in_background=True)
    a_sid = a["session_id"]
    store = server._get_session_store()
    await yield_until(
        lambda: store.run_semaphore.locked(),
        description="session A acquires the run semaphore",
    )

    b_task = asyncio.create_task(server._run_skill(skill, "go"))
    for _ in range(5):
        await asyncio.sleep(0)
    assert not b_task.done()

    can_finish_a.set()
    b_result = await asyncio.wait_for(b_task, timeout=1.0)
    assert b_result["status"] == "idle"
    assert b_result["response"] == "b-done"

    await asyncio.wait_for(store.drain(a_sid), timeout=1.0)
    await asyncio.wait_for(store.drain(b_result["session_id"]), timeout=1.0)


async def test_semaphore_releases_after_turn_completes(tmp_path: Path) -> None:
    """A completed turn releases its semaphore slot for the next caller."""
    _override_backpressure_config(tmp_path, max_concurrent_runs=1)
    skill = _make_skill(tmp_path)
    _install_fake(_simple_turn(), _simple_turn())

    first = await server._run_skill(skill, "one", run_in_background=True)
    store = server._get_session_store()
    await asyncio.wait_for(store.drain(first["session_id"]), timeout=1.0)

    assert not store.run_semaphore.locked()

    second = await server._run_skill(skill, "two", run_in_background=True)
    assert second["status"] in ("running", "queued")
    await asyncio.wait_for(store.drain(second["session_id"]), timeout=1.0)


# -- stop_session --


async def test_stop_session_tool_not_found(tmp_path: Path) -> None:
    result = json.loads(await server.stop_session("never-existed"))
    assert result["status"] == "not_found"
    assert result["session_id"] == "never-existed"
    assert result["in_flight_cancelled"] == 0
    assert result["queued_dropped"] == 0


async def test_stop_session_tool_already_idle(tmp_path: Path) -> None:
    store = server._get_session_store()
    session = store.create("skill", "gemma4:12b")

    result = json.loads(await server.stop_session(session.session_id))
    assert result["status"] == "already_idle"


async def test_stop_session_tool_cancels_inflight_run(tmp_path: Path) -> None:
    """Stopping a session with an in-flight run cancels it."""
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()
    _install_fake(_TurnSpec(chunks=["partial"], gate=can_finish))

    first = await server._run_skill(skill, "go", run_in_background=True)
    sid = first["session_id"]
    fake = server._ollama_client
    assert isinstance(fake, _FakeOllamaClient)
    await yield_until(
        lambda: len(fake.calls) >= 1,
        description="worker enters agent call",
    )

    result = json.loads(await server.stop_session(sid))

    assert result["status"] == "stopped"
    assert result["in_flight_cancelled"] == 1
    assert result["queued_dropped"] == 0

    inbox = server._get_inbox()
    notifications, _ = inbox.read(since="")
    assert len(notifications) == 1
    assert notifications[0]["status"] == "cancelled"
    assert notifications[0]["session_id"] == sid

    store = server._get_session_store()
    reloaded = store.get(sid)
    assert reloaded is not None
    assert reloaded.status == "failed"

    assert store.has_worker(sid) is False
    assert not store.run_semaphore.locked()


async def test_stop_session_tool_drops_queued_items(tmp_path: Path) -> None:
    """Queued items are dropped from the mailbox without running."""
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()
    _install_fake(
        _TurnSpec(chunks=["one"], gate=can_finish),
        _simple_turn("two"),
        _simple_turn("three"),
    )

    first = await server._run_skill(skill, "one", run_in_background=True)
    sid = first["session_id"]
    fake = server._ollama_client
    assert isinstance(fake, _FakeOllamaClient)
    await yield_until(
        lambda: len(fake.calls) >= 1,
        description="worker enters first agent call",
    )

    await server._run_skill(skill, "two", session_id=sid, run_in_background=True)
    await server._run_skill(skill, "three", session_id=sid, run_in_background=True)

    store = server._get_session_store()
    assert store.mailbox_depth(sid) == 2

    result = json.loads(await server.stop_session(sid))

    assert result["status"] == "stopped"
    assert result["in_flight_cancelled"] == 1
    assert result["queued_dropped"] == 2
    # Only the in-flight item ran -- the queued ones were dropped
    assert len(fake.calls) == 1


async def test_stop_session_unblocks_foreground_waiter(tmp_path: Path) -> None:
    """A foreground caller waiting on a queued item gets cancelled, not hung."""
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()
    _install_fake(
        _TurnSpec(chunks=["first"], gate=can_finish),
        _simple_turn("second"),
    )

    first = await server._run_skill(skill, "go", run_in_background=True)
    sid = first["session_id"]
    fake = server._ollama_client
    assert isinstance(fake, _FakeOllamaClient)
    await yield_until(
        lambda: len(fake.calls) >= 1,
        description="worker enters agent call",
    )

    fg_task = asyncio.create_task(
        server._run_skill(skill, "again", session_id=sid)
    )
    for _ in range(5):
        await asyncio.sleep(0)
    assert not fg_task.done()

    await server.stop_session(sid)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(fg_task, timeout=1.0)


async def test_stop_session_is_idempotent_via_tool(tmp_path: Path) -> None:
    """Calling stop twice in a row is safe."""
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()
    _install_fake(_TurnSpec(chunks=["one"], gate=can_finish))

    first = await server._run_skill(skill, "go", run_in_background=True)
    sid = first["session_id"]
    fake = server._ollama_client
    assert isinstance(fake, _FakeOllamaClient)
    await yield_until(
        lambda: len(fake.calls) >= 1,
        description="worker enters agent call",
    )

    result1 = json.loads(await server.stop_session(sid))
    result2 = json.loads(await server.stop_session(sid))

    assert result1["status"] == "stopped"
    assert result2["status"] == "already_idle"
    assert result2["in_flight_cancelled"] == 0
    assert result2["queued_dropped"] == 0


async def test_stop_session_followed_by_resume(tmp_path: Path) -> None:
    """After stop, the session can be resumed and runs cleanly."""
    skill = _make_skill(tmp_path)
    first_can_finish = asyncio.Event()
    _install_fake(
        _TurnSpec(chunks=["first"], gate=first_can_finish),
        _simple_turn("second"),
    )

    first = await server._run_skill(skill, "go", run_in_background=True)
    sid = first["session_id"]
    fake = server._ollama_client
    assert isinstance(fake, _FakeOllamaClient)
    await yield_until(
        lambda: len(fake.calls) >= 1,
        description="worker enters first agent call",
    )

    await server.stop_session(sid)

    result = await server._run_skill(skill, "again", session_id=sid)

    assert result["session_id"] == sid
    assert result["status"] == "idle"
    assert result["response"] == "second"


async def test_mailbox_full_does_not_apply_to_skill_mismatch(
    tmp_path: Path,
) -> None:
    """Skill-mismatch check still wins even on a full mailbox."""
    _override_backpressure_config(tmp_path, mailbox_max_depth=1)
    skill_a = _make_skill(tmp_path, name="skill-a")
    skill_b = _make_skill(tmp_path, name="skill-b")
    can_finish = asyncio.Event()
    _install_fake(
        _TurnSpec(chunks=["one"], gate=can_finish),
        _simple_turn("two"),
    )

    first = await server._run_skill(skill_a, "go", run_in_background=True)
    sid = first["session_id"]
    fake = server._ollama_client
    assert isinstance(fake, _FakeOllamaClient)
    await yield_until(
        lambda: len(fake.calls) >= 1,
        description="worker enters first agent call",
    )

    await server._run_skill(
        skill_a, "again", session_id=sid, run_in_background=True
    )

    wrong = await server._run_skill(
        skill_b, "go", session_id=sid, run_in_background=True
    )
    assert wrong["status"] == "skill_mismatch"

    can_finish.set()
    store = server._get_session_store()
    await asyncio.wait_for(store.drain(sid), timeout=1.0)
