"""Tests for the MCP server module."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import TracebackType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from pydantic_ai_subagent_mcp import server
from pydantic_ai_subagent_mcp.config import ServerConfig
from pydantic_ai_subagent_mcp.skills import Skill


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
    monkeypatch.delenv("SUBAGENT_MCP_STREAMING", raising=False)
    server._config = ServerConfig(
        session_dir=str(tmp_path / "sessions"),
        inbox_dir=str(tmp_path / "inbox"),
        streaming=False,
    )
    server._session_store = None
    server._inbox = None
    server._skills = []
    yield
    if server._session_store is not None:
        await server._session_store.shutdown()


def _make_skill(tmp_path: Path, name: str = "test-skill") -> Skill:
    """Create a test skill with a temp markdown file."""
    md = tmp_path / f"{name}.md"
    md.write_text(f"# {name}\n\nA test skill.\n")
    return Skill(name=name, description="A test skill", source_path=md)


def _override_streaming_config(tmp_path: Path) -> None:
    """Switch the server config into streaming mode for one test.

    Resets every server-level singleton (config, session store, inbox)
    so each call yields a fresh, fully isolated state pointed at
    ``tmp_path``. Tests that need streaming mode should call this in
    place of mutating ``server._config`` directly — without resetting
    ``_inbox``, completion notifications would leak into the real
    ``.subagent-inbox/`` directory in the project root.
    """
    server._config = ServerConfig(
        session_dir=str(tmp_path / "sessions"),
        inbox_dir=str(tmp_path / "inbox"),
        streaming=True,
    )
    server._session_store = None
    server._inbox = None


def _make_mock_result(output: str = "Hello from the agent!") -> MagicMock:
    """Create a mock AgentRunResult with all_messages() support."""
    mock_result = MagicMock()
    mock_result.output = output
    mock_result.all_messages.return_value = [
        ModelRequest(parts=[UserPromptPart(content="test prompt")]),
        ModelResponse(parts=[TextPart(content=output)]),
    ]
    return mock_result


def test_build_model() -> None:
    model = server._build_model("gemma4:12b")
    assert model.model_name == "gemma4:12b"


def test_build_model_uses_config_default() -> None:
    model = server._build_model()
    assert model.model_name == server._get_config().default_model


def test_build_agent(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path)
    agent = server._build_agent(skill)
    assert agent is not None


async def test_run_skill_success(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path)
    mock_result = _make_mock_result()

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_build.return_value = mock_agent

        result = await server._run_skill(skill, "say hello")

    assert result["response"] == "Hello from the agent!"
    assert result["skill"] == "test-skill"
    assert "session_id" in result
    assert "error" not in result

    # Verify agent.run was called with message_history
    mock_agent.run.assert_called_once()
    call_kwargs = mock_agent.run.call_args
    assert call_kwargs[0][0] == "say hello"  # positional prompt arg


async def test_run_skill_passes_message_history(tmp_path: Path) -> None:
    """Verify that resuming a session passes native message_history."""
    skill = _make_skill(tmp_path)

    # First call — no history
    mock_result1 = _make_mock_result("Response 1")
    # Second call — should have history from first call
    mock_result2 = _make_mock_result("Response 2")
    mock_result2.all_messages.return_value = [
        ModelRequest(parts=[UserPromptPart(content="first")]),
        ModelResponse(parts=[TextPart(content="Response 1")]),
        ModelRequest(parts=[UserPromptPart(content="second")]),
        ModelResponse(parts=[TextPart(content="Response 2")]),
    ]

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=[mock_result1, mock_result2])
        mock_build.return_value = mock_agent

        result1 = await server._run_skill(skill, "first")
        session_id = result1["session_id"]

        result2 = await server._run_skill(skill, "second", session_id=session_id)

    assert result2["session_id"] == session_id
    assert result2["response"] == "Response 2"

    # Second call should have received message_history
    second_call = mock_agent.run.call_args_list[1]
    assert second_call.kwargs.get("message_history") is not None
    history = second_call.kwargs["message_history"]
    assert len(history) == 2  # request + response from first call


async def test_run_skill_error(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path)

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Ollama is down"))
        mock_build.return_value = mock_agent

        result = await server._run_skill(skill, "say hello")

    assert "error" in result
    assert "Ollama is down" in result["error"]
    assert "session_id" in result


async def test_run_skill_resumes_session(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path)
    mock_result = _make_mock_result("Response 1")

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_build.return_value = mock_agent

        result1 = await server._run_skill(skill, "first message")
        session_id = result1["session_id"]

        mock_result2 = _make_mock_result("Response 2")
        mock_agent.run = AsyncMock(return_value=mock_result2)
        result2 = await server._run_skill(
            skill, "second message", session_id=session_id
        )

    assert result2["session_id"] == session_id


async def test_list_sessions_tool(tmp_path: Path) -> None:
    store = server._get_session_store()
    store.create("skill-a", "gemma4:12b")

    result_json = await server.list_sessions()
    result = json.loads(result_json)
    assert len(result) == 1
    assert result[0]["skill_name"] == "skill-a"


async def test_get_session_transcript_not_found() -> None:
    result_json = await server.get_session_transcript("nonexistent-uuid")
    result = json.loads(result_json)
    assert "error" in result


async def test_list_available_skills(tmp_path: Path) -> None:
    server._skills = [_make_skill(tmp_path, "alpha"), _make_skill(tmp_path, "beta")]
    result_json = await server.list_available_skills()
    result = json.loads(result_json)
    assert len(result) == 2
    names = {s["name"] for s in result}
    assert names == {"alpha", "beta"}


async def test_run_skill_by_name_not_found() -> None:
    result_json = await server.run_skill_by_name("nonexistent", "hello")
    result = json.loads(result_json)
    assert "error" in result
    assert "not found" in result["error"]


async def test_check_ollama_success() -> None:
    """Test health check logs success when Ollama is reachable."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"models": [{"name": "gemma4:12b"}]}
    mock_response.raise_for_status = MagicMock()

    with patch("pydantic_ai_subagent_mcp.server.httpx") as mock_httpx:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_httpx.AsyncClient.return_value = mock_client

        config = ServerConfig()
        await server._check_ollama(config)

        mock_client.get.assert_called_once()


async def test_check_ollama_failure_logs_warning() -> None:
    """Test health check logs warning but doesn't raise when Ollama is unreachable."""
    with patch("pydantic_ai_subagent_mcp.server.httpx") as mock_httpx:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
        mock_httpx.AsyncClient.return_value = mock_client

        config = ServerConfig()
        # Should not raise
        await server._check_ollama(config)


# -- Streaming tests --


class _FakeStreamResult:
    """Minimal fake for pydantic-ai's StreamedRunResult."""

    def __init__(self, chunks: list[str], messages: list[ModelMessage]) -> None:
        self._chunks = chunks
        self._messages = messages
        self._output = "".join(chunks)

    async def stream_text(self, *, delta: bool = False) -> AsyncIterator[str]:
        assert delta is True  # production code always calls with delta=True
        for chunk in self._chunks:
            yield chunk

    async def get_output(self) -> str:
        return self._output

    def all_messages(self) -> list[ModelMessage]:
        return self._messages


class _FakeStreamCtx:
    """Async context manager wrapper for _FakeStreamResult."""

    def __init__(self, result: _FakeStreamResult) -> None:
        self._result = result

    async def __aenter__(self) -> _FakeStreamResult:
        return self._result

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def _make_fake_stream(chunks: list[str], output: str) -> _FakeStreamCtx:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="test prompt")]),
        ModelResponse(parts=[TextPart(content=output)]),
    ]
    return _FakeStreamCtx(_FakeStreamResult(chunks, messages))


class _FailingStreamResult:
    """Fake StreamedRunResult whose stream_text raises mid-iteration."""

    def __init__(
        self, chunks_before_error: list[str], exc: BaseException
    ) -> None:
        self._chunks = chunks_before_error
        self._exc = exc

    async def stream_text(self, *, delta: bool = False) -> AsyncIterator[str]:
        assert delta is True
        for chunk in self._chunks:
            yield chunk
        raise self._exc

    async def get_output(self) -> str:  # pragma: no cover — never reached
        raise AssertionError("get_output should not be called after failure")

    def all_messages(self) -> list[ModelMessage]:  # pragma: no cover
        raise AssertionError("all_messages should not be called after failure")


def _make_failing_stream(
    chunks_before: list[str], exc: BaseException
) -> _FakeStreamCtx:
    return _FakeStreamCtx(_FailingStreamResult(chunks_before, exc))  # type: ignore[arg-type]


async def test_run_skill_streaming_writes_log(tmp_path: Path) -> None:
    """With streaming=True, _run_skill writes deltas to the session log file."""
    _override_streaming_config(tmp_path)

    skill = _make_skill(tmp_path)
    chunks = ["Hello", " from", " the", " agent!"]
    expected_output = "Hello from the agent!"

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = MagicMock()
        mock_agent.run_stream = MagicMock(
            return_value=_make_fake_stream(chunks, expected_output)
        )
        mock_build.return_value = mock_agent

        result = await server._run_skill(skill, "say hello")

    assert result["response"] == expected_output
    assert result["skill"] == "test-skill"
    assert "error" not in result

    store = server._get_session_store()
    log_path = store.log_path(result["session_id"])
    assert log_path.exists()
    log_text = log_path.read_text()
    assert expected_output in log_text
    assert "--- response ---" in log_text
    assert "say hello" in log_text
    assert "--- end ok " in log_text  # trailer present on success


async def test_run_skill_streaming_multi_turn_appends(tmp_path: Path) -> None:
    """Each turn appends a new prompt/response block to the same log."""
    _override_streaming_config(tmp_path)

    skill = _make_skill(tmp_path)

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = MagicMock()
        mock_agent.run_stream = MagicMock(
            side_effect=[
                _make_fake_stream(["Response", " one"], "Response one"),
                _make_fake_stream(["Response", " two"], "Response two"),
            ]
        )
        mock_build.return_value = mock_agent

        result1 = await server._run_skill(skill, "first")
        session_id = result1["session_id"]
        result2 = await server._run_skill(skill, "second", session_id=session_id)

    assert result2["session_id"] == session_id

    store = server._get_session_store()
    log_text = store.log_path(session_id).read_text()
    assert log_text.count("--- response ---") == 2
    assert log_text.count("--- end ok ") == 2  # one trailer per turn
    assert "first" in log_text
    assert "second" in log_text
    assert "Response one" in log_text
    assert "Response two" in log_text


async def test_run_skill_streaming_disabled_creates_no_log(tmp_path: Path) -> None:
    """With streaming=False, no .log file is created."""
    # Fixture already sets streaming=False
    skill = _make_skill(tmp_path)
    mock_result = _make_mock_result()

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_build.return_value = mock_agent

        result = await server._run_skill(skill, "say hello")

    # Verify the happy path completed
    assert result["response"] == "Hello from the agent!"
    assert "error" not in result

    # No .log file created for this session, and none anywhere in session_dir
    store = server._get_session_store()
    log_path = store.log_path(result["session_id"])
    assert not log_path.exists()
    assert list(store.session_dir.glob("*.log")) == []


async def test_run_skill_streaming_writes_ok_trailer(tmp_path: Path) -> None:
    """Successful streaming writes `--- end ok {ts} ---` trailer as final line.

    Tests issue #8 acceptance criterion 1.
    """
    _override_streaming_config(tmp_path)

    skill = _make_skill(tmp_path)
    chunks = ["Hello", " world"]

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = MagicMock()
        mock_agent.run_stream = MagicMock(
            return_value=_make_fake_stream(chunks, "Hello world")
        )
        mock_build.return_value = mock_agent

        result = await server._run_skill(skill, "say hi")

    assert "error" not in result
    log_text = server._get_session_store().log_path(result["session_id"]).read_text()

    # Trailer is the last non-empty line of the log
    last_line = log_text.rstrip("\n").rsplit("\n", 1)[-1]
    assert last_line.startswith("--- end ok "), f"unexpected last line: {last_line!r}"
    assert last_line.endswith(" ---"), f"unexpected last line: {last_line!r}"
    assert "--- end error " not in log_text


async def test_run_skill_streaming_writes_error_trailer_and_propagates(
    tmp_path: Path,
) -> None:
    """A mid-stream exception is recorded in an error trailer.

    The exception surfaces via _run_skill's error response.
    Tests issue #8 acceptance criterion 2.
    """
    _override_streaming_config(tmp_path)

    skill = _make_skill(tmp_path)
    boom = RuntimeError("ollama vanished")

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = MagicMock()
        mock_agent.run_stream = MagicMock(
            return_value=_make_failing_stream(["partial"], boom)
        )
        mock_build.return_value = mock_agent

        # Outer _run_skill catches the re-raised exception and returns an
        # error dict — we get the error response, not an exception out of _run_skill.
        result = await server._run_skill(skill, "trigger failure")

    assert "error" in result
    assert "ollama vanished" in result["error"]

    log_text = server._get_session_store().log_path(result["session_id"]).read_text()
    assert "partial" in log_text  # the chunk before the failure was flushed
    last_line = log_text.rstrip("\n").rsplit("\n", 1)[-1]
    assert last_line.startswith("--- end error "), f"unexpected last line: {last_line!r}"
    assert "RuntimeError: ollama vanished" in last_line
    assert last_line.endswith(" ---"), f"unexpected last line: {last_line!r}"
    assert "--- end ok " not in log_text


async def test_run_skill_streaming_error_trailer_sanitizes_newlines(
    tmp_path: Path,
) -> None:
    """Newlines and carriage returns in exception messages are replaced.

    Ensures the trailer stays on a single line.
    """
    _override_streaming_config(tmp_path)

    skill = _make_skill(tmp_path)
    boom = ValueError("line one\nline two\rline three")

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = MagicMock()
        mock_agent.run_stream = MagicMock(
            return_value=_make_failing_stream([], boom)
        )
        mock_build.return_value = mock_agent

        await server._run_skill(skill, "go")

    log_files = list((tmp_path / "sessions").glob("*.log"))
    assert len(log_files) == 1, f"expected exactly one log file, got {log_files}"
    log_text = log_files[0].read_text()

    trailer_line = next(
        line for line in log_text.splitlines() if line.startswith("--- end error ")
    )
    # splitlines() already guarantees no \n in the line, but be explicit:
    assert "\n" not in trailer_line
    assert "\r" not in trailer_line
    assert "line one line two line three" in trailer_line


async def test_run_skill_streaming_cancelled_writes_cancelled_trailer(
    tmp_path: Path,
) -> None:
    """A cancelled stream writes an `--- end cancelled ---` trailer.

    Tests issue #8 follow-up: CancelledError is BaseException, not Exception,
    so it propagates past _run_skill's outer error handler. The trailer must
    still be written before the exception escapes _run_skill_streaming.
    """
    _override_streaming_config(tmp_path)

    skill = _make_skill(tmp_path)
    cancelled = asyncio.CancelledError()

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = MagicMock()
        mock_agent.run_stream = MagicMock(
            return_value=_make_failing_stream(["partial"], cancelled)
        )
        mock_build.return_value = mock_agent

        # CancelledError is BaseException, so it propagates past _run_skill's
        # `except Exception` and out of _run_skill entirely.
        with pytest.raises(asyncio.CancelledError):
            await server._run_skill(skill, "trigger cancellation")

    # Find the log file (we don't have a session_id from a returned dict)
    log_files = list((tmp_path / "sessions").glob("*.log"))
    assert len(log_files) == 1, f"expected exactly one log file, got {log_files}"
    log_text = log_files[0].read_text()

    assert "partial" in log_text  # the chunk before cancellation was flushed
    last_line = log_text.rstrip("\n").rsplit("\n", 1)[-1]
    assert last_line.startswith("--- end cancelled "), (
        f"unexpected last line: {last_line!r}"
    )
    assert "CancelledError" in last_line
    assert last_line.endswith(" ---"), f"unexpected last line: {last_line!r}"
    assert "--- end ok " not in log_text
    assert "--- end error " not in log_text


async def test_tail_session_log_tool_returns_bytes(tmp_path: Path) -> None:
    """tail_session_log tool reads from the session log at a given offset."""
    store = server._get_session_store()
    session = store.create("test-skill", "gemma4:12b")
    log = store.log_path(session.session_id)
    log.write_text("hello world")

    result_json = await server.tail_session_log(session.session_id, offset=0)
    result = json.loads(result_json)
    assert result["session_id"] == session.session_id
    assert result["text"] == "hello world"
    assert result["next_offset"] == len("hello world")


async def test_tail_session_log_tool_incremental(tmp_path: Path) -> None:
    """Feeding next_offset back returns only the newly appended content."""
    store = server._get_session_store()
    session = store.create("test-skill", "gemma4:12b")
    log = store.log_path(session.session_id)
    log.write_text("first")

    first = json.loads(await server.tail_session_log(session.session_id, offset=0))
    assert first["text"] == "first"
    assert first["next_offset"] == 5

    with log.open("a") as f:
        f.write(" second")

    second = json.loads(
        await server.tail_session_log(session.session_id, offset=first["next_offset"])
    )
    assert second["text"] == " second"


async def test_tail_session_log_tool_nonexistent() -> None:
    """Tailing a nonexistent session returns empty text and offset 0."""
    result_json = await server.tail_session_log("nonexistent-uuid", offset=0)
    result = json.loads(result_json)
    assert result["text"] == ""
    assert result["next_offset"] == 0


# -- Phase 2: background execution, status transitions, busy reject --


async def test_run_skill_status_transitions_idle_to_idle(tmp_path: Path) -> None:
    """A successful foreground run leaves the session idle and bumps last_active."""
    skill = _make_skill(tmp_path)
    mock_result = _make_mock_result()

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_build.return_value = mock_agent

        result = await server._run_skill(skill, "say hello")

    assert result["status"] == "idle"
    store = server._get_session_store()
    session = store.get(result["session_id"])
    assert session is not None
    assert session.status == "idle"
    # last_active should have advanced past created_at after the turn
    assert session.last_active >= session.created_at


async def test_run_skill_status_transitions_to_failed_on_error(
    tmp_path: Path,
) -> None:
    """An exception during the agent call leaves the session in 'failed'."""
    skill = _make_skill(tmp_path)

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("ollama down"))
        mock_build.return_value = mock_agent

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
    """While the agent call is in flight, the on-disk status reads 'running'.

    Verifies that observers reading list_sessions or session JSON during a
    turn see the in-flight state, not the stale prior status.
    """
    skill = _make_skill(tmp_path)
    store = server._get_session_store()
    observed: dict[str, str] = {}

    async def slow_run(*_a: object, **_k: object) -> MagicMock:
        # Read the on-disk session JSON from inside the agent call to
        # verify it shows status=running
        sessions = store.list_sessions()
        observed["status"] = sessions[0]["status"]
        return _make_mock_result()

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=slow_run)
        mock_build.return_value = mock_agent

        await server._run_skill(skill, "go")

    assert observed["status"] == "running"


async def test_run_skill_background_returns_immediately(tmp_path: Path) -> None:
    """A background launch returns before the agent call resolves."""
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()

    async def blocking_run(*_a: object, **_k: object) -> MagicMock:
        await can_finish.wait()
        return _make_mock_result("background done")

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=blocking_run)
        mock_build.return_value = mock_agent

        # The background launch must return promptly even though the agent
        # call is still blocked. Bound it with wait_for so a regression
        # would fail with TimeoutError instead of hanging the suite.
        result = await asyncio.wait_for(
            server._run_skill(skill, "go", run_in_background=True),
            timeout=1.0,
        )

        assert result["status"] == "running"
        assert "session_id" in result
        assert "log_path" in result
        assert "mailbox_depth" in result
        assert "response" not in result  # response only on completion
        session_id = result["session_id"]

        # The store should have a worker drain the mailbox
        store = server._get_session_store()
        assert store.has_worker(session_id) is True

        # Let the worker finish processing this item
        can_finish.set()
        await asyncio.wait_for(store.drain(session_id), timeout=1.0)

        session = store.get(session_id)
        assert session is not None
        assert session.status == "idle"
        # The completed run wrote messages back to the session
        assert len(session.messages) == 2


async def test_run_skill_resume_after_completion_succeeds(tmp_path: Path) -> None:
    """Once the in-flight item completes, the session is resumable again."""
    skill = _make_skill(tmp_path)

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            side_effect=[_make_mock_result("one"), _make_mock_result("two")]
        )
        mock_build.return_value = mock_agent

        first = await server._run_skill(
            skill, "first", run_in_background=True
        )
        session_id = first["session_id"]

        # Drain the background work
        store = server._get_session_store()
        await asyncio.wait_for(store.drain(session_id), timeout=1.0)

        # Now a resume should succeed (foreground)
        second = await server._run_skill(
            skill, "second", session_id=session_id
        )
        assert second["status"] == "idle"
        assert second["session_id"] == session_id
        assert second["response"] == "two"


async def test_run_skill_by_name_background(tmp_path: Path) -> None:
    """The MCP tool wrapper passes run_in_background through correctly."""
    skill = _make_skill(tmp_path)
    server._skills = [skill]
    can_finish = asyncio.Event()

    async def blocking_run(*_a: object, **_k: object) -> MagicMock:
        await can_finish.wait()
        return _make_mock_result()

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=blocking_run)
        mock_build.return_value = mock_agent

        result_json = await asyncio.wait_for(
            server.run_skill_by_name(
                skill.name, "go", run_in_background=True
            ),
            timeout=1.0,
        )
        result = json.loads(result_json)
        assert result["status"] == "running"

        # Drain
        can_finish.set()
        store = server._get_session_store()
        await asyncio.wait_for(
            store.drain(result["session_id"]), timeout=1.0
        )


async def test_list_sessions_tool_exposes_status(tmp_path: Path) -> None:
    """The list_sessions MCP tool surfaces the new status/last_active fields."""
    store = server._get_session_store()
    s = store.create("skill-a", "gemma4:12b")
    s.status = "running"
    s.last_active = "2026-04-08T10:00:00+00:00"
    store.save(s)

    result_json = await server.list_sessions()
    result = json.loads(result_json)
    assert len(result) == 1
    assert result[0]["status"] == "running"
    assert result[0]["last_active"] == "2026-04-08T10:00:00+00:00"


# -- Phase 3: inbox notifications + read_inbox tool --


async def test_run_skill_emits_ok_notification_on_success(tmp_path: Path) -> None:
    """A successful turn writes an 'ok' notification with the response summary."""
    skill = _make_skill(tmp_path)
    mock_result = _make_mock_result("agent said hi")

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_build.return_value = mock_agent

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
    """An exception during the turn writes an 'error' notification."""
    skill = _make_skill(tmp_path)

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        mock_build.return_value = mock_agent

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
    """CancelledError writes a 'cancelled' notification before propagating.

    Streaming mode is required to trigger the BaseException path
    (CancelledError raised mid-stream — see existing test for the
    cancelled trailer behavior).
    """
    _override_streaming_config(tmp_path)

    skill = _make_skill(tmp_path)
    cancelled = asyncio.CancelledError()

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = MagicMock()
        mock_agent.run_stream = MagicMock(
            return_value=_make_failing_stream(["partial"], cancelled)
        )
        mock_build.return_value = mock_agent

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
    mock_result = _make_mock_result("agent succeeded")

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_build.return_value = mock_agent

        # Make Inbox.write raise unconditionally
        def boom(*args: object, **kwargs: object) -> None:
            raise OSError("inbox volume full")

        with patch.object(server._get_inbox(), "write", side_effect=boom):
            result = await server._run_skill(skill, "go")

    # The agent's response is preserved despite the inbox failure
    assert "error" not in result
    assert result["response"] == "agent succeeded"
    assert result["status"] == "idle"


async def test_run_skill_background_emits_notification_on_completion(
    tmp_path: Path,
) -> None:
    """A background run still writes a notification when its task finishes."""
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()

    async def blocking_run(*_a: object, **_k: object) -> MagicMock:
        await can_finish.wait()
        return _make_mock_result("background ok")

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=blocking_run)
        mock_build.return_value = mock_agent

        first = await server._run_skill(skill, "go", run_in_background=True)
        session_id = first["session_id"]
        assert first["status"] == "running"

        # Inbox is empty until the background task completes
        inbox = server._get_inbox()
        notifications, _ = inbox.read(since="")
        assert notifications == []

        can_finish.set()
        store = server._get_session_store()
        await asyncio.wait_for(store.drain(session_id), timeout=1.0)

        notifications, _ = inbox.read(since="")
        assert len(notifications) == 1
        assert notifications[0]["status"] == "ok"
        assert notifications[0]["session_id"] == session_id
        assert notifications[0]["summary"] == "background ok"


async def test_read_inbox_tool_returns_records_and_head(tmp_path: Path) -> None:
    """read_inbox MCP tool returns the {notifications, head} JSON shape."""
    inbox = server._get_inbox()
    n = inbox.write(
        session_id="s1",
        skill="skill",
        model="m",
        status="ok",
        summary="hello",
    )

    result_json = await server.read_inbox()
    result = json.loads(result_json)
    assert "notifications" in result
    assert "head" in result
    assert len(result["notifications"]) == 1
    assert result["notifications"][0]["notification_id"] == n.notification_id
    assert result["head"] == n.notification_id


async def test_read_inbox_tool_advances_with_cursor(tmp_path: Path) -> None:
    """Passing the previous head as 'since' returns only newer records."""
    inbox = server._get_inbox()
    first = inbox.write(
        session_id="s1", skill="k", model="m", status="ok", summary="one"
    )

    initial = json.loads(await server.read_inbox(since=""))
    assert initial["head"] == first.notification_id

    # No new records yet — cursor should not advance
    second = json.loads(await server.read_inbox(since=initial["head"]))
    assert second["notifications"] == []
    assert second["head"] == initial["head"]

    # Write more, then drain forward
    second_n = inbox.write(
        session_id="s2", skill="k", model="m", status="error", summary="boom"
    )
    third = json.loads(await server.read_inbox(since=initial["head"]))
    assert len(third["notifications"]) == 1
    assert third["notifications"][0]["notification_id"] == second_n.notification_id
    assert third["head"] == second_n.notification_id


async def test_read_inbox_tool_empty_inbox(tmp_path: Path) -> None:
    """read_inbox on an empty inbox returns empty notifications and empty head."""
    result = json.loads(await server.read_inbox())
    assert result["notifications"] == []
    assert result["head"] == ""


# -- Phase 5: per-session mailbox queue + worker --


async def test_resume_busy_session_enqueues_and_runs_in_order(
    tmp_path: Path,
) -> None:
    """Two background pushes to the same session run FIFO without rejection.

    The second push must NOT return status='busy' (Phase 2 behavior);
    it must enqueue and the worker must process both items in order.
    The session's message history is the ground truth for ordering.
    """
    skill = _make_skill(tmp_path)
    can_finish_first = asyncio.Event()
    call_order: list[str] = []

    async def staged_run(*args: Any, **kwargs: Any) -> MagicMock:
        # First call records "first-start" then waits for the gate.
        # Second call records "second-start" and returns immediately.
        history = kwargs.get("message_history") or []
        label = "first" if not history else "second"
        call_order.append(f"{label}-start")
        if label == "first":
            await can_finish_first.wait()
        return _make_mock_result(f"{label}-done")

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=staged_run)
        mock_build.return_value = mock_agent

        first = await server._run_skill(skill, "go", run_in_background=True)
        sid = first["session_id"]
        # First item should be running (mailbox was empty before push)
        assert first["status"] == "running"

        # Yield so the worker actually picks up the first item and
        # blocks inside the slow agent call before we push the second.
        for _ in range(5):
            await asyncio.sleep(0)
        assert "first-start" in call_order

        # Second push must enqueue, not be rejected
        second = await server._run_skill(
            skill, "again", session_id=sid, run_in_background=True
        )
        assert second["status"] == "queued"
        assert second["session_id"] == sid
        assert second["mailbox_depth"] == 1  # one item waiting behind the inflight

        # Release the first item; both should drain in order
        can_finish_first.set()
        store = server._get_session_store()
        await asyncio.wait_for(store.drain(sid), timeout=1.0)

        # FIFO ordering invariant: the worker processed first then second.
        # (The mock's all_messages() returns a fresh pair on each call so
        # we can't introspect the cumulative history; call_order is the
        # ground truth for ordering.)
        assert call_order == ["first-start", "second-start"]
        session = store.get(sid)
        assert session is not None
        assert session.status == "idle"
        # mailbox is fully drained
        assert store.mailbox_depth(sid) == 0


async def test_foreground_resume_blocks_until_its_turn(tmp_path: Path) -> None:
    """A foreground resume on a busy session waits for its enqueued item."""
    skill = _make_skill(tmp_path)
    can_finish_first = asyncio.Event()

    async def staged_run(*args: Any, **kwargs: Any) -> MagicMock:
        history = kwargs.get("message_history") or []
        if not history:
            await can_finish_first.wait()
            return _make_mock_result("first-done")
        return _make_mock_result("second-done")

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=staged_run)
        mock_build.return_value = mock_agent

        # Background run holds the worker
        first = await server._run_skill(skill, "go", run_in_background=True)
        sid = first["session_id"]

        # Foreground resume must enqueue and wait, not return immediately.
        # We start it as a task so we can verify it's pending.
        fg_task = asyncio.create_task(
            server._run_skill(skill, "again", session_id=sid)
        )
        # Give the event loop a few ticks; the foreground task should
        # still be pending because the first item is blocking the worker.
        for _ in range(5):
            await asyncio.sleep(0)
        assert not fg_task.done()

        # Release the first; foreground should now complete with its result
        can_finish_first.set()
        result = await asyncio.wait_for(fg_task, timeout=1.0)
        assert result["status"] == "idle"
        assert result["response"] == "second-done"


async def test_first_background_run_returns_running_not_queued(
    tmp_path: Path,
) -> None:
    """A push to an idle session reports status='running' (not 'queued')."""
    skill = _make_skill(tmp_path)

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=_make_mock_result())
        mock_build.return_value = mock_agent

        result = await server._run_skill(skill, "go", run_in_background=True)
        assert result["status"] == "running"
        assert result["mailbox_depth"] == 1
        store = server._get_session_store()
        await asyncio.wait_for(store.drain(result["session_id"]), timeout=1.0)


async def test_skill_mismatch_on_resume_returns_error(tmp_path: Path) -> None:
    """Resuming a session with a different skill is rejected."""
    skill_a = _make_skill(tmp_path, name="skill-a")
    skill_b = _make_skill(tmp_path, name="skill-b")

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=_make_mock_result())
        mock_build.return_value = mock_agent

        first = await server._run_skill(skill_a, "go")
        sid = first["session_id"]

        # Now try to resume the same session with a different skill
        second = await server._run_skill(skill_b, "go", session_id=sid)
        assert second["status"] == "skill_mismatch"
        assert "skill-a" in second["error"]
        assert "skill-b" in second["error"]
        # Resume must NOT have run -- only the original turn's messages exist
        store = server._get_session_store()
        session = store.get(sid)
        assert session is not None
        assert len(session.messages) == 2  # only the first turn


async def test_list_sessions_includes_mailbox_depth(tmp_path: Path) -> None:
    """The list_sessions output exposes the per-session mailbox_depth."""
    skill = _make_skill(tmp_path)
    can_finish = asyncio.Event()

    async def blocking_run(*_a: object, **_k: object) -> MagicMock:
        await can_finish.wait()
        return _make_mock_result()

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=blocking_run)
        mock_build.return_value = mock_agent

        first = await server._run_skill(skill, "go", run_in_background=True)
        sid = first["session_id"]
        # Park a second item in the mailbox while the first is in flight
        await server._run_skill(
            skill, "again", session_id=sid, run_in_background=True
        )

        # Yield so the worker has actually picked up the first item;
        # mailbox_depth then reflects only the queued (not in-flight) item.
        for _ in range(5):
            await asyncio.sleep(0)

        result_json = await server.list_sessions()
        listing = json.loads(result_json)
        entry = next(s for s in listing if s["session_id"] == sid)
        assert entry["mailbox_depth"] == 1
        assert entry["status"] == "running"

        # Drain so the fixture teardown is clean
        can_finish.set()
        store = server._get_session_store()
        await asyncio.wait_for(store.drain(sid), timeout=1.0)


async def test_worker_lazy_started_on_first_push(tmp_path: Path) -> None:
    """A worker is created on the first push and survives subsequent pushes."""
    skill = _make_skill(tmp_path)

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=_make_mock_result())
        mock_build.return_value = mock_agent

        store = server._get_session_store()
        first = await server._run_skill(skill, "one", run_in_background=True)
        sid = first["session_id"]
        # Worker exists immediately after the first push
        assert store.has_worker(sid)
        await asyncio.wait_for(store.drain(sid), timeout=1.0)
        # Worker is still alive between turns -- it blocks on mailbox.get()
        assert store.has_worker(sid)


async def test_shutdown_drains_workers_no_pending_warnings(
    tmp_path: Path,
) -> None:
    """SessionStore.shutdown() must terminate workers cleanly.

    The fixture teardown calls shutdown() automatically, but exercising
    it explicitly verifies no 'Task was destroyed but it is pending'
    warnings appear when the loop tears down.
    """
    skill = _make_skill(tmp_path)

    with patch.object(server, "_build_agent") as mock_build:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=_make_mock_result())
        mock_build.return_value = mock_agent

        first = await server._run_skill(skill, "go", run_in_background=True)
        sid = first["session_id"]
        store = server._get_session_store()
        await asyncio.wait_for(store.drain(sid), timeout=1.0)
        assert store.has_worker(sid) is True

        await store.shutdown()
        assert store.has_worker(sid) is False
