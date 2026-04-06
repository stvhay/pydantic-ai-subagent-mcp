"""Tests for the MCP server module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from pydantic_ai_subagent_mcp import server
from pydantic_ai_subagent_mcp.config import ServerConfig
from pydantic_ai_subagent_mcp.skills import Skill


@pytest.fixture(autouse=True)
def _reset_server_globals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset server module globals between tests."""
    monkeypatch.delenv("SUBAGENT_MCP_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("SUBAGENT_MCP_STREAMING", raising=False)
    server._config = ServerConfig(
        session_dir=str(tmp_path / "sessions"),
        streaming=False,
    )
    server._session_store = None
    server._skills = []


def _make_skill(tmp_path: Path, name: str = "test-skill") -> Skill:
    """Create a test skill with a temp markdown file."""
    md = tmp_path / f"{name}.md"
    md.write_text(f"# {name}\n\nA test skill.\n")
    return Skill(name=name, description="A test skill", source_path=md)


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

    def __init__(self, chunks: list[str], messages: list) -> None:
        self._chunks = chunks
        self._messages = messages
        self._output = "".join(chunks)

    async def stream_text(self, *, delta: bool = False):  # noqa: ARG002
        for chunk in self._chunks:
            yield chunk

    async def get_output(self) -> str:
        return self._output

    def all_messages(self):
        return self._messages


class _FakeStreamCtx:
    """Async context manager wrapper for _FakeStreamResult."""

    def __init__(self, result: _FakeStreamResult) -> None:
        self._result = result

    async def __aenter__(self) -> _FakeStreamResult:
        return self._result

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _make_fake_stream(chunks: list[str], output: str) -> _FakeStreamCtx:
    messages = [
        ModelRequest(parts=[UserPromptPart(content="test prompt")]),
        ModelResponse(parts=[TextPart(content=output)]),
    ]
    return _FakeStreamCtx(_FakeStreamResult(chunks, messages))


async def test_run_skill_streaming_writes_log(tmp_path: Path) -> None:
    """With streaming=True, _run_skill writes deltas to the session log file."""
    server._config = ServerConfig(
        session_dir=str(tmp_path / "sessions"),
        streaming=True,
    )
    server._session_store = None  # force rebuild with new config

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


async def test_run_skill_streaming_multi_turn_appends(tmp_path: Path) -> None:
    """Each turn appends a new prompt/response block to the same log."""
    server._config = ServerConfig(
        session_dir=str(tmp_path / "sessions"),
        streaming=True,
    )
    server._session_store = None

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

    store = server._get_session_store()
    log_path = store.log_path(result["session_id"])
    assert not log_path.exists()


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
