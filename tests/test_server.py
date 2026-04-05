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
    server._config = ServerConfig(session_dir=str(tmp_path / "sessions"))
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
