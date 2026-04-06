"""Integration tests that run the streaming path against a real Ollama endpoint.

These tests are skipped automatically when Ollama is not reachable or the
chosen test model is not installed, so they are safe to run in CI. Operators
can point them at a live endpoint by setting:

  OLLAMA_BASE_URL            — Ollama base URL (default: http://localhost:11434)
  SUBAGENT_MCP_TEST_MODEL    — model to use (default: llama3.2:1b)

The goal is to exercise the full ``agent.run_stream()`` → log-file side channel
wiring end-to-end, which the unit tests cover only with a fake streamer.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from pydantic_ai_subagent_mcp.server import _run_skill_streaming
from pydantic_ai_subagent_mcp.session import SessionStore

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
TEST_MODEL = os.environ.get("SUBAGENT_MCP_TEST_MODEL", "llama3.2:1b")


def _ollama_skip_reason() -> str | None:
    """Return a skip reason string, or None if Ollama + model are available."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001 — any failure means skip
        return f"Ollama not reachable at {OLLAMA_BASE_URL}: {e}"
    installed = [m.get("name", "") for m in resp.json().get("models", [])]
    if not any(name == TEST_MODEL or name.startswith(f"{TEST_MODEL}:") for name in installed):
        return (
            f"Test model {TEST_MODEL!r} not installed on Ollama "
            f"(installed: {installed or '[]'}). "
            f"Run `ollama pull {TEST_MODEL}` or set SUBAGENT_MCP_TEST_MODEL."
        )
    return None


pytestmark = pytest.mark.skipif(
    _ollama_skip_reason() is not None,
    reason=_ollama_skip_reason() or "",
)


def _build_test_agent() -> Agent[None, str]:
    """Build a minimal agent pointed at the live Ollama endpoint."""
    model = OpenAIChatModel(
        model_name=TEST_MODEL,
        provider=OllamaProvider(base_url=f"{OLLAMA_BASE_URL}/v1"),
    )
    return Agent(
        model,
        system_prompt=(
            "You are a terse test assistant. Answer in one short sentence."
        ),
    )


async def test_streaming_against_real_ollama_writes_log(tmp_path: Path) -> None:
    """End-to-end: real agent streams deltas through the log side-channel.

    Verifies that _run_skill_streaming:
      1. returns a non-empty final output and a populated message history
      2. writes the --- prompt --- / --- response --- header block
      3. flushes streamed deltas into the .log file on disk
    """
    store = SessionStore(tmp_path / "sessions")
    session = store.create("integration-test", TEST_MODEL)
    agent = _build_test_agent()

    output, messages = await _run_skill_streaming(
        agent,
        "Say hello.",
        session,
        store,
    )

    assert isinstance(output, str)
    assert output.strip(), "model should return non-empty output"
    assert len(messages) >= 2, "expected at least a request + response pair"

    log_path = store.log_path(session.session_id)
    assert log_path.exists(), "streaming should create the log file"

    content = log_path.read_text(encoding="utf-8")
    # Header is "--- {iso_ts} prompt ---\n"; response marker has no timestamp.
    assert "prompt ---\n" in content
    assert "--- response ---\n" in content
    assert "Say hello." in content

    # Everything after the response marker should be non-empty — it's the
    # concatenation of streamed deltas.
    response_text = content.split("--- response ---\n", 1)[1]
    assert response_text.strip(), "streamed deltas should be flushed to the log"


async def test_streaming_against_real_ollama_multi_turn_appends(
    tmp_path: Path,
) -> None:
    """A second turn on the same session appends a new block to the log."""
    store = SessionStore(tmp_path / "sessions")
    session = store.create("integration-test", TEST_MODEL)
    agent = _build_test_agent()

    _, messages1 = await _run_skill_streaming(
        agent, "Say 'one'.", session, store
    )
    session.messages = messages1

    _, messages2 = await _run_skill_streaming(
        agent, "Say 'two'.", session, store
    )

    assert len(messages2) > len(messages1), "history should grow across turns"

    content = store.log_path(session.session_id).read_text(encoding="utf-8")
    # Header is "--- {iso_ts} prompt ---\n"; response marker has no timestamp.
    assert content.count("prompt ---\n") == 2
    assert content.count("--- response ---\n") == 2
    assert "Say 'one'." in content
    assert "Say 'two'." in content
