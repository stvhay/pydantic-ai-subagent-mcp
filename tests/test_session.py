"""Tests for session management with native pydantic-ai messages."""

import json
from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from pydantic_ai_subagent_mcp.session import SessionStore


def _make_messages() -> list[ModelMessage]:
    """Create a simple request/response pair for testing."""
    return [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi there")]),
    ]


def test_create_and_retrieve_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("test-skill", "gemma4:12b")
    assert session.session_id
    assert session.skill_name == "test-skill"
    assert session.messages == []

    retrieved = store.get(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id


def test_session_persistence_with_native_messages(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    store = SessionStore(session_dir)
    session = store.create("test-skill", "gemma4:12b")
    session.messages = _make_messages()
    store.save(session)

    # New store instance should load from disk
    store2 = SessionStore(session_dir)
    loaded = store2.get(session.session_id)
    assert loaded is not None
    assert len(loaded.messages) == 2

    # Verify types are preserved
    req = loaded.messages[0]
    resp = loaded.messages[1]
    assert isinstance(req, ModelRequest)
    assert isinstance(resp, ModelResponse)
    assert req.parts[0].content == "hello"  # type: ignore[union-attr]
    assert resp.parts[0].content == "hi there"  # type: ignore[union-attr]


def test_session_to_dict(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("test-skill", "gemma4:12b")
    session.messages = _make_messages()

    d = session.to_dict()
    assert d["session_id"] == session.session_id
    assert d["skill_name"] == "test-skill"
    assert d["model"] == "gemma4:12b"
    assert isinstance(d["messages"], list)
    assert len(d["messages"]) == 2
    # Messages should be JSON-serializable dicts
    assert d["messages"][0]["kind"] == "request"
    assert d["messages"][1]["kind"] == "response"


def test_list_sessions(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.create("skill-a", "gemma4:12b")
    store.create("skill-b", "gemma4:27b")

    listing = store.list_sessions()
    assert len(listing) == 2
    names = {s["skill_name"] for s in listing}
    assert names == {"skill-a", "skill-b"}


def test_session_disk_format_uses_type_adapter(tmp_path: Path) -> None:
    """Verify on-disk format uses ModelMessagesTypeAdapter for messages."""
    session_dir = tmp_path / "sessions"
    store = SessionStore(session_dir)
    session = store.create("test-skill", "gemma4:12b")
    session.messages = _make_messages()
    store.save(session)

    # Read raw JSON from disk
    raw = json.loads((session_dir / f"{session.session_id}.json").read_text())
    assert "messages" in raw
    assert raw["messages"][0]["kind"] == "request"
    assert raw["messages"][1]["kind"] == "response"

    # Verify messages can be deserialized by the type adapter directly
    restored = ModelMessagesTypeAdapter.validate_python(raw["messages"])
    assert len(restored) == 2
    assert isinstance(restored[0], ModelRequest)
