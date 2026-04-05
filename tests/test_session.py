"""Tests for session management."""

from pathlib import Path

from pydantic_ai_subagent_mcp.session import SessionStore


def test_create_and_retrieve_session(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions")
    session = store.create("test-skill", "gemma4:12b")
    assert session.session_id
    assert session.skill_name == "test-skill"

    retrieved = store.get(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id


def test_session_persistence(tmp_path: Path):
    session_dir = tmp_path / "sessions"
    store = SessionStore(session_dir)
    session = store.create("test-skill", "gemma4:12b")
    session.add_message("user", "hello")
    session.add_message("assistant", "hi there")
    store.save(session)

    # New store instance should load from disk
    store2 = SessionStore(session_dir)
    loaded = store2.get(session.session_id)
    assert loaded is not None
    assert len(loaded.messages) == 2
    assert loaded.messages[0].content == "hello"
    assert loaded.messages[1].content == "hi there"


def test_list_sessions(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions")
    store.create("skill-a", "gemma4:12b")
    store.create("skill-b", "gemma4:27b")

    listing = store.list_sessions()
    assert len(listing) == 2
    names = {s["skill_name"] for s in listing}
    assert names == {"skill-a", "skill-b"}
