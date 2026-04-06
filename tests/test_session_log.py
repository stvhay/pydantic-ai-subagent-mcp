"""Tests for SessionStore.log_path() and tail()."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai_subagent_mcp.session import SessionStore


def test_log_path_uses_session_id(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("test-skill", "gemma4:12b")
    log = store.log_path(session.session_id)
    assert log == tmp_path / "sessions" / f"{session.session_id}.log"


def test_tail_nonexistent_session_returns_empty(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    text, offset = store.tail("nonexistent-uuid")
    assert text == ""
    assert offset == 0


def test_tail_reads_full_log_from_offset_zero(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("test-skill", "gemma4:12b")
    log = store.log_path(session.session_id)
    log.write_text("hello world")

    text, offset = store.tail(session.session_id, offset=0)
    assert text == "hello world"
    assert offset == len("hello world")


def test_tail_incremental_reads(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("test-skill", "gemma4:12b")
    log = store.log_path(session.session_id)
    log.write_text("first")

    text1, offset1 = store.tail(session.session_id, offset=0)
    assert text1 == "first"
    assert offset1 == 5

    # Append more content, then tail from the previous offset
    with log.open("a") as f:
        f.write(" second")

    text2, offset2 = store.tail(session.session_id, offset=offset1)
    assert text2 == " second"
    assert offset2 == 5 + 7  # "first" + " second"


def test_tail_respects_max_bytes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("test-skill", "gemma4:12b")
    log = store.log_path(session.session_id)
    log.write_text("abcdefghij")  # 10 bytes

    text, offset = store.tail(session.session_id, offset=0, max_bytes=4)
    assert text == "abcd"
    assert offset == 4


def test_tail_handles_split_utf8_at_boundary(tmp_path: Path) -> None:
    """A multi-byte UTF-8 sequence split across a read boundary must not crash."""
    store = SessionStore(tmp_path / "sessions")
    session = store.create("test-skill", "gemma4:12b")
    log = store.log_path(session.session_id)
    # "é" is 0xc3 0xa9 in UTF-8 — write both bytes
    log.write_bytes(b"ab\xc3\xa9cd")  # "abécd" = 6 bytes total

    # Read only the first 3 bytes: "ab" + first byte of "é" (0xc3)
    text, offset = store.tail(session.session_id, offset=0, max_bytes=3)
    assert offset == 3
    # Must not raise. With errors="replace", the lone 0xc3 becomes "\ufffd" (replacement char).
    assert text.startswith("ab")
    assert len(text) == 3  # "ab" + replacement char


def test_tail_offset_beyond_file_size(tmp_path: Path) -> None:
    """Offset past end of file returns empty text and preserves the offset."""
    store = SessionStore(tmp_path / "sessions")
    session = store.create("test-skill", "gemma4:12b")
    log = store.log_path(session.session_id)
    log.write_text("abc")  # 3 bytes

    text, offset = store.tail(session.session_id, offset=100)
    assert text == ""
    assert offset == 100  # preserved, not reset


def test_tail_empty_log_file_exists(tmp_path: Path) -> None:
    """Empty log file (exists but zero bytes) returns empty text at offset 0."""
    store = SessionStore(tmp_path / "sessions")
    session = store.create("test-skill", "gemma4:12b")
    log = store.log_path(session.session_id)
    log.write_bytes(b"")  # empty file that exists

    text, offset = store.tail(session.session_id, offset=0)
    assert text == ""
    assert offset == 0
