"""Tests for the durable on-disk completion-notification inbox."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pydantic_ai_subagent_mcp.inbox import Inbox


def _write(inbox: Inbox, summary: str = "done", status: str = "ok") -> str:
    """Write a notification with default fields and return its id."""
    n = inbox.write(
        session_id="sess",
        skill="skill",
        model="gemma4:12b",
        status=status,  # type: ignore[arg-type]
        summary=summary,
    )
    return n.notification_id


def test_write_creates_atomic_file(tmp_path: Path) -> None:
    """A write produces a single .json file named after its uuid7."""
    inbox = Inbox(tmp_path / "inbox")
    n = inbox.write(
        session_id="s1",
        skill="skill-a",
        model="gemma4:12b",
        status="ok",
        summary="hello",
    )
    files = sorted(p.name for p in (tmp_path / "inbox").iterdir())
    assert files == [f"{n.notification_id}.json"]
    record = json.loads((tmp_path / "inbox" / f"{n.notification_id}.json").read_text())
    assert record["session_id"] == "s1"
    assert record["skill"] == "skill-a"
    assert record["status"] == "ok"
    assert record["summary"] == "hello"
    assert record["notification_id"] == n.notification_id
    assert "timestamp" in record


def test_write_truncates_long_summary(tmp_path: Path) -> None:
    """Summaries beyond SUMMARY_MAX_CHARS are truncated with an ellipsis."""
    inbox = Inbox(tmp_path / "inbox")
    long = "x" * (Inbox.SUMMARY_MAX_CHARS + 50)
    n = inbox.write(
        session_id="s",
        skill="k",
        model="m",
        status="ok",
        summary=long,
    )
    assert len(n.summary) == Inbox.SUMMARY_MAX_CHARS
    assert n.summary.endswith("...")


def test_uuid7_ids_are_monotonically_ordered(tmp_path: Path) -> None:
    """Sequential writes yield ids that sort lexicographically by time.

    This is the core invariant the inbox relies on to use filename
    sorting as time ordering.
    """
    inbox = Inbox(tmp_path / "inbox")
    ids = [_write(inbox) for _ in range(10)]
    assert ids == sorted(ids)


def test_read_empty_inbox_returns_empty_with_empty_head(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path / "inbox")
    records, head = inbox.read()
    assert records == []
    assert head == ""


def test_read_with_empty_since_returns_recent_tail(tmp_path: Path) -> None:
    """Empty cursor returns the most recent ``limit`` records."""
    inbox = Inbox(tmp_path / "inbox")
    ids = [_write(inbox, summary=f"r{i}") for i in range(5)]
    records, head = inbox.read(since="", limit=3)
    # Most recent 3 (oldest first within the slice)
    assert [r["notification_id"] for r in records] == ids[-3:]
    assert head == ids[-1]


def test_read_with_since_returns_strictly_greater(tmp_path: Path) -> None:
    """Cursor returns only records with id > since (no duplicates)."""
    inbox = Inbox(tmp_path / "inbox")
    first = _write(inbox, summary="one")
    second = _write(inbox, summary="two")
    third = _write(inbox, summary="three")

    records, head = inbox.read(since=first)
    ids = [r["notification_id"] for r in records]
    assert first not in ids  # strictly greater
    assert ids == [second, third]
    assert head == third


def test_read_returns_since_as_head_when_no_new_records(tmp_path: Path) -> None:
    """Cursor unchanged when there are no new records to advance past."""
    inbox = Inbox(tmp_path / "inbox")
    only = _write(inbox)
    records, head = inbox.read(since=only)
    assert records == []
    assert head == only  # cursor doesn't drift


def test_read_limit_caps_results(tmp_path: Path) -> None:
    """``limit`` bounds the returned batch size when draining a backlog."""
    inbox = Inbox(tmp_path / "inbox")
    ids = [_write(inbox) for _ in range(10)]
    # Drain from the start, two at a time
    records, head = inbox.read(since="", limit=2)
    assert len(records) == 2
    # With since="", we get the LAST 2 (tail view)
    assert [r["notification_id"] for r in records] == ids[-2:]

    # With a cursor, we get the FIRST 2 newer (forward drain)
    records2, head2 = inbox.read(since=ids[0], limit=2)
    assert [r["notification_id"] for r in records2] == ids[1:3]
    assert head2 == ids[2]


def test_read_skips_corrupt_files(tmp_path: Path) -> None:
    """A malformed JSON file does not block draining the rest of the inbox."""
    inbox = Inbox(tmp_path / "inbox")
    good1 = _write(inbox, summary="good1")
    # Inject a corrupt file with a name that would sort between two valid ids
    corrupt_path = tmp_path / "inbox" / "01900000-0000-7000-8000-000000000000.json"
    corrupt_path.write_text("{not valid json")
    good2 = _write(inbox, summary="good2")

    records, _ = inbox.read(since="")
    ids = [r["notification_id"] for r in records]
    assert good1 in ids
    assert good2 in ids
    assert "01900000-0000-7000-8000-000000000000" not in ids


def test_head_returns_latest_id(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path / "inbox")
    assert inbox.head() == ""
    a = _write(inbox)
    assert inbox.head() == a
    b = _write(inbox)
    assert inbox.head() == b
    assert b > a


def test_atomic_write_leaves_no_tempfiles_on_success(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path / "inbox")
    _write(inbox)
    _write(inbox)
    files = sorted(p.name for p in (tmp_path / "inbox").iterdir())
    # Only .json records, no .tmp leftovers
    assert all(name.endswith(".json") for name in files)
    assert all(not name.startswith(".") for name in files)


def test_atomic_write_cleans_tempfile_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If os.replace raises mid-write, the tempfile is cleaned up."""
    inbox = Inbox(tmp_path / "inbox")

    def boom(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("pydantic_ai_subagent_mcp.inbox.os.replace", boom)

    with pytest.raises(OSError, match="simulated rename failure"):
        _write(inbox, summary="never lands")

    leftover = [p.name for p in (tmp_path / "inbox").iterdir()]
    assert leftover == []


def test_dotfiles_are_ignored_by_read(tmp_path: Path) -> None:
    """Hidden files (e.g. tempfiles named .X.tmp.json) are not returned."""
    inbox = Inbox(tmp_path / "inbox")
    _write(inbox)
    # Create a hidden file that would otherwise match *.json
    (tmp_path / "inbox" / ".hidden.json").write_text(
        json.dumps({"notification_id": "z", "session_id": "s"})
    )
    records, _ = inbox.read()
    assert all(not r["notification_id"].startswith(".") for r in records)
    assert len(records) == 1


def test_inbox_dir_is_created_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "inbox"
    assert not target.exists()
    Inbox(target)
    assert target.exists() and target.is_dir()
