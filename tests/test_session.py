"""Tests for session management with native pydantic-ai messages."""

import asyncio
import json
import os
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from pydantic_ai_subagent_mcp.session import Session, SessionStore


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


# -- Phase 1: per-session locks + atomic persistence --


async def test_per_session_lock_serializes_concurrent_holders(
    tmp_path: Path,
) -> None:
    """Two coroutines holding the same session lock must run sequentially.

    Verifies the linearizability invariant required by Phase 1: when two
    callers race on the same session_id, one must fully complete its
    critical section before the other starts.
    """
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")

    order: list[str] = []

    async def critical_section(label: str) -> None:
        async with store.lock(session.session_id):
            order.append(f"start-{label}")
            # Force a context switch while holding the lock; if the lock
            # is broken, the other coroutine will interleave here.
            await asyncio.sleep(0.01)
            order.append(f"end-{label}")

    await asyncio.gather(critical_section("a"), critical_section("b"))

    # Whichever ran first must have fully completed before the other started.
    assert order in (
        ["start-a", "end-a", "start-b", "end-b"],
        ["start-b", "end-b", "start-a", "end-a"],
    )


async def test_per_session_locks_are_independent(tmp_path: Path) -> None:
    """Locks for distinct session_ids must not block each other."""
    store = SessionStore(tmp_path / "sessions")
    s1 = store.create("skill", "gemma4:12b")
    s2 = store.create("skill", "gemma4:12b")

    s1_holding = asyncio.Event()
    s1_can_release = asyncio.Event()
    s2_acquired = asyncio.Event()

    async def hold_s1() -> None:
        async with store.lock(s1.session_id):
            s1_holding.set()
            await s1_can_release.wait()

    async def acquire_s2() -> None:
        await s1_holding.wait()
        async with store.lock(s2.session_id):
            s2_acquired.set()

    t1 = asyncio.create_task(hold_s1())
    t2 = asyncio.create_task(acquire_s2())

    # s2 must be acquirable while s1 is held; if locks were global this
    # would deadlock and time out.
    await asyncio.wait_for(s2_acquired.wait(), timeout=1.0)

    s1_can_release.set()
    await asyncio.gather(t1, t2)


async def test_lock_returns_same_instance_per_session(tmp_path: Path) -> None:
    """The lock for a given session_id must be the same Lock object on
    repeated access — otherwise concurrent callers would each get their
    own lock and the serialization guarantee would be lost."""
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")

    lock_a = store._get_lock(session.session_id)
    lock_b = store._get_lock(session.session_id)
    assert lock_a is lock_b


def test_atomic_persist_leaves_no_tempfiles_on_success(tmp_path: Path) -> None:
    """A successful save must leave only the final {session_id}.json file
    in the session directory — no leaked tempfiles."""
    session_dir = tmp_path / "sessions"
    store = SessionStore(session_dir)
    session = store.create("skill", "gemma4:12b")
    session.messages = _make_messages()
    store.save(session)

    files = sorted(p.name for p in session_dir.iterdir())
    assert files == [f"{session.session_id}.json"]


def test_atomic_persist_cleans_tempfile_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the atomic rename fails mid-write, the tempfile must be cleaned
    up and the original file (if any) must remain intact and parseable."""
    session_dir = tmp_path / "sessions"
    store = SessionStore(session_dir)
    session = store.create("skill", "gemma4:12b")
    # Establish a known-good baseline on disk before the failing save
    session.messages = _make_messages()
    store.save(session)
    baseline = (session_dir / f"{session.session_id}.json").read_text()

    # Now make os.replace raise on the next call
    real_replace = os.replace

    def boom(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("pydantic_ai_subagent_mcp.session.os.replace", boom)

    # Mutate and try to save — must raise
    session.messages = [
        ModelRequest(parts=[UserPromptPart(content="updated")]),
        ModelResponse(parts=[TextPart(content="updated")]),
    ]
    with pytest.raises(OSError, match="simulated rename failure"):
        store.save(session)

    # Restore the real os.replace so subsequent assertions work
    monkeypatch.setattr("pydantic_ai_subagent_mcp.session.os.replace", real_replace)

    # The original file is intact and parseable
    current = (session_dir / f"{session.session_id}.json").read_text()
    assert current == baseline
    json.loads(current)  # parses

    # No tempfile leaked
    leftover = [p.name for p in session_dir.iterdir() if p.name.endswith(".tmp")]
    assert leftover == []


# -- Phase 2: status enum, last_active, task registry --


def test_new_session_starts_idle_with_last_active(tmp_path: Path) -> None:
    """A freshly created session has status=idle and last_active=created_at."""
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")
    assert session.status == "idle"
    assert session.last_active == session.created_at


def test_session_status_round_trips_through_disk(tmp_path: Path) -> None:
    """status and last_active survive a save/reload cycle."""
    session_dir = tmp_path / "sessions"
    store = SessionStore(session_dir)
    session = store.create("skill", "gemma4:12b")
    session.status = "running"
    session.last_active = "2026-04-08T12:34:56+00:00"
    store.save(session)

    store2 = SessionStore(session_dir)
    loaded = store2.get(session.session_id)
    assert loaded is not None
    assert loaded.status == "running"
    assert loaded.last_active == "2026-04-08T12:34:56+00:00"


def test_session_to_dict_includes_status_and_last_active(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")
    session.status = "failed"
    session.last_active = "2026-04-08T01:02:03+00:00"

    d = session.to_dict()
    assert d["status"] == "failed"
    assert d["last_active"] == "2026-04-08T01:02:03+00:00"


def test_legacy_session_files_default_to_idle(tmp_path: Path) -> None:
    """Pre-Phase-2 session files (no status/last_active fields) load as idle."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    sid = "legacy-uuid"
    legacy = {
        "session_id": sid,
        "skill_name": "skill",
        "model": "gemma4:12b",
        "created_at": "2026-01-01T00:00:00+00:00",
        "messages": [],
    }
    (session_dir / f"{sid}.json").write_text(json.dumps(legacy))

    store = SessionStore(session_dir)
    loaded = store.get(sid)
    assert loaded is not None
    assert loaded.status == "idle"
    assert loaded.last_active == "2026-01-01T00:00:00+00:00"


def test_list_sessions_exposes_status_and_last_active(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    a = store.create("skill-a", "gemma4:12b")
    a.status = "running"
    a.last_active = "2026-04-08T10:00:00+00:00"
    store.save(a)

    listing = store.list_sessions()
    assert len(listing) == 1
    entry = listing[0]
    assert entry["status"] == "running"
    assert entry["last_active"] == "2026-04-08T10:00:00+00:00"


# -- Phase 5: per-session mailbox + worker primitives --


async def test_get_or_create_mailbox_returns_same_queue(tmp_path: Path) -> None:
    """Repeated calls for the same session_id return the same Queue object."""
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")
    mb1 = store.get_or_create_mailbox(session.session_id)
    mb2 = store.get_or_create_mailbox(session.session_id)
    assert mb1 is mb2


async def test_mailbox_depth_reflects_queue_size(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")
    assert store.mailbox_depth(session.session_id) == 0
    mb = store.get_or_create_mailbox(session.session_id)
    await mb.put("a")
    await mb.put("b")
    assert store.mailbox_depth(session.session_id) == 2
    await mb.get()
    assert store.mailbox_depth(session.session_id) == 1


async def test_mailbox_depth_unknown_session_is_zero(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    assert store.mailbox_depth("never-touched") == 0


async def test_has_worker_tracks_registration_and_completion(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")

    assert store.has_worker(session.session_id) is False

    finished = asyncio.Event()

    async def worker() -> None:
        await finished.wait()

    task = asyncio.create_task(worker())
    store.register_worker(session.session_id, task)
    assert store.has_worker(session.session_id) is True

    finished.set()
    await task
    # has_worker reads task.done() so the next check reflects completion
    assert store.has_worker(session.session_id) is False


async def test_is_busy_true_when_mailbox_has_items(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")
    mb = store.get_or_create_mailbox(session.session_id)
    assert store.is_busy(session.session_id) is False
    await mb.put("work")
    assert store.is_busy(session.session_id) is True


async def test_is_busy_true_when_session_status_running(tmp_path: Path) -> None:
    """Even with an empty mailbox, status='running' counts as busy."""
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")
    session.status = "running"
    assert store.is_busy(session.session_id) is True


async def test_drain_waits_for_mailbox_to_empty(tmp_path: Path) -> None:
    """drain() returns once every put has been matched by task_done."""
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")
    mb = store.get_or_create_mailbox(session.session_id)

    await mb.put("a")
    await mb.put("b")

    async def consumer() -> None:
        for _ in range(2):
            await mb.get()
            mb.task_done()

    consumer_task = asyncio.create_task(consumer())
    await asyncio.wait_for(store.drain(session.session_id), timeout=1.0)
    await consumer_task


async def test_drain_unknown_session_is_noop(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    # Must not raise even though there's no mailbox for this id
    await store.drain("never-touched")


async def test_shutdown_cancels_workers_and_clears_registry(
    tmp_path: Path,
) -> None:
    """shutdown() cancels every registered worker and awaits termination."""
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")
    mb = store.get_or_create_mailbox(session.session_id)

    # Worker blocks forever on an empty mailbox
    async def worker() -> None:
        await mb.get()

    task = asyncio.create_task(worker())
    store.register_worker(session.session_id, task)
    # Yield once so the worker actually starts and reaches the await
    await asyncio.sleep(0)
    assert store.has_worker(session.session_id) is True

    await store.shutdown()
    assert task.cancelled()
    assert store.has_worker(session.session_id) is False
    # Mailboxes are cleared too so a fresh test/run starts clean
    assert store.mailbox_depth(session.session_id) == 0


def test_touch_updates_last_active_in_cache(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("skill", "gemma4:12b")
    original = session.last_active
    # Touch should set last_active to a timestamp >= the original
    store.touch(session.session_id)
    assert session.last_active >= original


def test_touch_unknown_session_is_noop(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.touch("never-existed")  # must not raise


def test_session_dataclass_defaults() -> None:
    """Session can still be constructed with the minimum required fields."""
    s = Session(
        session_id="abc",
        skill_name="skill",
        model="gemma4:12b",
        created_at="2026-04-08T00:00:00+00:00",
    )
    assert s.status == "idle"
    assert s.last_active == ""
    assert s.messages == []
