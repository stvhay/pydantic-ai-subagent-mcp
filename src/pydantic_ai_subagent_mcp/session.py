"""Session management with UUID-keyed transcripts using native pydantic-ai messages."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

# Lifecycle states for an individual session.
#
# - ``idle``: no in-flight turn; safe to resume.
# - ``running``: a turn is currently executing for this session.
# - ``failed``: the most recent turn raised; the session is still resumable
#   (the failure is informational, not a terminal state).
SessionStatus = Literal["idle", "running", "failed"]


@dataclass
class Session:
    """A session with UUID key and native pydantic-ai message history."""

    session_id: str
    skill_name: str
    model: str
    created_at: str
    messages: list[ModelMessage] = field(default_factory=list)
    status: SessionStatus = "idle"
    last_active: str = ""

    def to_dict(self) -> dict[str, Any]:
        messages_json = json.loads(
            ModelMessagesTypeAdapter.dump_json(self.messages)
        )
        return {
            "session_id": self.session_id,
            "skill_name": self.skill_name,
            "model": self.model,
            "created_at": self.created_at,
            "status": self.status,
            "last_active": self.last_active or self.created_at,
            "messages": messages_json,
        }


class SessionStore:
    """Persists sessions to disk as JSON files keyed by UUID.

    Concurrency model: each ``session_id`` has its own ``asyncio.Lock`` that
    callers must hold via :meth:`lock` while reading-modifying-writing the
    session's ``messages``. Per-session locks guarantee linearizable history:
    at most one in-flight turn per session, while distinct sessions remain
    independent. Persistence is atomic (tempfile + ``os.replace``) so a crash
    mid-write leaves either the prior or the next coherent state on disk,
    never a half-written file.
    """

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Per-session mailbox queues. Each session has at most one
        # ``asyncio.Queue`` of pending work items, drained FIFO by a
        # dedicated worker task. The actor model: a session is an
        # addressable mailbox; a resume is a "tell" to that mailbox.
        self._mailboxes: dict[str, asyncio.Queue[Any]] = {}
        # Per-session worker tasks. The worker is lazily started on the
        # first push to the mailbox and lives until ``shutdown()`` is
        # called (or its loop tears down). One worker per session
        # guarantees serial execution of that session's turns without
        # the per-session lock having to do contended work.
        self._workers: dict[str, asyncio.Task[None]] = {}

    def _session_path(self, session_id: str) -> Path:
        return self.session_dir / f"{session_id}.json"

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Lazily create and return the per-session asyncio Lock.

        Safe under cooperative concurrency because dict membership check
        and assignment in pure Python contain no ``await`` points, so no
        other coroutine can interleave between the check and the create.
        """
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    @asynccontextmanager
    async def lock(self, session_id: str) -> AsyncIterator[None]:
        """Acquire the per-session lock for the duration of the context.

        Callers must hold this lock around any read-modify-write of a
        session's ``messages`` to prevent concurrent turns from clobbering
        each other's history. Distinct sessions are not blocked.
        """
        async with self._get_lock(session_id):
            yield

    def create(self, skill_name: str, model: str) -> Session:
        """Create a new session with a fresh UUID."""
        session_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        session = Session(
            session_id=session_id,
            skill_name=skill_name,
            model=model,
            created_at=now,
            last_active=now,
        )
        self._sessions[session_id] = session
        self._persist(session)
        return session

    def get(self, session_id: str) -> Session | None:
        """Load a session by ID, from cache or disk."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        path = self._session_path(session_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        messages = ModelMessagesTypeAdapter.validate_python(
            data.get("messages", [])
        )
        # status/last_active may be absent in pre-Phase-2 session files;
        # default to idle and the creation timestamp so old sessions load.
        status: SessionStatus = data.get("status", "idle")
        session = Session(
            session_id=data["session_id"],
            skill_name=data["skill_name"],
            model=data["model"],
            created_at=data["created_at"],
            messages=messages,
            status=status,
            last_active=data.get("last_active", data["created_at"]),
        )
        self._sessions[session_id] = session
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all session IDs and metadata.

        ``mailbox_depth`` reflects in-memory queued work for the session
        (server-local; not persisted on disk). It is 0 for any session
        whose mailbox has never been touched in the current process,
        even if the on-disk record exists.
        """
        sessions: list[dict[str, Any]] = []
        for path in self.session_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                sid = data["session_id"]
                sessions.append({
                    "session_id": sid,
                    "skill_name": data["skill_name"],
                    "model": data["model"],
                    "created_at": data["created_at"],
                    "status": data.get("status", "idle"),
                    "last_active": data.get("last_active", data["created_at"]),
                    "message_count": len(data.get("messages", [])),
                    "mailbox_depth": self.mailbox_depth(sid),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    # -- mailbox + worker registry (for the actor model) --

    def get_or_create_mailbox(self, session_id: str) -> asyncio.Queue[Any]:
        """Return the per-session mailbox, creating it lazily if absent.

        The contained item type is intentionally ``Any`` here so the
        store does not have to import the server-side ``_WorkItem``
        dataclass; the worker that drains the queue knows what to expect.
        """
        mb = self._mailboxes.get(session_id)
        if mb is None:
            mb = asyncio.Queue()
            self._mailboxes[session_id] = mb
        return mb

    def mailbox_depth(self, session_id: str) -> int:
        """Return the number of pending work items in the session's mailbox.

        This counts only *queued* items, not the in-flight one. A session
        with one item being processed and nothing else queued has
        ``mailbox_depth == 0`` and ``status == "running"``.
        """
        mb = self._mailboxes.get(session_id)
        return mb.qsize() if mb is not None else 0

    def has_worker(self, session_id: str) -> bool:
        """Return True if a live worker task exists for this session."""
        worker = self._workers.get(session_id)
        return worker is not None and not worker.done()

    def register_worker(
        self, session_id: str, worker: asyncio.Task[None]
    ) -> None:
        """Track a session worker task. Replaces any prior dead worker."""
        self._workers[session_id] = worker

    def is_busy(self, session_id: str) -> bool:
        """True if the session has queued work or a turn currently running.

        Used by callers and observers to decide whether a freshly
        enqueued item will run immediately ("running") or wait behind
        existing work ("queued"). The check is best-effort: between the
        snapshot and the next event-loop tick the worker may have
        advanced, but the value is purely informational.
        """
        if self.mailbox_depth(session_id) > 0:
            return True
        session = self._sessions.get(session_id)
        return session is not None and session.status == "running"

    async def drain(self, session_id: str) -> None:
        """Wait for all queued items in the session's mailbox to complete.

        Tests use this to synchronize on background work without polling.
        Returns immediately if the session has no mailbox or the
        mailbox is already empty and idle.
        """
        mb = self._mailboxes.get(session_id)
        if mb is not None:
            await mb.join()

    async def shutdown(self) -> None:
        """Cancel every session worker and await its termination.

        Called from test fixtures and from the server's shutdown path
        to avoid leaking pending tasks across event loops. Workers
        block on ``mailbox.get()`` between items, so cancellation
        unblocks them cleanly. Items already in flight inside
        ``_execute_skill_turn`` will receive ``CancelledError`` and
        the streaming-log "cancelled" trailer / inbox notification
        paths will fire on the way out.
        """
        workers = list(self._workers.values())
        for w in workers:
            w.cancel()
        for w in workers:
            with suppress(asyncio.CancelledError, Exception):
                await w
        self._workers.clear()
        self._mailboxes.clear()

    def touch(self, session_id: str) -> None:
        """Bump ``last_active`` to now for an in-cache session.

        No-op if the session is not in cache. Does not persist on its
        own; the next ``save`` writes the new value to disk.
        """
        session = self._sessions.get(session_id)
        if session is not None:
            session.last_active = datetime.now(UTC).isoformat()

    def _persist(self, session: Session) -> None:
        """Atomically write session to disk via tempfile + ``os.replace``.

        The payload is written to a sibling tempfile in the same directory,
        ``fsync``'d, then atomically renamed over the destination. A crash
        between fsync and rename leaves the prior file intact; a crash after
        rename leaves the new file intact. No half-written file is ever
        observable through ``_session_path``. The tempfile is cleaned up on
        any failure so it cannot leak.
        """
        path = self._session_path(session.session_id)
        payload = json.dumps(session.to_dict(), indent=2)
        fd, tmp_path_str = tempfile.mkstemp(
            prefix=f".{session.session_id}.",
            suffix=".tmp",
            dir=self.session_dir,
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            with suppress(FileNotFoundError):
                tmp_path.unlink()
            raise

    def save(self, session: Session) -> None:
        """Save an updated session."""
        self._sessions[session.session_id] = session
        self._persist(session)

    def log_path(self, session_id: str) -> Path:
        """Return the path to the streaming log file for a session."""
        return self.session_dir / f"{session_id}.log"

    def tail(
        self,
        session_id: str,
        offset: int = 0,
        max_bytes: int = 16384,
    ) -> tuple[str, int]:
        """Read new bytes from a session log.

        Returns ``(text, new_offset)``. If the log does not exist, returns
        ``("", 0)``. Partial UTF-8 sequences at read boundaries are decoded
        with ``errors="replace"``.
        """
        path = self.log_path(session_id)
        if not path.exists():
            return ("", 0)
        with path.open("rb") as f:
            f.seek(offset)
            data = f.read(max_bytes)
        return (data.decode("utf-8", errors="replace"), offset + len(data))
