"""Session management with UUID-keyed transcripts in Ollama-native message shape."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("subagent-mcp.session")

# Lifecycle states for an individual session.
#
# - ``idle``: no in-flight turn; safe to resume.
# - ``running``: a turn is currently executing for this session.
# - ``failed``: the most recent turn raised; the session is still resumable
#   (the failure is informational, not a terminal state).
SessionStatus = Literal["idle", "running", "failed"]


@dataclass
class Session:
    """A session with UUID key and Ollama-native message history.

    ``messages`` is a list of plain dicts in /api/chat shape:
    ``{"role": "system|user|assistant|tool", "content": str, ...}``.
    Assistant turns may carry ``tool_calls`` and ``thinking``; tool
    turns carry ``tool_name``. Storing the wire shape directly means
    persistence is plain ``json.dumps``/``json.loads`` and resuming a
    session is appending to the list and re-running the agent loop --
    no serialize/deserialize translation tax.
    """

    session_id: str
    skill_name: str
    model: str
    created_at: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    status: SessionStatus = "idle"
    last_active: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "skill_name": self.skill_name,
            "model": self.model,
            "created_at": self.created_at,
            "status": self.status,
            "last_active": self.last_active or self.created_at,
            "messages": self.messages,
        }


class SessionStore:
    """Persists sessions to disk as JSON files keyed by UUID.

    Concurrency model: each session is an actor. Work items are
    enqueued onto a per-session ``asyncio.Queue`` (mailbox) and drained
    FIFO by a single long-lived worker task. Per-session linearizability
    is guaranteed by the mailbox + single-consumer worker: one turn at
    a time, in submission order, with no other coroutine able to touch
    ``session.messages`` concurrently. Distinct sessions never block
    each other -- they run on independent workers and only contend on
    the store's server-wide ``run_semaphore``.

    Server-wide concurrency is bounded by ``run_semaphore``, which the
    worker acquires around the actual turn execution. The semaphore
    is owned by the store so its lifecycle matches the store's: a
    fresh store gets a fresh semaphore on the loop where it is first
    used, and ``shutdown()`` does not need to reset anything.

    Persistence is atomic (tempfile + ``os.replace``) so a crash
    mid-write leaves either the prior or the next coherent state on
    disk, never a half-written file.
    """

    def __init__(
        self,
        session_dir: str | Path,
        *,
        max_concurrent_runs: int = 4,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        # Per-session mailbox queues. Each session has at most one
        # ``asyncio.Queue`` of pending work items, drained FIFO by a
        # dedicated worker task. The actor model: a session is an
        # addressable mailbox; a resume is a "tell" to that mailbox.
        self._mailboxes: dict[str, asyncio.Queue[Any]] = {}
        # Per-session worker tasks. The worker is lazily started on
        # the first push to the mailbox, lives until ``shutdown()``
        # or ``stop_session`` tears it down, and blocks on
        # ``mailbox.get()`` between items so idle sessions cost one
        # suspended task each. One worker per session is the
        # linearizability primitive: serial draining guarantees no
        # two turns of the same session ever run concurrently.
        self._workers: dict[str, asyncio.Task[None]] = {}
        # Server-wide concurrency gate. Acquired by the worker around
        # the actual turn execution (not around admission), so
        # mailbox queueing is unaffected. ``asyncio.Semaphore`` binds
        # to the event loop on first ``acquire``; constructing it
        # here is safe because we never acquire outside an async
        # context.
        self.run_semaphore = asyncio.Semaphore(max_concurrent_runs)

    def _session_path(self, session_id: str) -> Path:
        return self.session_dir / f"{session_id}.json"

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
        # Messages are stored in Ollama-native shape (list of dicts);
        # no validation layer needed because the agent loop and the
        # wire format are already the same shape.
        messages = list(data.get("messages") or [])
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

        A worker that raises a non-cancellation exception during
        shutdown is logged but does not prevent other workers from
        being torn down -- shutdown must be infallible so we never
        leak tasks across loops.
        """
        workers = list(self._workers.items())
        for _, w in workers:
            w.cancel()
        if workers:
            tasks = [w for _, w in workers]
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._mailboxes.clear()

    async def stop_session(
        self,
        session_id: str,
        on_drop: Callable[[Any], None] | None = None,
    ) -> dict[str, Any]:
        """Stop a session: drain queued items, cancel the worker, await it.

        Returns a structured snapshot of what was actually stopped:

        * ``status`` -- one of ``"stopped"`` (work was cancelled),
          ``"already_idle"`` (no worker / nothing to stop), or
          ``"not_found"`` (no on-disk session for this id).
        * ``in_flight_cancelled`` -- ``1`` if a worker task was alive
          and got cancelled, ``0`` otherwise.
        * ``queued_dropped`` -- number of items pulled out of the
          mailbox without being executed.

        Idempotent by construction: calling on a stopped/idle/missing
        session returns a sensible status without raising. Concurrent
        ``stop_session`` calls on the same session are safe -- the
        second call observes ``has_worker == False`` and returns
        ``already_idle``.

        ``on_drop`` is invoked once per dropped queued item before
        ``task_done`` is called, so the server-side caller can
        cancel any per-item futures (foreground waiters) that would
        otherwise hang. The in-flight item is *not* passed to
        ``on_drop`` -- its future is handled by the worker's own
        cancellation path in ``_session_worker``.
        """
        # Session must exist on disk OR in cache to be considered
        # "found". A bare worker/mailbox without a session record is
        # legitimately "stopped" but not "not_found", so check the
        # session existence after the worker check below.
        has_worker = self.has_worker(session_id)
        mailbox = self._mailboxes.get(session_id)
        on_disk = self._session_path(session_id).exists()
        in_cache = session_id in self._sessions

        if not has_worker and mailbox is None:
            if not on_disk and not in_cache:
                return {
                    "session_id": session_id,
                    "status": "not_found",
                    "in_flight_cancelled": 0,
                    "queued_dropped": 0,
                }
            return {
                "session_id": session_id,
                "status": "already_idle",
                "in_flight_cancelled": 0,
                "queued_dropped": 0,
            }

        # Drain any queued items first so the worker, when it wakes
        # from cancellation, has nothing left to mistakenly process.
        # Each drained item is passed to on_drop so the caller can
        # unblock any waiting foreground future, then task_done is
        # called to keep the queue's join() counter consistent.
        dropped = 0
        if mailbox is not None:
            while not mailbox.empty():
                try:
                    item = mailbox.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if on_drop is not None:
                    with suppress(Exception):
                        on_drop(item)
                mailbox.task_done()
                dropped += 1

        # Cancel and await the worker. The worker may be:
        #   * blocked on mailbox.get() -- cancellation unblocks
        #     immediately, no in-flight item to clean up
        #   * inside _execute_skill_turn -- cancellation propagates
        #     into the streaming code; the worker's own cancellation
        #     handler cancels the in-flight item's future and the
        #     cancelled trailer / inbox notification fire on the way
        #     out
        in_flight = 0
        worker = self._workers.get(session_id)
        if worker is not None and not worker.done():
            in_flight = 1
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                # Expected: we just cancelled it.
                pass
            except Exception:
                # A worker that dies with a non-cancellation exception
                # is a bug worth surfacing, but stop_session is
                # contracted to be idempotent and non-raising --
                # so log it and tear down the rest of the session.
                logger.exception(
                    "session worker for %s raised during shutdown",
                    session_id,
                )

        # Clear registries so the next push to this session starts a
        # fresh worker on a fresh mailbox. Leaving stale entries
        # would let queued items survive a stop, defeating the
        # whole point.
        self._workers.pop(session_id, None)
        self._mailboxes.pop(session_id, None)

        return {
            "session_id": session_id,
            "status": "stopped",
            "in_flight_cancelled": in_flight,
            "queued_dropped": dropped,
        }

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
