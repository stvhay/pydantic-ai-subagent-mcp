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
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter


@dataclass
class Session:
    """A session with UUID key and native pydantic-ai message history."""

    session_id: str
    skill_name: str
    model: str
    created_at: str
    messages: list[ModelMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        messages_json = json.loads(
            ModelMessagesTypeAdapter.dump_json(self.messages)
        )
        return {
            "session_id": self.session_id,
            "skill_name": self.skill_name,
            "model": self.model,
            "created_at": self.created_at,
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
        session = Session(
            session_id=session_id,
            skill_name=skill_name,
            model=model,
            created_at=datetime.now(UTC).isoformat(),
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
        session = Session(
            session_id=data["session_id"],
            skill_name=data["skill_name"],
            model=data["model"],
            created_at=data["created_at"],
            messages=messages,
        )
        self._sessions[session_id] = session
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all session IDs and metadata."""
        sessions: list[dict[str, Any]] = []
        for path in self.session_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                sessions.append({
                    "session_id": data["session_id"],
                    "skill_name": data["skill_name"],
                    "model": data["model"],
                    "created_at": data["created_at"],
                    "message_count": len(data.get("messages", [])),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

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
