"""Session management with UUID-keyed transcripts using native pydantic-ai messages."""

from __future__ import annotations

import json
import uuid
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
    """Persists sessions to disk as JSON files keyed by UUID."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}

    def _session_path(self, session_id: str) -> Path:
        return self.session_dir / f"{session_id}.json"

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

    def list_sessions(self) -> list[dict[str, str]]:
        """List all session IDs and metadata."""
        sessions: list[dict[str, str]] = []
        for path in self.session_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                sessions.append({
                    "session_id": data["session_id"],
                    "skill_name": data["skill_name"],
                    "model": data["model"],
                    "created_at": data["created_at"],
                    "message_count": str(len(data.get("messages", []))),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def _persist(self, session: Session) -> None:
        """Write session to disk."""
        path = self._session_path(session.session_id)
        path.write_text(json.dumps(session.to_dict(), indent=2))

    def save(self, session: Session) -> None:
        """Save an updated session."""
        self._sessions[session.session_id] = session
        self._persist(session)
