"""Durable on-disk outbox of completion notifications for async runs.

Each notification is a small JSON record stored as a single file named
``{notification_id}.json`` where ``notification_id`` is a uuid7. uuid7
is time-ordered and lexicographically sortable, so the on-disk filename
order is also the temporal arrival order. This gives a durable
append-only outbox with O(1) writes, O(n) directory scans, and trivial
atomic semantics: each notification is its own file, written via
tempfile + ``os.replace``, so a partial write is never observable.

Readers use ``read(since=<id>, limit=<n>)`` to fetch notifications
strictly newer than ``since`` and track their own cursor — at-least-once
delivery with an idempotent consumer. This is the durable substrate
that the Phase 4 hook bridge will drain to push completion events into
Claude Code via ``UserPromptSubmit``.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# Outcome of a completed turn from the perspective of the inbox consumer.
# Mirrors the streaming-log trailer vocabulary so observers can correlate
# the structured notification with the raw log file.
NotificationStatus = Literal["ok", "error", "cancelled"]


@dataclass
class Notification:
    """A single completion notification record."""

    notification_id: str  # uuid7, time-ordered
    session_id: str
    skill: str
    model: str
    status: NotificationStatus
    timestamp: str  # ISO8601 UTC
    summary: str  # short response excerpt on success, error message on failure

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Inbox:
    """Durable on-disk outbox of completion notifications.

    Each notification is its own file named ``{uuid7}.json``, atomically
    written via tempfile + ``os.replace``. Reads sort the directory by
    filename (= uuid7 lexicographic order = time order). The on-disk
    layout is intentionally trivial so a shell hook can drain it later
    without needing this module to be importable.
    """

    # Maximum length of the human-visible summary stored in a notification.
    # Long agent outputs would bloat the inbox and the eventual hook
    # injection; truncate so each record stays small enough for a single
    # tool-result-sized payload.
    SUMMARY_MAX_CHARS = 500

    def __init__(self, inbox_dir: str | Path) -> None:
        self.inbox_dir = Path(inbox_dir)
        self.inbox_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        session_id: str,
        skill: str,
        model: str,
        status: NotificationStatus,
        summary: str,
    ) -> Notification:
        """Append a notification atomically and return the stored record."""
        nid = str(uuid.uuid7())
        truncated = summary
        if len(truncated) > self.SUMMARY_MAX_CHARS:
            truncated = truncated[: self.SUMMARY_MAX_CHARS - 3] + "..."
        notification = Notification(
            notification_id=nid,
            session_id=session_id,
            skill=skill,
            model=model,
            status=status,
            timestamp=datetime.now(UTC).isoformat(),
            summary=truncated,
        )
        self._persist(notification)
        return notification

    def _persist(self, notification: Notification) -> None:
        """Atomically write a notification record via tempfile + replace."""
        path = self.inbox_dir / f"{notification.notification_id}.json"
        payload = json.dumps(notification.to_dict(), indent=2)
        fd, tmp_path_str = tempfile.mkstemp(
            prefix=f".{notification.notification_id}.",
            suffix=".tmp",
            dir=self.inbox_dir,
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

    def read(
        self, since: str = "", limit: int = 50
    ) -> tuple[list[dict[str, Any]], str]:
        """Return notifications strictly after ``since``, up to ``limit``.

        Semantics:
          * ``since=""``: return the *most recent* ``limit`` notifications
            (a "tail" view, useful for first-time readers).
          * ``since=<id>``: return up to ``limit`` notifications with id
            strictly greater than ``since``, oldest first (so resuming
            from a cursor advances forward in time).

        Returns ``(records, head)`` where ``head`` is the latest
        ``notification_id`` in the returned batch, or ``since`` if no
        new records were found. Callers should pass ``head`` back as
        their next ``since`` to advance the cursor without gaps or
        duplicates.

        Files that fail to parse are skipped silently — a corrupted
        record must not block the rest of the inbox from draining.
        """
        # uuid7 sorts lexicographically by time, so sorting filenames
        # is equivalent to sorting by timestamp.
        files = sorted(
            p for p in self.inbox_dir.glob("*.json")
            if not p.name.startswith(".")
        )

        # Two distinct semantics — tail vs forward-drain — kept as
        # separate branches rather than a ternary so the difference is
        # legible at a glance.
        if since:  # noqa: SIM108
            files = [p for p in files if p.stem > since][:limit]
        else:
            files = files[-limit:]

        records: list[dict[str, Any]] = []
        for path in files:
            try:
                records.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue

        head = records[-1]["notification_id"] if records else since
        return records, head

    def head(self) -> str:
        """Return the latest ``notification_id`` on disk, or ``""`` if empty."""
        files = sorted(
            p.stem for p in self.inbox_dir.glob("*.json")
            if not p.name.startswith(".")
        )
        return files[-1] if files else ""
