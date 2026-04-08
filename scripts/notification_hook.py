#!/usr/bin/env python3
"""Drain the subagent-mcp completion inbox and emit notification blocks.

Runs as a Claude Code ``UserPromptSubmit`` hook. Each invocation:

1. Reads the cursor file at ``{inbox_dir}/.cursor`` (empty on first run).
2. Lists ``{inbox_dir}/*.json`` notification records sorted by filename
   (uuid7 lexicographic order = arrival/time order).
3. Filters to records strictly newer than the cursor.
4. Prints a ``<subagent-mcp-notification>`` block per unread record to
   stdout, which Claude Code injects into the next-turn context.
5. Atomically updates the cursor to the newest emitted notification id.

This script is intentionally **standalone**: it does not import the
``pydantic_ai_subagent_mcp`` package. The on-disk inbox format is
deliberately trivial (one JSON file per record, uuid7-named) precisely
so a hook bridge can drain it without runtime coupling to the server.
See ``src/pydantic_ai_subagent_mcp/inbox.py`` for the format spec.

Wire it up in ``.claude/settings.json``::

    {
      "hooks": {
        "UserPromptSubmit": [
          {
            "hooks": [
              {
                "type": "command",
                "command": "/abs/path/to/scripts/notification-hook.sh"
              }
            ]
          }
        ]
      }
    }

Environment:
  ``SUBAGENT_MCP_INBOX_DIR`` -- inbox directory (default: ``.subagent-inbox``)

Exit code is always 0 on a recognized condition: a hook that crashes
or exits non-zero would break the user's prompt submit, which is
strictly worse than dropping a notification (which will redeliver
on the next prompt anyway, since cursor advancement is the only state
the hook owns).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO

DEFAULT_INBOX_DIR = ".subagent-inbox"
CURSOR_FILE = ".cursor"

# Cap notifications per invocation. Each block is up to ~600 chars
# (500-char summary + framing), so 10 keeps the injected context under
# ~6KB even on a fully saturated batch. A backlog larger than this
# drains across successive prompts -- the cursor advances by `limit`
# each call so steady-state delivery is preserved.
READ_LIMIT = 10

# Canonical UUID string length (8-4-4-4-12). Used as a sanity check on
# the cursor file: anything that doesn't look like a UUID is treated as
# "no cursor" so a corrupted cursor self-heals on next run instead of
# silently masking new notifications forever.
UUID_LEN = 36
UUID_DASH_COUNT = 4


def _read_cursor(cursor_path: Path) -> str:
    """Return the cursor value, or ``""`` if missing/unreadable/invalid.

    A garbage cursor (truncated, binary, hand-edited) self-heals to
    "no cursor" on the next run rather than blocking delivery.
    """
    if not cursor_path.exists():
        return ""
    try:
        text = cursor_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""
    # Loose UUID shape check -- not a strict parse, just enough to
    # reject obvious corruption like empty files or stray text.
    if len(text) != UUID_LEN or text.count("-") != UUID_DASH_COUNT:
        return ""
    return text


def _write_cursor(cursor_path: Path, head: str) -> None:
    """Atomically write the cursor via tempfile + ``os.replace``."""
    fd, tmp_str = tempfile.mkstemp(
        prefix=".cursor.",
        suffix=".tmp",
        dir=cursor_path.parent,
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(head)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, cursor_path)
    except BaseException:
        # Clean up the tempfile on any failure (including the rename
        # itself raising); it must not pollute the inbox dir.
        with suppress(FileNotFoundError):
            tmp.unlink()
        raise


def _load_records(
    inbox_dir: Path, since: str, limit: int
) -> list[dict[str, Any]]:
    """Return notification records strictly newer than ``since``.

    Mirrors ``Inbox.read``: empty cursor returns the most recent
    ``limit`` records (tail view, oldest-first within the slice);
    a cursor returns up to ``limit`` records with stem strictly
    greater than ``since``, oldest-first. Corrupt or unreadable
    files are skipped silently -- a single broken record must not
    block the rest of the inbox from draining.
    """
    files = sorted(
        p for p in inbox_dir.glob("*.json") if not p.name.startswith(".")
    )
    # Two distinct semantics -- cursor drain vs tail view -- kept as
    # separate branches rather than a ternary so the difference is
    # legible at a glance. Mirrors Inbox.read in the package.
    if since:  # noqa: SIM108
        files = [p for p in files if p.stem > since][:limit]
    else:
        files = files[-limit:]
    records: list[dict[str, Any]] = []
    for path in files:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _xml_escape(value: Any) -> str:
    """Escape a value for safe inclusion in an XML attribute or text node.

    The hook output is parsed by humans and language models, not by a
    strict XML parser, but escaping &/</>/" defends against summaries
    that contain literal angle brackets (e.g. an agent that emitted
    HTML or a quoted exception traceback) breaking the framing.
    """
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_notification(record: dict[str, Any]) -> str:
    """Render one record as a ``<subagent-mcp-notification>`` block."""
    sid = _xml_escape(record.get("session_id", ""))
    skill = _xml_escape(record.get("skill", ""))
    model = _xml_escape(record.get("model", ""))
    status = _xml_escape(record.get("status", ""))
    timestamp = _xml_escape(record.get("timestamp", ""))
    summary = _xml_escape(record.get("summary", ""))
    return (
        f"<subagent-mcp-notification "
        f'session_id="{sid}" skill="{skill}" model="{model}" '
        f'status="{status}" timestamp="{timestamp}">\n'
        f"{summary}\n"
        f"</subagent-mcp-notification>"
    )


def main(inbox_dir: Path, stdout: TextIO) -> int:
    """Drain the inbox and emit notification blocks. Returns exit code."""
    if not inbox_dir.exists():
        # Server has never run in this CWD, or inbox lives elsewhere.
        # Silent no-op so the hook is harmless on projects that don't
        # use subagent-mcp.
        return 0

    cursor_path = inbox_dir / CURSOR_FILE
    cursor = _read_cursor(cursor_path)

    records = _load_records(inbox_dir, since=cursor, limit=READ_LIMIT)
    if not records:
        return 0

    for record in records:
        stdout.write(_format_notification(record))
        stdout.write("\n")
    stdout.flush()

    # Advance the cursor to the latest notification we just emitted.
    # If persistence fails, the same notifications will redeliver on
    # the next prompt -- at-least-once delivery, idempotent consumer.
    new_head = str(records[-1].get("notification_id", ""))
    if new_head and new_head != cursor:
        # Best-effort: if persistence fails the same notifications
        # will redeliver on the next prompt -- at-least-once delivery,
        # idempotent consumer.
        with suppress(OSError):
            _write_cursor(cursor_path, new_head)
    return 0


if __name__ == "__main__":
    inbox_dir = Path(
        os.environ.get("SUBAGENT_MCP_INBOX_DIR", DEFAULT_INBOX_DIR)
    )
    # Drain stdin: Claude Code passes hook event JSON on stdin. We
    # don't need it, but reading it prevents the writer from seeing
    # EPIPE if the kernel pipe buffer fills.
    with suppress(OSError):
        sys.stdin.read()
    sys.exit(main(inbox_dir=inbox_dir, stdout=sys.stdout))
