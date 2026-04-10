"""PoC MCP server that fires every known notification type.

This is a throwaway proof-of-concept for issue #28.  It registers four
notification types so we can observe which ones Claude Code surfaces to the
assistant conversation.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.message import ServerMessageMetadata, SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification
from pydantic import AnyUrl

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "notification-test",
    instructions=(
        "PoC notification test matrix server.  Call fire_notifications to "
        "send all four notification types once, or start_notification_timer "
        "to send them on a recurring interval."
    ),
)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_counter: int = 0
_status: str = "idle"
_timer_task: asyncio.Task[None] | None = None


def _ts() -> str:
    """Return a compact UTC timestamp for traceability."""
    return datetime.now(UTC).strftime("%H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------


@mcp.resource("notification-test://status")
def get_status() -> str:
    """Return the current notification status string."""
    return _status


# ---------------------------------------------------------------------------
# Core notification helper
# ---------------------------------------------------------------------------


async def _fire_all(
    session: Any,  # ServerSession at runtime
    source: str,
) -> list[str]:
    """Fire all four notification types and return a log of what was sent."""
    global _counter, _status  # noqa: PLW0603

    _counter += 1
    n = _counter
    ts = _ts()
    fired: list[str] = []

    # 1) notifications/message
    msg = f"[{ts}] notification-test #{n} from {source}"
    await session.send_log_message(level="info", data=msg, logger="notification-test")
    fired.append(f"log: {msg}")

    # 2) notifications/progress
    pmsg = f"progress #{n} from {source}"
    await session.send_progress_notification(
        progress_token=f"poc-{n}",
        progress=float(n),
        total=100.0,
        message=pmsg,
    )
    fired.append(f"progress: {pmsg}")

    # 3) notifications/resources/updated
    await session.send_resource_updated(AnyUrl("notification-test://status"))
    _status = f"updated #{n} at {ts} by {source}"
    fired.append(f"resource_updated: {_status}")

    # 4) notifications/claude/channel  (bypass SDK union)
    content = f"[{ts}] claude/channel #{n} from {source}"
    meta = {"source": "notification-test", "counter": n}
    notification = JSONRPCNotification(
        jsonrpc="2.0",
        method="notifications/claude/channel",
        params={"content": content, "meta": meta},
    )
    raw_msg = SessionMessage(
        message=JSONRPCMessage(root=notification),
        metadata=ServerMessageMetadata(related_request_id=None),
    )
    await session.send_message(raw_msg)
    fired.append(f"claude/channel: {content}")

    return fired


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def fire_notifications(ctx: Context[Any, Any, Any]) -> str:
    """Fire all four notification types once and return what was sent."""
    session = ctx.session
    fired = await _fire_all(session, source="fire_notifications")
    return "\n".join(fired)


@mcp.tool()
async def start_notification_timer(
    ctx: Context[Any, Any, Any],
    interval_seconds: int = 15,
) -> str:
    """Start a background timer that fires all notifications every N seconds."""
    global _timer_task  # noqa: PLW0603

    if _timer_task is not None and not _timer_task.done():
        return "Timer already running.  Call stop_notification_timer first."

    session = ctx.session

    async def _loop() -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await _fire_all(session, source="timer")
            except Exception as exc:  # noqa: BLE001
                # Session may have closed; just stop quietly.
                print(f"[notification-test] timer error: {exc}")
                break

    _timer_task = asyncio.create_task(_loop())
    return f"Timer started — firing every {interval_seconds}s."


@mcp.tool()
async def stop_notification_timer() -> str:
    """Cancel the background notification timer."""
    global _timer_task  # noqa: PLW0603

    if _timer_task is None or _timer_task.done():
        _timer_task = None
        return "No timer running."

    _timer_task.cancel()
    _timer_task = None
    return "Timer stopped."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
