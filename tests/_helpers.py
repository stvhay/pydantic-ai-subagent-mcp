"""Shared test helpers for the pydantic-ai-subagent-mcp test suite."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


async def yield_until(
    predicate: Callable[[], bool],
    *,
    max_ticks: int = 100,
    description: str = "predicate",
) -> None:
    """Yield event-loop ticks until ``predicate()`` returns True.

    Replaces the ``for _ in range(N): await asyncio.sleep(0)`` pattern
    with an explicit wait on observable state. ``asyncio.sleep(0)``
    advances the loop by exactly one tick, so a predicate that
    becomes true after the worker schedules its next await will
    resolve promptly without the caller having to guess how many
    ticks that takes.

    Raises ``AssertionError`` (not ``TimeoutError``) if the predicate
    never becomes true within ``max_ticks`` -- a predicate that never
    fires is a test-logic bug, not a timeout condition.
    """
    for _ in range(max_ticks):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(
        f"yield_until: {description} never became True after {max_ticks} ticks"
    )
