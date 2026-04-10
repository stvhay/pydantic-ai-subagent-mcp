"""Pytest configuration for subagent-mcp tests."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip @pytest.mark.live tests unless SUBAGENT_MCP_LIVE_TESTS=1."""
    if os.environ.get("SUBAGENT_MCP_LIVE_TESTS", "") == "1":
        return
    skip_live = pytest.mark.skip(reason="live tests disabled (set SUBAGENT_MCP_LIVE_TESTS=1)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
