"""Regression tests for parallel tool_calls in run_agent.

These tests verify that the agent loop correctly handles single and multiple
tool_calls per assistant turn, producing the expected message history shape.
Probe results confirmed all four gemma4 models (e2b, e4b, 26b, 31b) emit
multi tool_calls naturally and ingest the current history shape correctly.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai_subagent_mcp.agent import AgentResult, Tool, run_agent
from pydantic_ai_subagent_mcp.ollama import TurnResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeClient:
    """Minimal stand-in for OllamaClient that returns canned TurnResults."""

    def __init__(self, turns: list[TurnResult]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def chat_turn(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        num_ctx: int = 32768,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        on_content_delta: Any = None,
    ) -> TurnResult:
        self.calls.append({"model": model, "messages": list(messages)})
        assert self._turns, "FakeClient ran out of canned turns"
        return self._turns.pop(0)


def _make_tool(name: str, result: str) -> Tool:
    """Create a trivial test tool that returns a fixed string."""

    async def fn(args: dict[str, Any]) -> str:
        return result

    return Tool(
        name=name,
        description=f"Test tool {name}",
        parameters={"type": "object", "properties": {}},
        fn=fn,
    )


def _turn(
    *,
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    thinking: str = "",
) -> TurnResult:
    """Build a canned TurnResult."""
    return TurnResult(
        content=content,
        thinking=thinking,
        tool_calls=tool_calls or [],
        prompt_eval_count=10,
        eval_count=5,
        total_duration_ns=1_000_000,
        done_reason="stop",
    )


def _tc(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Shorthand for a single tool_call dict in Ollama wire format."""
    return {"function": {"name": name, "arguments": arguments or {}}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_single_tool_call_turn_history_shape() -> None:
    """A single tool_call per turn produces the expected message sequence."""
    client = FakeClient([
        _turn(tool_calls=[_tc("alpha")]),
        _turn(content="done"),
    ])
    result: AgentResult = await run_agent(
        client=client,  # type: ignore[arg-type]
        model="test-model",
        system="You are a test agent.",
        user="hello",
        tools=[_make_tool("alpha", "alpha-result")],
    )
    assert result.stopped_reason == "ok"
    assert result.turns == 2

    msgs = result.messages
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


async def test_multi_tool_call_turn_history_shape() -> None:
    """Two tool_calls in one turn produce shape 2a: one assistant, then N tool messages."""
    client = FakeClient([
        _turn(tool_calls=[_tc("alpha"), _tc("beta")]),
        _turn(content="done"),
    ])
    result = await run_agent(
        client=client,  # type: ignore[arg-type]
        model="test-model",
        system="sys",
        user="go",
        tools=[_make_tool("alpha", "a-result"), _make_tool("beta", "b-result")],
    )
    assert result.stopped_reason == "ok"
    assert result.turns == 2

    msgs = result.messages
    roles = [m["role"] for m in msgs]
    # Shape 2a: system, user, assistant(2 tool_calls), tool, tool, assistant(final)
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant"]

    # The assistant message should carry both tool_calls
    assistant_msg = msgs[2]
    assert len(assistant_msg["tool_calls"]) == 2

    # Tool results appear in order
    assert msgs[3]["content"] == "a-result"
    assert msgs[4]["content"] == "b-result"


async def test_tool_result_field_name() -> None:
    """Tool-result messages use 'tool_name' as the identifier field (Ollama format)."""
    client = FakeClient([
        _turn(tool_calls=[_tc("alpha")]),
        _turn(content="done"),
    ])
    result = await run_agent(
        client=client,  # type: ignore[arg-type]
        model="test-model",
        system="sys",
        user="go",
        tools=[_make_tool("alpha", "result")],
    )
    tool_msg = [m for m in result.messages if m["role"] == "tool"][0]
    assert "tool_name" in tool_msg
    assert tool_msg["tool_name"] == "alpha"


async def test_three_parallel_tool_calls() -> None:
    """Three parallel tool_calls all dispatch correctly with results in order."""
    client = FakeClient([
        _turn(tool_calls=[_tc("a"), _tc("b"), _tc("c")]),
        _turn(content="final"),
    ])
    result = await run_agent(
        client=client,  # type: ignore[arg-type]
        model="test-model",
        system="sys",
        user="go",
        tools=[
            _make_tool("a", "r-a"),
            _make_tool("b", "r-b"),
            _make_tool("c", "r-c"),
        ],
    )
    assert result.stopped_reason == "ok"

    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 3
    assert [m["tool_name"] for m in tool_msgs] == ["a", "b", "c"]
    assert [m["content"] for m in tool_msgs] == ["r-a", "r-b", "r-c"]


async def test_multi_tool_call_with_unknown_tool() -> None:
    """When one of multiple parallel tool_calls is unknown, the loop records the error."""
    client = FakeClient([
        _turn(tool_calls=[_tc("alpha"), _tc("nonexistent"), _tc("beta")]),
        _turn(content="recovered"),
    ])
    result = await run_agent(
        client=client,  # type: ignore[arg-type]
        model="test-model",
        system="sys",
        user="go",
        tools=[_make_tool("alpha", "a-ok"), _make_tool("beta", "b-ok")],
    )
    # The loop continues despite the unknown tool; stopped_reason reflects it
    # only if max_turns is not reached first. With 2 canned turns, the second
    # turn returns content without tool_calls, so stopped_reason becomes "ok".
    # But the intermediate state was "unknown_tool" — we verify the error
    # message landed in the history.
    assert result.turns == 2

    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 3

    # The unknown tool produces an error message
    unknown_msg = tool_msgs[1]
    assert unknown_msg["tool_name"] == "nonexistent"
    assert "ERROR: unknown tool" in unknown_msg["content"]

    # The other tools still ran successfully
    assert tool_msgs[0]["content"] == "a-ok"
    assert tool_msgs[2]["content"] == "b-ok"
