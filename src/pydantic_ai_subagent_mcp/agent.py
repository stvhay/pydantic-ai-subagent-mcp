"""Multi-turn agent loop over Ollama /api/chat with unified tool dispatch.

The loop is the entire framework: ``run_agent`` consumes a list of
``Tool`` objects, drives ``OllamaClient.chat_turn`` until the model
stops calling tools, and returns the final assistant text plus the
full message history.

Why a ``Tool`` dataclass and not a function registry: a tool can be a
plain Python coroutine (built-ins in ``tools.py``) or a thin shim that
forwards to a long-lived MCP ``ClientSession`` (external MCP servers
loaded at startup). Both wear the same shape so the dispatcher does
not have to know the difference. Adding a new transport (HTTP, SSE,
in-process) is a one-line wrapper, not a branch in the loop.

Why messages are ``list[dict[str, Any]]``: Ollama's /api/chat already
speaks JSON. Round-tripping through a typed model class would force a
serialize/deserialize tax on every turn -- and we already paid for
that mistake once with pydantic-ai's ``ModelMessage`` ↔ wire shape
mismatch (Bug 1).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .ollama import (
    DEFAULT_NUM_CTX,
    GEMMA4_TEMPERATURE,
    GEMMA4_TOP_K,
    GEMMA4_TOP_P,
    ChatClient,
    ContentDeltaCallback,
)

logger = logging.getLogger("subagent-mcp.agent")


# Async callable that takes a parsed JSON-object of arguments and
# returns a string result. Tool implementations are responsible for
# their own error handling -- the dispatcher catches anything that
# escapes and converts it into a tool-error message that the model can
# observe and recover from.
ToolFn = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class Tool:
    """One tool the agent can call.

    ``parameters`` is a JSON Schema object exactly as it appears on the
    wire in Ollama's tool format. MCP-loaded tools get this for free
    from ``mcp.types.Tool.inputSchema``; built-in Python tools build it
    by hand in ``tools.py``. Either way, the schema is passed through
    to Ollama unchanged -- no translation layer.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn

    def to_ollama_schema(self) -> dict[str, Any]:
        """Render the tool in Ollama's /api/chat tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class AgentResult:
    """The terminal state of one ``run_agent`` call.

    ``messages`` is the full conversation including the system prompt,
    the user turn, every assistant turn (with tool_calls when present),
    and every tool result. It is in the exact shape Ollama expects on
    the wire, so resuming a session is a matter of appending to the
    list and re-calling ``run_agent``.

    ``turns`` is the number of /api/chat round trips it took to reach
    a no-tool-call assistant turn. Useful for diagnostics; the caller
    can trip a circuit breaker by setting ``max_turns``.
    """

    output: str
    messages: list[dict[str, Any]]
    turns: int
    stopped_reason: str  # "ok" | "max_turns" | "unknown_tool"
    prompt_eval_total: int = 0
    eval_total: int = 0


def _tool_call_args(raw: Any) -> dict[str, Any]:
    """Coerce a tool_calls[i].function.arguments value to a dict.

    Ollama and gemma4 sometimes return the arguments as a JSON string
    instead of an already-parsed object (this is a known oddity of the
    /api/chat tool format). The dispatcher accepts both shapes so a
    well-formed call is never rejected on a stylistic technicality.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def _dispatch_tool(
    name: str,
    args: dict[str, Any],
    tools_by_name: dict[str, Tool],
) -> tuple[str, bool]:
    """Run one tool call and return ``(result_text, was_unknown)``.

    Errors raised by the tool implementation are converted into a
    text payload the model can observe -- never re-raised. This keeps
    the loop running so the model can recover from a bad call rather
    than aborting the whole run.
    """
    tool = tools_by_name.get(name)
    if tool is None:
        avail = ", ".join(sorted(tools_by_name)) or "(none)"
        return f"ERROR: unknown tool {name!r}. Available tools: {avail}", True
    try:
        result = await tool.fn(args)
    except Exception as e:  # noqa: BLE001 — surface to model, not the runtime
        logger.exception("tool %s raised", name)
        return f"ERROR calling {name}: {type(e).__name__}: {e}", False
    return result, False


async def run_agent(
    *,
    client: ChatClient,
    model: str,
    system: str,
    user: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    tools: list[Tool],
    max_turns: int = 10,
    num_ctx: int = DEFAULT_NUM_CTX,
    temperature: float = GEMMA4_TEMPERATURE,
    top_p: float = GEMMA4_TOP_P,
    top_k: int = GEMMA4_TOP_K,
    on_content_delta: ContentDeltaCallback | None = None,
    tool_result_max_chars: int = 8000,
) -> AgentResult:
    """Drive the multi-turn loop until the model stops calling tools.

    ``user`` is convenience for fresh runs: the system+user pair is
    built and the loop starts. ``messages`` is for resuming a prior
    session: the caller passes the existing history in Ollama-native
    shape and the loop appends new turns to it. Exactly one of the
    two must be provided.

    Each turn:
        1. Call /api/chat with the current messages + tool schemas.
        2. Append the assistant message (including ``tool_calls`` and
           ``thinking`` per gemma4 docs -- thoughts are kept across
           turns only when an active tool sequence is in flight).
        3. If no tool calls, return.
        4. Otherwise, dispatch every tool in order, append a ``role:
           tool`` message per result, and loop.

    Tool results are clipped to ``tool_result_max_chars`` to keep the
    context window bounded across long sequences. The default (8KB)
    is comfortable inside the default 32k ``num_ctx``: even with the
    full 30-tool definitions burning ~6KB up front, ten 8KB tool
    results still leave room for the model's own reasoning.
    """
    if messages is None and user is None:
        msg = "run_agent requires either user= or messages="
        raise ValueError(msg)

    convo: list[dict[str, Any]]
    if messages is not None:
        convo = list(messages)
        if user is not None:
            convo.append({"role": "user", "content": user})
    else:
        convo = [
            {"role": "system", "content": system},
            {"role": "user", "content": user or ""},
        ]
        # If the caller passed messages= they bring their own system
        # prompt. Only inject ours when starting fresh.

    tools_by_name = {t.name: t for t in tools}
    tool_schemas = [t.to_ollama_schema() for t in tools] if tools else None

    prompt_eval_total = 0
    eval_total = 0
    stopped_reason = "max_turns"
    final_output = ""
    turns_done = 0

    for _ in range(max_turns):
        result = await client.chat_turn(
            model=model,
            messages=convo,
            tools=tool_schemas,
            num_ctx=num_ctx,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            on_content_delta=on_content_delta,
        )
        turns_done += 1
        # Last turn's prompt_eval_count is the final prompt size (Ollama
        # reports it cumulatively); eval_count is per-turn so we sum it.
        prompt_eval_total = result.prompt_eval_count
        eval_total += result.eval_count

        # Append the assistant message in gemma4-correct shape.
        # ``as_assistant_message`` already drops thinking when there
        # are no tool_calls -- per gemma4 docs, thoughts are only
        # carried forward during active tool sequences.
        convo.append(result.as_assistant_message())

        if not result.has_tool_calls:
            final_output = result.content
            stopped_reason = "ok"
            break

        # Dispatch every tool call in the order the model emitted them.
        # Each result becomes its own ``role: tool`` message; the next
        # turn picks them all up at once.
        unknown_seen = False
        for tc in result.tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            args = _tool_call_args(fn.get("arguments"))
            text, was_unknown = await _dispatch_tool(
                name, args, tools_by_name
            )
            if was_unknown:
                unknown_seen = True
            convo.append({
                "role": "tool",
                "tool_name": name,
                "content": text[:tool_result_max_chars],
            })

        if unknown_seen:
            # We let the model see the error message and try again on
            # the next turn rather than aborting. But we record the
            # observation so callers can spot a model that is wedged
            # on a non-existent tool name.
            stopped_reason = "unknown_tool"
        # else: keep stopped_reason as "max_turns" until we either
        # exit the no-tool-call branch above or run out of turns.

    return AgentResult(
        output=final_output,
        messages=convo,
        turns=turns_done,
        stopped_reason=stopped_reason,
        prompt_eval_total=prompt_eval_total,
        eval_total=eval_total,
    )
