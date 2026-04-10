"""Native Ollama client for /api/chat with NDJSON streaming.

A thin httpx wrapper around Ollama's native chat endpoint. No SDK, no
provider abstraction, no compat layer. The wire format is Ollama's
documented JSON shape; messages are plain ``dict[str, Any]`` so they
serialize to disk and to the wire without translation.

Why /api/chat and not /v1/chat/completions:
    The OpenAI compat shim at /v1 strips Ollama-specific options (notably
    ``num_ctx``), so there is no way to lift the default 4096 context
    window through it. /api/chat honours every option Ollama supports
    and returns gemma4's ``thinking`` field as a separate top-level
    key instead of merging it into ``content``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

# Async callable type alias used by chat_turn's on_content_delta param.
ContentDeltaCallback = Callable[[str], Awaitable[None]]

# Recommended sampling params for gemma4 per Google's model card.
# Override per-call only if you know what you're doing.
GEMMA4_TEMPERATURE = 1.0
GEMMA4_TOP_P = 0.95
GEMMA4_TOP_K = 64

# Sane default context window: bigger than any single skill prompt we
# realistically build (~6k tokens for system+30 tools+user), with room
# for several turns of tool output. Override via config or per-call.
DEFAULT_NUM_CTX = 32768


@dataclass
class StreamChunk:
    """One NDJSON line from /api/chat parsed into the fields the loop cares about.

    Streaming chunks carry incremental deltas; the final chunk has
    ``done=True`` and the accumulated counts. ``tool_calls`` only
    appears in the final chunk for gemma4 — earlier chunks carry text
    deltas.
    """

    delta_content: str = ""
    delta_thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    done_reason: str = ""
    prompt_eval_count: int = 0
    eval_count: int = 0
    total_duration_ns: int = 0


@dataclass
class TurnResult:
    """The accumulated outcome of one /api/chat call.

    Built by ``OllamaClient.chat_turn`` from the stream of ``StreamChunk``s.
    Carries everything the agent loop needs to decide whether to call
    tools or return.
    """

    content: str
    thinking: str
    tool_calls: list[dict[str, Any]]
    prompt_eval_count: int
    eval_count: int
    total_duration_ns: int
    done_reason: str

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def as_assistant_message(self) -> dict[str, Any]:
        """Build the assistant message dict to append to history.

        Per gemma4 docs, ``thinking`` is only carried forward in
        history during an active tool-call sequence — otherwise it
        gets stripped between turns. Callers handle that policy; this
        method just emits the full message.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.thinking:
            msg["thinking"] = self.thinking
        return msg


@runtime_checkable
class ChatClient(Protocol):
    """Structural interface for anything that can drive one chat turn.

    ``run_agent`` depends on this Protocol rather than the concrete
    ``OllamaClient``, so tests can supply a fake without ``type: ignore``.
    """

    async def chat_turn(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = GEMMA4_TEMPERATURE,
        top_p: float = GEMMA4_TOP_P,
        top_k: int = GEMMA4_TOP_K,
        on_content_delta: ContentDeltaCallback | None = None,
    ) -> TurnResult: ...


class OllamaClient:
    """Stateless client for Ollama /api/chat with streaming.

    One instance can serve every concurrent agent run; nothing here
    holds per-call state. Pass a shared ``httpx.AsyncClient`` to pool
    connections across runs.
    """

    def __init__(
        self,
        base_url: str,
        *,
        http: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.AsyncClient(timeout=timeout)
        self._owns_http = http is None

    async def aclose(self) -> None:
        """Close the underlying http client if we own it."""
        if self._owns_http:
            await self._http.aclose()

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = GEMMA4_TEMPERATURE,
        top_p: float = GEMMA4_TOP_P,
        top_k: int = GEMMA4_TOP_K,
    ) -> AsyncIterator[StreamChunk]:
        """POST /api/chat with stream=true and yield parsed NDJSON chunks.

        Each line of the response is one chunk. Text chunks carry
        ``delta_content`` and/or ``delta_thinking``; the final chunk
        has ``done=True`` and (for tool-using turns) the accumulated
        ``tool_calls`` array.

        Raises ``httpx.HTTPStatusError`` on non-2xx status; the caller
        is responsible for retry/backoff policy if any.
        """
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_ctx": num_ctx,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
        }
        if tools:
            body["tools"] = tools

        async with self._http.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                yield _parse_chunk(line)

    async def chat_turn(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = GEMMA4_TEMPERATURE,
        top_p: float = GEMMA4_TOP_P,
        top_k: int = GEMMA4_TOP_K,
        on_content_delta: ContentDeltaCallback | None = None,
    ) -> TurnResult:
        """Run one turn end-to-end and return the accumulated result.

        Convenience helper that consumes ``chat_stream`` and accumulates
        text + thinking + tool_calls. If ``on_content_delta`` is
        provided, each text delta is also pushed to the callback so
        the caller can stream output to a log/socket without writing
        the accumulator loop themselves.
        """
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        prompt_eval = 0
        eval_ct = 0
        duration_ns = 0
        done_reason = ""

        async for chunk in self.chat_stream(
            model=model,
            messages=messages,
            tools=tools,
            num_ctx=num_ctx,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ):
            if chunk.delta_content:
                content_parts.append(chunk.delta_content)
                if on_content_delta is not None:
                    await on_content_delta(chunk.delta_content)
            if chunk.delta_thinking:
                thinking_parts.append(chunk.delta_thinking)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            if chunk.done:
                prompt_eval = chunk.prompt_eval_count
                eval_ct = chunk.eval_count
                duration_ns = chunk.total_duration_ns
                done_reason = chunk.done_reason

        return TurnResult(
            content="".join(content_parts),
            thinking="".join(thinking_parts),
            tool_calls=tool_calls,
            prompt_eval_count=prompt_eval,
            eval_count=eval_ct,
            total_duration_ns=duration_ns,
            done_reason=done_reason,
        )


def _parse_chunk(line: str) -> StreamChunk:
    """Parse one NDJSON line from /api/chat into a ``StreamChunk``.

    Tolerant of fields being absent — Ollama omits empty fields rather
    than emitting null/empty strings, so we default everything.
    """
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        # An unparseable line is treated as an empty chunk rather than
        # raising — the stream may emit transient lines we don't care
        # about, and we don't want one bad line to kill a long run.
        return StreamChunk()

    msg = d.get("message") or {}
    return StreamChunk(
        delta_content=msg.get("content", "") or "",
        delta_thinking=msg.get("thinking", "") or "",
        tool_calls=msg.get("tool_calls") or [],
        done=bool(d.get("done", False)),
        done_reason=d.get("done_reason", "") or "",
        prompt_eval_count=int(d.get("prompt_eval_count", 0) or 0),
        eval_count=int(d.get("eval_count", 0) or 0),
        total_duration_ns=int(d.get("total_duration", 0) or 0),
    )
