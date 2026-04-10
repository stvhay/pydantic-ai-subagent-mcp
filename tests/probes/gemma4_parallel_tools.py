#!/usr/bin/env python3
"""Probe: do gemma4 models emit parallel tool_calls and handle multi-tool history?

Tests four gemma4 model tags against a live Ollama endpoint. For each model:

1. Turn 1 -- Send a prompt requiring two independent tool calls. Record whether
   the model emits multiple tool_calls in one assistant turn ("multi"), a single
   call ("single"), or none ("none").

2. Turn 2a (current shape) -- One assistant message with tool_calls=[A, B],
   followed by two separate tool messages. Ask the model to synthesize.

3. Turn 2b (fan-out shape) -- Two assistant messages each with one tool_call,
   each immediately followed by its tool message. Ask the model to synthesize.

If Turn 1 only produced a single call, we synthesize the multi-call assistant
message so ingestion shapes can still be tested.

Results are written to a JSON file and a summary matrix is printed to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from typing import Any

# Ensure the project src is importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pydantic_ai_subagent_mcp.ollama import OllamaClient  # noqa: E402

# ---------------------------------------------------------------------------
# Tool definitions (Ollama format)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tz": {"type": "string", "description": "IANA timezone string"},
                },
                "required": ["tz"],
            },
        },
    },
]

# Fake tool results keyed by function name.
FAKE_RESULTS: dict[str, str] = {
    "get_weather": "72\u00b0F sunny",
    "get_time": "14:32 JST",
}

SYSTEM_MSG = "You are a helpful assistant. Use the provided tools when needed."
USER_MSG = (
    "What time is it in Asia/Tokyo AND what's the weather in Paris? "
    "Use tools for both."
)

DEFAULT_MODELS = ["gemma4:e2b", "gemma4:e4b", "gemma4:26b", "gemma4:31b"]
NUM_CTX = 8192


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_call(name: str, args: dict[str, str]) -> dict[str, Any]:
    """Build a single tool_call dict in Ollama wire format."""
    return {"function": {"name": name, "arguments": args}}


def _tool_result_msg(name: str, content: str) -> dict[str, Any]:
    """Build a tool-result message (role=tool)."""
    return {"role": "tool", "tool_name": name, "content": content}


def _resolve_tool_result(tc: dict[str, Any]) -> tuple[str, str]:
    """Given a tool_call dict, return (name, fake_result)."""
    fn = tc.get("function", {})
    name = fn.get("name", "")
    return name, FAKE_RESULTS.get(name, f"<unknown tool {name}>")


def _coherence_check(text: str) -> str:
    """Quick heuristic: does the final answer mention both results?"""
    lower = text.lower()
    has_weather = any(w in lower for w in ["72", "sunny", "weather"])
    has_time = any(w in lower for w in ["14:32", "jst", "time"])
    if has_weather and has_time:
        return "ok"
    if has_weather or has_time:
        return "partial"
    return "incoherent"


# Canonical tool calls for both tools (used when synthesizing).
WEATHER_TC = _tool_call("get_weather", {"city": "Paris"})
TIME_TC = _tool_call("get_time", {"tz": "Asia/Tokyo"})
ALL_TCS = [WEATHER_TC, TIME_TC]


# ---------------------------------------------------------------------------
# Per-model probe
# ---------------------------------------------------------------------------


async def probe_model(
    client: OllamaClient, model: str
) -> dict[str, Any]:
    """Run the full probe for one model. Returns a result dict."""
    result: dict[str, Any] = {"model": model, "error": None}

    messages_base = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_MSG},
    ]

    # --- Turn 1: Does the model emit parallel tool_calls? ---
    print(f"  [turn1] Sending prompt to {model} ...", flush=True)
    t0 = time.monotonic()
    turn1 = await client.chat_turn(
        model=model,
        messages=list(messages_base),
        tools=TOOLS,
        num_ctx=NUM_CTX,
    )
    t1_elapsed = time.monotonic() - t0
    print(f"  [turn1] Done in {t1_elapsed:.1f}s  "
          f"tool_calls={len(turn1.tool_calls)}  "
          f"done_reason={turn1.done_reason}", flush=True)

    n_calls = len(turn1.tool_calls)
    if n_calls == 0:
        result["emission"] = "none"
    elif n_calls == 1:
        result["emission"] = "single"
    else:
        result["emission"] = "multi"

    result["turn1_tool_calls"] = turn1.tool_calls
    result["turn1_content"] = turn1.content
    result["turn1_done_reason"] = turn1.done_reason
    result["turn1_elapsed_s"] = round(t1_elapsed, 2)

    if n_calls == 0 and not turn1.content:
        # Model returned nothing useful; skip ingestion tests.
        result["ingestion_2a"] = "skipped"
        result["ingestion_2b"] = "skipped"
        return result

    # Build the multi-tool-call assistant message.
    # If the model emitted both calls, use them directly.
    # If it emitted only one, synthesize the missing one.
    if n_calls >= 2:
        actual_tcs = turn1.tool_calls[:2]
    elif n_calls == 1:
        # Use the one the model actually emitted + synthesize the other.
        actual_name = turn1.tool_calls[0].get("function", {}).get("name", "")
        if actual_name == "get_weather":
            actual_tcs = [turn1.tool_calls[0], TIME_TC]
        else:
            actual_tcs = [turn1.tool_calls[0], WEATHER_TC]
    else:
        # No tool calls -- use fully synthetic ones.
        actual_tcs = list(ALL_TCS)

    # Resolve results for each tool call.
    tc_results = [_resolve_tool_result(tc) for tc in actual_tcs]

    # Assistant message with all tool calls.
    assistant_multi = {
        "role": "assistant",
        "content": turn1.content or "",
        "tool_calls": actual_tcs,
    }

    # --- Turn 2a: Current shape (one assistant + N tool messages) ---
    print(f"  [turn2a] Testing current ingestion shape ...", flush=True)
    msgs_2a = list(messages_base) + [assistant_multi]
    for name, res in tc_results:
        msgs_2a.append(_tool_result_msg(name, res))

    try:
        t0 = time.monotonic()
        turn2a = await client.chat_turn(
            model=model,
            messages=msgs_2a,
            tools=TOOLS,
            num_ctx=NUM_CTX,
        )
        t2a_elapsed = time.monotonic() - t0
        coherence_2a = _coherence_check(turn2a.content)
        print(f"  [turn2a] Done in {t2a_elapsed:.1f}s  "
              f"coherence={coherence_2a}  "
              f"content_len={len(turn2a.content)}", flush=True)
        result["ingestion_2a"] = coherence_2a
        result["turn2a_content"] = turn2a.content
        result["turn2a_elapsed_s"] = round(t2a_elapsed, 2)
        result["turn2a_extra_tool_calls"] = len(turn2a.tool_calls)
    except Exception as exc:
        print(f"  [turn2a] ERROR: {exc}", flush=True)
        result["ingestion_2a"] = "error"
        result["turn2a_error"] = str(exc)

    # --- Turn 2b: Fan-out shape (N x [assistant+tool]) ---
    print(f"  [turn2b] Testing fan-out ingestion shape ...", flush=True)
    msgs_2b = list(messages_base)
    for tc, (name, res) in zip(actual_tcs, tc_results):
        msgs_2b.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [tc],
        })
        msgs_2b.append(_tool_result_msg(name, res))

    try:
        t0 = time.monotonic()
        turn2b = await client.chat_turn(
            model=model,
            messages=msgs_2b,
            tools=TOOLS,
            num_ctx=NUM_CTX,
        )
        t2b_elapsed = time.monotonic() - t0
        coherence_2b = _coherence_check(turn2b.content)
        print(f"  [turn2b] Done in {t2b_elapsed:.1f}s  "
              f"coherence={coherence_2b}  "
              f"content_len={len(turn2b.content)}", flush=True)
        result["ingestion_2b"] = coherence_2b
        result["turn2b_content"] = turn2b.content
        result["turn2b_elapsed_s"] = round(t2b_elapsed, 2)
        result["turn2b_extra_tool_calls"] = len(turn2b.tool_calls)
    except Exception as exc:
        print(f"  [turn2b] ERROR: {exc}", flush=True)
        result["ingestion_2b"] = "error"
        result["turn2b_error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Decision matrix
# ---------------------------------------------------------------------------


def print_decision_matrix(results: list[dict[str, Any]]) -> None:
    """Print a summary table to stdout."""
    print()
    print("=" * 78)
    print("DECISION MATRIX")
    print("=" * 78)
    header = f"{'Model':<16} {'Emission':<10} {'2a (current)':<14} {'2b (fan-out)':<14}"
    print(header)
    print("-" * len(header))
    for r in results:
        model = r.get("model", "?")
        emission = r.get("emission", "?")
        ing_2a = r.get("ingestion_2a", "?")
        ing_2b = r.get("ingestion_2b", "?")
        if r.get("error"):
            print(f"{model:<16} ERROR: {r['error']}")
        else:
            print(f"{model:<16} {emission:<10} {ing_2a:<14} {ing_2b:<14}")
    print("=" * 78)

    # Recommendation
    print()
    print("RECOMMENDATION:")
    # Check if any model does multi-emission
    multi_models = [r["model"] for r in results if r.get("emission") == "multi"]
    if multi_models:
        print(f"  Models emitting parallel calls: {', '.join(multi_models)}")
    else:
        print("  No models emitted parallel tool_calls natively.")

    # Check which ingestion shape works better
    ok_2a = sum(1 for r in results if r.get("ingestion_2a") in ("ok", "partial"))
    ok_2b = sum(1 for r in results if r.get("ingestion_2b") in ("ok", "partial"))
    err_2a = sum(1 for r in results if r.get("ingestion_2a") == "error")
    err_2b = sum(1 for r in results if r.get("ingestion_2b") == "error")

    print(f"  Shape 2a (current):  {ok_2a} coherent, {err_2a} errors")
    print(f"  Shape 2b (fan-out):  {ok_2b} coherent, {err_2b} errors")

    if err_2a > 0 and err_2b == 0:
        print("  -> Fan-out shape (2b) is more reliable; consider adopting it.")
    elif err_2b > 0 and err_2a == 0:
        print("  -> Current shape (2a) is more reliable; keep it.")
    elif ok_2a >= ok_2b:
        print("  -> Current shape (2a) works at least as well; no change needed.")
    else:
        print("  -> Fan-out shape (2b) shows better coherence; consider adopting it.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe gemma4 parallel tool_calls behavior"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model tags to test (default: all four gemma4 tags)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "gemma4_parallel_tools.out.json"
        ),
        help="Output JSON file path",
    )
    args = parser.parse_args()

    print(f"Ollama endpoint: {args.base_url}")
    print(f"Models: {args.models}")
    print(f"Output: {args.out}")
    print()

    client = OllamaClient(args.base_url, timeout=600.0)
    all_results: list[dict[str, Any]] = []

    try:
        for model in args.models:
            print(f"--- Probing {model} ---", flush=True)
            try:
                result = await probe_model(client, model)
            except Exception as exc:
                print(f"  FATAL ERROR for {model}: {exc}", flush=True)
                traceback.print_exc()
                result = {"model": model, "error": str(exc)}
            all_results.append(result)
            print()
    finally:
        await client.aclose()

    # Write results
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results written to {args.out}")

    print_decision_matrix(all_results)


if __name__ == "__main__":
    asyncio.run(main())
