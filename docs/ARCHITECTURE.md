# Architecture

## Overview

pydantic-ai-subagent-mcp is an MCP server that bridges Claude Code skills to local
Ollama models. It sits between Claude Code (MCP client) and Ollama (LLM backend),
using a native multi-turn agent loop over Ollama's /api/chat endpoint.

## System Context

```
Claude Code (MCP client)
  |
  +-- stdio --> subagent-mcp (FastMCP server)
  |               |
  |               +-- run_agent loop (agent.py)
  |               |    +-- OllamaClient.chat_turn --> Ollama /api/chat
  |               |    +-- Tool[] (unified dispatch)
  |               |         +-- Built-in tools (file I/O, search, shell)
  |               |         +-- MCP-proxied tools (thin shims via MCPToolLoader)
  |               |
  |               +-- MCPToolLoader (mcp_loader.py)
  |               |    +-- Long-lived MCP child sessions (AsyncExitStack)
  |               |    \-- srclight, self (recursive), user-declared servers
  |               |
  |               +-- SessionStore (JSON on disk, UUID-keyed)
  |               |    +-- per-session mailbox (asyncio.Queue) + worker task
  |               |    +-- server-wide asyncio.Semaphore (max_concurrent_runs)
  |               |    \-- per-session .log files for streaming output
  |               |
  |               \-- Inbox (.subagent-inbox/, uuid7-named JSON files)
  |                     ^
  |                     | (atomic per-record writes on every turn exit)
  |                     |
  |                     +-- read_inbox MCP tool (pull mode)
  |                     |
  +-- UserPromptSubmit hook --> notification_hook.py (push mode, cursor-tracked)
  |
  +-- srclight (optional, code indexing MCP server -- loaded via MCPToolLoader)
```

## Key Architectural Decisions

### ADR-1: Native /api/chat loop over pydantic-ai

> *Supersedes the original ADR-1 ("pydantic-ai as agent framework") as of #25.*
>
> In the context of running multi-turn tool-using conversations against Ollama,
> facing a choice between keeping pydantic-ai or owning the loop directly, we
> replaced pydantic-ai with a hand-rolled multi-turn loop (`run_agent` in
> `agent.py`) driving `OllamaClient.chat_turn` (ollama.py). Two bugs motivated
> the switch: (1) pydantic-ai's `ModelMessage` types did not round-trip cleanly
> to Ollama's /api/chat wire shape, requiring fragile translation on every turn;
> (2) the OpenAI-compat shim at /v1 (which pydantic-ai used) strips
> Ollama-specific options like `num_ctx`, making it impossible to raise the
> default 4096 context window. We accept owning the multi-turn loop, tool
> dispatch, and streaming accumulation ourselves, with no framework-provided
> retry or backoff.

### ADR-2: Ollama-native message history

> *Supersedes the original ADR-2 ("Native message history over text
> concatenation") as of #25.*
>
> In the context of persisting multi-turn sessions, facing a choice between a
> typed message model with a serialization adapter or storing Ollama's wire
> shape directly, we chose plain `list[dict[str, Any]]` -- the exact shape
> Ollama's /api/chat expects and returns. Session persistence is
> `json.dumps`/`json.loads` with no translation layer. Resuming a session
> appends to the list and re-calls `run_agent`. We accept that there is no
> typed validation on deserialization; the wire format is the contract.

### ADR-3: Unified Tool dataclass for built-in and MCP tools

> *Supersedes the original ADR-3 ("MCP client for external tool access") as
> of #25.*
>
> In the context of giving the agent loop a uniform tool interface, facing a
> choice between separate dispatch paths for Python tools and MCP tools, we
> chose a single `Tool` dataclass (`agent.py`) with `name`, `description`,
> `parameters` (JSON Schema), and `fn` (async callable). Built-in Python
> tools construct `Tool` directly; external MCP servers are spawned once at
> startup by `MCPToolLoader` (`mcp_loader.py`), which wraps each remote tool
> as a `Tool` with a thin shim that forwards to the long-lived
> `ClientSession`. The agent loop dispatches all tools through the same
> callable interface and never knows the difference. We accept that external
> MCP child processes are held open for the server's entire lifetime rather
> than started per-run.

### ADR-4: Log-based streaming over MCP progress notifications

> In the context of wanting incremental output visibility during long-running
> skill runs, facing a choice between MCP `notifications/progress` (requires
> client-side `progressToken` handling) and writing deltas to per-session log
> files, we chose the log-file approach to keep the MCP tool contract unchanged
> (callers still receive a complete result) and to make transcripts inspectable
> out-of-band, accepting that clients must poll a `tail_session_log` tool to
> read live output.

### ADR-5: Single-process actor model for session linearizability

> In the context of supporting background runs and concurrent resumes against
> the same session without corrupting transcripts, facing a choice between a
> global lock (serializes everything), per-session locks (linearizable but
> rejects concurrent resumes with nothing to queue them on), and a per-session
> mailbox + worker actor model, we chose per-session mailboxes drained by a
> single long-lived worker task per session. The mailbox FIFO + single-consumer
> worker *is* the linearizability primitive: every turn on a session flows
> through one queue drained by one task, so `session.messages` is only ever
> read or written by the worker and no additional lock is necessary. Distinct
> sessions run on independent workers and never block each other.
> Backpressure has two independent dimensions: a server-wide
> `asyncio.Semaphore(max_concurrent_runs)` owned by the `SessionStore` and
> acquired by the worker around the actual turn execution (so a swarm of
> background launches across many sessions cannot saturate the host) and a
> per-session `mailbox_max_depth` cap that bounds queued work behind an
> in-flight item. The mailbox cap is the only admission-time rejection:
> pushes that would exceed it return `status="mailbox_full"` regardless of
> mode. Foreground and background launches both enqueue unconditionally when
> the mailbox has room; the wait on the server-wide gate happens inside the
> worker at the semaphore acquire (callers simply wait longer). There is no
> admission-time "server too busy" rejection because that would be inherently
> racy against the execution-time acquire -- the mailbox + semaphore pair
> are self-regulating. We accept that this design is single-process by
> construction: the actor model lives entirely inside one event loop and
> would need a different transport (Redis, message queue) to scale across
> processes, which is out of scope for a local MCP server.

### ADR-6: Hook-bridged completion notifications via on-disk outbox

> In the context of wanting Claude Code (the MCP client) to be notified when
> a background subagent run completes, facing a choice between MCP
> server-initiated notifications (not supported by stdio MCP), client-side
> polling of `read_inbox`, and a side-channel via Claude Code hooks, we
> chose a durable on-disk outbox written by the server and drained by a
> standalone `UserPromptSubmit` hook. Each completed turn writes one
> `{notification_id}.json` file under `.subagent-inbox/`, named with a
> uuid7 so lexicographic filename order matches arrival order. The hook
> script (`scripts/notification_hook.py`) reads the directory directly,
> tracks its own cursor at `.subagent-inbox/.cursor`, and emits one
> `<subagent-mcp-notification>` block per unread record into the next
> prompt's context. Properties: at-least-once delivery (the cursor only
> advances after successful emission, so a crash mid-write redelivers on
> the next prompt; consumers must be idempotent), the on-disk format is
> the contract (the hook does not import the package, so the bridge keeps
> working across server upgrades), and the hook is fail-soft (always
> exits 0 so a misconfigured hook can never break a prompt submit). The
> `read_inbox` MCP tool exposes the same outbox for poll-mode consumers.
> Cancellation -- including stops via the `stop_session` tool -- writes
> a `cancelled` notification on the same channel, so observers detect
> stops on their normal completion path without needing a separate
> "session was stopped" event type. We accept that the outbox is a
> file-per-record format (which is heavier than a single append-only
> log) in exchange for atomic per-record writes and trivial idempotency
> via filename ordering.

### ADR-7: gemma4 parallel tool_calls handling

> In the context of the `run_agent` loop (`agent.py`) building multi-turn
> tool-call history for Ollama's /api/chat, facing uncertainty about whether
> gemma4 models emit multiple tool_calls per assistant turn and whether they
> handle the resulting `[assistant(multi), tool, tool, ...]` history shape,
> we verified empirically that all four available gemma4 tags (e2b, e4b,
> 26b A4B MOE, 31b dense) both emit parallel tool_calls and continue
> coherently after the canonical OpenAI-compat history shape. We chose to
> keep the current implementation unchanged and lock the behavior in with
> regression tests.
>
> Probe results (recorded 2026-04-10):
> - gemma4:e2b — emission: multi, ingestion-2a: ok, ingestion-2b: ok
> - gemma4:e4b — emission: multi, ingestion-2a: ok, ingestion-2b: ok
> - gemma4:26b — emission: multi, ingestion-2a: ok, ingestion-2b: ok
> - gemma4:31b — emission: multi, ingestion-2a: ok, ingestion-2b: ok
>
> Consequences: the `run_agent` loop is correct as-is. The behavior is
> locked in by `tests/test_agent_parallel_tools.py` (unit) and the probe
> script at `tests/probes/gemma4_parallel_tools.py` (opt-in live test
> against a real endpoint). An upstream Ollama or model-template change
> that silently alters this contract will be caught by re-running the
> probe or by a future `@pytest.mark.live` regression test.

## Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `server.py` | FastMCP server, tool registration, `_run_skill_streaming` orchestration; per-session worker tasks; backpressure (run semaphore + mailbox cap); `stop_session` |
| `agent.py` | Multi-turn agent loop (`run_agent`), `Tool` dataclass (unified built-in + MCP dispatch), `AgentResult` |
| `ollama.py` | Native Ollama /api/chat client (`OllamaClient`), NDJSON streaming (`chat_stream`), `ChatClient` Protocol, `TurnResult`, `StreamChunk` |
| `mcp_loader.py` | External MCP server lifecycle (`MCPToolLoader`): spawns child processes at startup via `AsyncExitStack`, wraps remote tools as `Tool` shims |
| `config.py` | Configuration from file + environment |
| `skills.py` | Skill discovery from filesystem |
| `session.py` | Session persistence with Ollama-native messages (`list[dict]`); per-session `.log` files for streaming output; per-session mailboxes, worker registry, server-wide run semaphore, and `stop_session` drain |
| `inbox.py` | Durable on-disk outbox of completion notifications (uuid7-named JSON files, atomic per-record writes, cursor-based forward drain) |
| `tools.py` | Built-in tools for file I/O, search, shell |
| `scripts/notification_hook.py` | Standalone Claude Code `UserPromptSubmit` hook bridge that drains the inbox into prompt context (does not import the package) |
