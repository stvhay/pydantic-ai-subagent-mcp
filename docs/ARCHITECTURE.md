# Architecture

## Overview

pydantic-ai-subagent-mcp is an MCP server that bridges Claude Code skills to local
Ollama models. It sits between Claude Code (MCP client) and Ollama (LLM backend),
using pydantic-ai as the agent framework.

## System Context

```
Claude Code (MCP client)
  |
  +-- stdio --> subagent-mcp (FastMCP server)
  |               |
  |               +-- pydantic-ai Agent
  |               |    +-- OpenAIChatModel + OllamaProvider --> Ollama
  |               |    +-- Built-in tools (file I/O, search, shell)
  |               |    +-- MCPServerStdio toolsets --> srclight, self (recursive)
  |               |
  |               +-- SessionStore (JSON on disk, UUID-keyed)
  |               |    +-- per-session asyncio.Lock (defense in depth)
  |               |    +-- per-session mailbox (asyncio.Queue) + worker task
  |               |    \-- per-session .log files for streaming output
  |               |
  |               +-- Server-wide asyncio.Semaphore (max_concurrent_runs)
  |               |
  |               \-- Inbox (.subagent-inbox/, uuid7-named JSON files)
  |                     ^
  |                     | (atomic per-record writes on every turn exit)
  |                     |
  |                     +-- read_inbox MCP tool (pull mode)
  |                     |
  +-- UserPromptSubmit hook --> notification_hook.py (push mode, cursor-tracked)
  |
  +-- srclight (optional, code indexing MCP server)
```

## Key Architectural Decisions

### ADR-1: pydantic-ai as agent framework

> In the context of needing an agent framework for Ollama, facing choices between
> raw API calls, LangChain, and pydantic-ai, we chose pydantic-ai to get native
> Ollama support, typed tool definitions, and built-in MCP client capabilities,
> accepting coupling to the pydantic-ai API surface.

### ADR-2: Native message history over text concatenation

> In the context of multi-turn sessions, facing a choice between manually
> concatenating prior messages as text or using pydantic-ai's `message_history`
> parameter, we chose native message history to preserve tool call/result structure
> and let the framework manage context, accepting that session serialization must
> use `ModelMessagesTypeAdapter`.

### ADR-3: MCP client for external tool access

> In the context of giving subagents access to srclight and recursive sub-agents,
> facing a choice between custom HTTP integrations or pydantic-ai's MCP client,
> we chose `MCPServerStdio` toolsets to reuse the MCP protocol and enable the agent
> to call any MCP server, accepting subprocess management overhead.

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
> global lock (serializes everything), per-session locks alone (linearizable
> but rejects concurrent resumes), and a per-session mailbox + worker actor
> model, we chose per-session mailboxes drained by a single long-lived worker
> task per session. The mailbox FIFO + single-consumer worker is the
> linearizability primitive; the per-session `asyncio.Lock` remains as
> defense-in-depth around `_execute_skill_turn` so anything that bypasses the
> mailbox still cannot interleave reads and writes of `session.messages`.
> Backpressure has two independent dimensions: a server-wide
> `asyncio.Semaphore(max_concurrent_runs)` acquired by the worker around the
> actual turn execution (so a swarm of background launches across many
> sessions cannot saturate the host) and a per-session `mailbox_max_depth`
> cap that bounds queued work behind an in-flight item. Background launches
> that hit the semaphore cap return `status="saturated"`; pushes that hit
> the mailbox cap return `status="mailbox_full"`. Foreground (Ask-mode)
> launches always enqueue and end up waiting on the semaphore inside the
> worker -- the natural "wait until done" contract. We accept that this
> design is single-process by construction: the actor model lives entirely
> inside one event loop and would need a different transport (Redis,
> message queue) to scale across processes, which is out of scope for a
> local MCP server.

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

## Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `server.py` | FastMCP server, tool registration, agent orchestration; per-session worker tasks; backpressure (run semaphore + mailbox cap); `stop_session` |
| `config.py` | Configuration from file + environment |
| `skills.py` | Skill discovery from filesystem |
| `session.py` | Session persistence with native pydantic-ai messages; per-session `.log` files for streaming output; per-session locks, mailboxes, worker registry, and `stop_session` drain |
| `inbox.py` | Durable on-disk outbox of completion notifications (uuid7-named JSON files, atomic per-record writes, cursor-based forward drain) |
| `tools.py` | Built-in tools for file I/O, search, shell |
| `scripts/notification_hook.py` | Standalone Claude Code `UserPromptSubmit` hook bridge that drains the inbox into prompt context (does not import the package) |
