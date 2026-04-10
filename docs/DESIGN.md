# Design

## Skill Discovery

Skills are markdown files discovered from:
1. `.claude/commands/` (project-level)
2. `~/.claude/commands/` (user-level)
3. Plugin `skills/` directories (from `.claude/settings.json`)

Each `.md` file becomes an MCP tool. The skill name is derived from its relative
path (e.g., `ops/restart.md` -> `ops:restart`). The first non-heading paragraph
serves as the tool description.

## Session Management

Sessions are UUID-keyed JSON files stored in the configured `session_dir`.
Each session stores messages in Ollama's /api/chat wire shape --
`list[dict[str, Any]]` with roles `system`, `user`, `assistant`, and `tool`.
Persistence is plain `json.dumps`/`json.loads` with no translation layer.

Resuming a session passes the stored message list into `run_agent`, which
appends the new user message and continues the multi-turn loop. Tool
call/result structure is preserved natively because the wire shape already
carries `tool_calls` on assistant turns and `tool_name` on tool turns.

## Streaming

Skill execution uses `run_agent` with an `on_content_delta` callback when
`config.streaming` is true (the default). Internally, `OllamaClient.chat_stream`
yields NDJSON `StreamChunk`s from /api/chat; `chat_turn` accumulates them into a
`TurnResult` and pushes each text delta through the callback. The callback in
`_run_skill_streaming` (server.py) appends each chunk to
`{session_dir}/{session_id}.log` and flushes, while the final complete response
is returned from the MCP tool as before -- the MCP protocol requires tool results
to be complete, so streaming is a side-channel, not a change to the tool return
contract.

Each turn of a multi-turn session appends a new `--- prompt ---` /
`--- response ---` block to the log, giving a plain-text transcript alongside
the structured JSON session file.

Each turn is terminated by a trailer line so tail clients can detect
completion. On success the log gains `--- end ok {iso_ts} ---`. On failure
(any `Exception` escaping `agent.run_stream()` or `stream_text()`) the log
gains `--- end error {iso_ts}: {ExceptionType}: {message} ---` and the
original exception still propagates to the outer `_run_skill` handler so the
MCP tool caller receives the usual error-dict response. On cancellation
(`asyncio.CancelledError` from a client disconnect, or any other
`BaseException` like `KeyboardInterrupt`/`SystemExit`) the log gains
`--- end cancelled {iso_ts}: {ExceptionType}: {message} ---` and the
exception still propagates out of `_run_skill` — the outer handler only
catches `Exception`, so cancellation reaches the MCP runtime intact. Tail
clients should detect the end of a turn by matching the regex
`^--- end (ok|error|cancelled) ` at the start of a line. Trailer writes are
best-effort: if writing the trailer itself fails the failure is logged but
the caller's exception is never masked. Newlines in exception messages are
replaced with spaces so trailers are always single lines.

Internally, `SessionStore.tail(session_id, offset)` returns a
`(text, next_offset)` tuple. The `tail_session_log` MCP tool wraps that
helper and returns a JSON envelope with `session_id`, `text`, and
`next_offset` so callers can correlate polling results. Clients poll by
feeding the returned `next_offset` back on the next call. Partial UTF-8
sequences at read boundaries are decoded with `errors="replace"`.

During an active streaming run, `get_session_transcript` reflects the
session as last persisted to disk — the in-flight turn is not yet visible
because `session.messages` is only written and `store.save(session)` is
only called after the streaming turn fully completes and control returns
from `_run_skill_streaming`. Use `tail_session_log(session_id, offset)`
to observe the in-progress response as deltas are flushed; once the stream
completes (trailer line written, session saved) `get_session_transcript`
returns the full structured history.

Setting `streaming: false` (config or `SUBAGENT_MCP_STREAMING=false`) falls
back to `agent.run()` and no log file is written.

## Concurrency model

Each session is an actor: it owns a mailbox (`asyncio.Queue`) and is drained by a single long-lived worker task. The mailbox FIFO + single-consumer worker *is* the linearizability primitive: every turn on a session flows through one queue drained by one task, so `session.messages` is only ever read or written by the worker and no additional lock is necessary. Distinct sessions never block each other -- they run on independent workers and only contend on the server-wide concurrency semaphore.

`_run_skill` is the admission point. It resolves or creates the session, runs two rejection checks in order (skill mismatch, mailbox full), then enqueues a `_WorkItem` and either returns immediately (background / Tell mode) or awaits the item's per-call `asyncio.Future` (foreground / Ask mode). There is no admission-time "server too busy" check: the server-wide semaphore is acquired inside the worker, at execution time, so mailbox queueing is unaffected and any admission-time check would be inherently racy. The worker is lazily spun up on the first push and lives until `SessionStore.shutdown()` (or `stop_session`) cancels it; between items it blocks on `mailbox.get()`, so an idle session costs one suspended task.

Resuming a busy session **enqueues** rather than rejects: the second push lands behind the in-flight item and runs once the worker frees up. Resume order is preserved by the FIFO. The first push to an idle session reports `status="running"`; subsequent pushes that land behind in-flight or queued work report `status="queued"` with a `mailbox_depth` snapshot.

The foreground caller blocks on its **own** future, not on the whole queue draining, so a foreground call enqueued behind one in-flight item completes as soon as its turn finishes. Cancellation of a foreground call cancels its future but does not by itself stop the worker; use `stop_session` for that.

## Backpressure

Two independent caps protect the server from a runaway producer:

- **`max_concurrent_runs`** (default `4`) -- a server-wide `asyncio.Semaphore` owned by the `SessionStore` and acquired by the session worker around the actual `_execute_skill_turn` call. The slot is held only for the duration of in-flight execution, not for queue admission, so mailbox queueing is unaffected. Both foreground and background launches always enqueue onto the session mailbox when the mailbox has room; the wait on the server-wide gate happens inside the worker at the semaphore acquire. When the gate is fully held, callers simply wait longer -- there is no admission-time rejection. Admission-time rejection based on a `locked()` snapshot would be inherently racy against the worker's execution-time acquire, and the mailbox + semaphore pair are already self-regulating without it.

- **`mailbox_max_depth`** (default `16`) -- a per-session ceiling on queued items waiting behind the in-flight turn. A push that would exceed the cap returns `status="mailbox_full"` regardless of foreground/background mode, so a foreground caller cannot bypass the cap by simply omitting `run_in_background`. The check counts queued items only (`mailbox_depth` excludes the in-flight one), so the effective concurrent backlog per session is `mailbox_max_depth + 1`.

Both knobs load from `.subagent-mcp.json` with env overrides `SUBAGENT_MCP_MAX_CONCURRENT_RUNS` and `SUBAGENT_MCP_MAILBOX_MAX_DEPTH`. Non-positive or unparseable values silently fall through to the class default; a misconfigured backpressure knob must never crash the server at boot.

## Completion notifications (outbox + hook bridge)

Every exit path of `_execute_skill_turn` (`ok` / `error` / `cancelled`) writes one notification record to the inbox at `{config.inbox_dir}/{notification_id}.json`. Notification IDs are uuid7 values, so lexicographic filename order matches arrival order. Each record carries `notification_id`, `session_id`, `skill`, `model`, `status`, `timestamp`, and a short `summary` (truncated to ~500 chars from the response or exception message). Writes are atomic: payload goes to a sibling tempfile in the same directory, fsync, then `os.replace` over the destination -- no half-written file is ever observable through a directory listing. Inbox writes are best-effort: a failure is logged and swallowed so the caller's exit path (the run result, the cancelled trailer, the propagating exception) is never masked.

There are two ways to consume the outbox:

1. **Pull mode** -- the `read_inbox` MCP tool returns a JSON `{notifications, head}` envelope. Pass the previous `head` back as `since` on the next call to advance forward. Empty `since` returns the most recent `limit` records (tail view). The cursor lives entirely on the caller side; the server is stateless with respect to delivery.

2. **Push mode via Claude Code hook** -- `scripts/notification_hook.py` is a standalone script wired as a `UserPromptSubmit` hook. On every prompt submit it reads `.subagent-inbox/.cursor`, lists `.subagent-inbox/*.json` records strictly newer than the cursor, emits one `<subagent-mcp-notification>` block per record to stdout (which Claude Code injects into the next-turn prompt), and atomically advances the cursor to the newest emitted ID. The hook does **not** import the `pydantic_ai_subagent_mcp` package -- the on-disk inbox format is the contract -- so the bridge keeps working across server upgrades or even with the server uninstalled.

Properties of the hook:

- **At-least-once delivery, idempotent consumer** -- the cursor advances only after successful emission, so a crash mid-write redelivers the same notification on the next prompt. A corrupted cursor (truncated, hand-edited, wrong shape) self-heals to "no cursor" rather than silently masking new notifications forever.
- **Bounded latency and noise** -- each invocation emits at most `READ_LIMIT` (10) notifications. A backlog drains across successive prompts.
- **Fail-soft** -- always exits 0 on a recognized condition, so a misconfigured hook can never break the user's prompt submit (a hook that crashes would be strictly worse than a dropped notification, which redelivers anyway).

## Stopping a session

The `stop_session` MCP tool tears down a running or queued session:

1. **Drain queued items first** -- pull every pending `_WorkItem` from the mailbox via `get_nowait()`, call `task_done()` for each so the queue's `join()` counter stays consistent, and pass each item to an `on_drop` callback. The server's callback cancels any per-item `future`, so foreground callers waiting on a queued item raise `CancelledError` instead of hanging forever.
2. **Cancel the worker task** -- the worker may be blocked on `mailbox.get()` (in which case cancellation just unwinds it cleanly) or inside `_execute_skill_turn` (in which case the cancellation propagates through the streaming code, which writes a `cancelled` trailer to the session log; the worker's own `BaseException` handler cancels the in-flight item's future and `_execute_skill_turn` writes a `cancelled` notification to the inbox on the way out).
3. **Clear the registries** -- pop the worker and mailbox from `SessionStore._workers` and `_mailboxes` so the next push to this session spins up a fresh worker on a fresh mailbox.

The tool returns `{session_id, status, in_flight_cancelled, queued_dropped}`. Status is `stopped` if work was actually cancelled, `already_idle` if the session existed but had no active worker, or `not_found` if no on-disk session record existed. Stop is **idempotent**: calling on a session that is already idle, missing, or just stopped is safe and returns a sensible status without raising. After a stop, the session record stays on disk (with `status="failed"` from the cancellation) and can be resumed by a fresh push -- there is no separate "session was stopped" lifecycle state, because cancellation IS the stopped state and observers detect it via the existing `cancelled` trailer / inbox notification channels.

## Recursive Sub-Agents

Agents can spawn sub-agents by calling this same MCP server through the
external MCP server injection path: the server declares itself in
`.subagent-mcp.servers.json`, `MCPToolLoader` spawns it as a long-lived
child, and its tools appear as `Tool` shims alongside the built-in tools.
Recursion depth is bounded by configuration (`max_recursion_depth`) to
prevent infinite loops.

## Security Considerations

- `shell_exec` runs arbitrary commands -- scoped to the project working directory
- Recursive sub-agents are depth-limited
- No network access beyond Ollama and configured MCP servers
