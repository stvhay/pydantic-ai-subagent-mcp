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
Each session stores the pydantic-ai native message format (serialized via
`ModelMessagesTypeAdapter`) to preserve tool call/result structure across turns.

Resuming a session passes the stored messages as `message_history` to
`Agent.run()`, allowing the model to see the full conversation including
tool interactions.

## Streaming

Skill execution uses `agent.run_stream()` when `config.streaming` is true (the
default). Text deltas from the model are appended to `{session_dir}/{session_id}.log`
and flushed after each chunk, while the final complete response is returned
from the MCP tool as before — the MCP protocol requires tool results to be
complete, so streaming is a side-channel, not a change to the tool return
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
because pydantic-ai's `all_messages()` is unavailable until the
`agent.run_stream()` context exits. Use `tail_session_log(session_id, offset)`
to observe the in-progress response as deltas are flushed; once the stream
completes (trailer line written, session saved) `get_session_transcript`
returns the full structured history.

Setting `streaming: false` (config or `SUBAGENT_MCP_STREAMING=false`) falls
back to `agent.run()` and no log file is written.

## Recursive Sub-Agents

Agents can spawn sub-agents by calling this same MCP server through
pydantic-ai's `MCPServerStdio` toolset. Recursion depth is bounded by
configuration (`max_recursion_depth`) to prevent infinite loops.

## Security Considerations

- `shell_exec` runs arbitrary commands -- scoped to the project working directory
- Recursive sub-agents are depth-limited
- No network access beyond Ollama and configured MCP servers
