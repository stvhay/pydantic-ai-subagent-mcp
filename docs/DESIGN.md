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

Internally, `SessionStore.tail(session_id, offset)` returns a
`(text, next_offset)` tuple. The `tail_session_log` MCP tool wraps that
helper and returns a JSON envelope with `session_id`, `text`, and
`next_offset` so callers can correlate polling results. Clients poll by
feeding the returned `next_offset` back on the next call. Partial UTF-8
sequences at read boundaries are decoded with `errors="replace"`.

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
