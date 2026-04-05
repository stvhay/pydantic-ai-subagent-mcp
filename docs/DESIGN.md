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

Skill execution uses `agent.run_stream()` to produce incremental output.
For MCP tool responses, the full output is buffered and returned (MCP tools
return complete strings). Future: investigate MCP protocol extensions for
streaming tool output.

## Recursive Sub-Agents

Agents can spawn sub-agents by calling this same MCP server through
pydantic-ai's `MCPServerStdio` toolset. Recursion depth is bounded by
configuration (`max_recursion_depth`) to prevent infinite loops.

## Security Considerations

- `shell_exec` runs arbitrary commands -- scoped to the project working directory
- Recursive sub-agents are depth-limited
- No network access beyond Ollama and configured MCP servers
