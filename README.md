# pydantic-ai-subagent-mcp

MCP server that proxies Claude Code skills to local Ollama models (gemma4 family) via pydantic-ai agents.

## What it does

When loaded as an MCP server in Claude Code, it:

1. **Discovers skills** from Claude Code's command directories and installed plugins
2. **Registers each skill as an MCP tool** with optional model and session parameters
3. **Delegates execution** to a pydantic-ai agent backed by Ollama, with a rich set of built-in tools (file I/O, code search, shell execution, srclight)
4. **Manages sessions** with UUID-keyed transcripts so conversations can be resumed
5. **Supports recursive sub-agents** -- the spawned agent can itself use this MCP to create sub-sub-agents
6. **Streams output incrementally** — with `streaming: true` (default), each skill run writes text deltas to a `<session_id>.log` file alongside the session JSON. Use the `tail_session_log` tool to poll for live output while a run is in progress.

## Quick start

```bash
# Install dependencies
uv sync

# Configure Ollama endpoint (defaults shown)
export OLLAMA_BASE_URL=http://localhost:11434
export SUBAGENT_MCP_DEFAULT_MODEL=gemma4:12b

# Run the server directly
uv run subagent-mcp
```

The `.mcp.json` file is included for automatic loading in Claude Code.

## Configuration

Create `.subagent-mcp.json` in your project root:

```json
{
  "ollama_base_url": "http://localhost:11434",
  "default_model": "gemma4:12b",
  "session_dir": ".subagent-sessions",
  "streaming": true,
  "srclight_enabled": true
}
```

Environment variables `OLLAMA_BASE_URL`, `SUBAGENT_MCP_DEFAULT_MODEL`, and `SUBAGENT_MCP_STREAMING` override the config file.

## Background runs and completion notifications

Skill invocations can be launched in the background by passing `run_in_background=true` to `run_skill_by_name`. The MCP call returns immediately with `status: "running"` and a `session_id`; the run continues on the server's event loop. Use `tail_session_log` to poll live output and `get_session_transcript` once the run completes.

When a background run finishes, a small JSON record is appended to `.subagent-inbox/{notification_id}.json`. There are two ways to consume them:

1. **Polled** — call the `read_inbox` MCP tool. It returns unread records and a cursor; pass the cursor back as `since` on the next call to advance.
2. **Pushed via Claude Code hook** — wire `scripts/notification-hook.sh` as a `UserPromptSubmit` hook. On every prompt submit, the hook drains new inbox records and emits a `<subagent-mcp-notification>` block per record, which Claude Code injects into the next-turn context. No polling required.

To enable the hook, add this to `.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/abs/path/to/pydantic-ai-subagent-mcp/scripts/notification-hook.sh"
          }
        ]
      }
    ]
  }
}
```

Properties of the hook:

- **At-least-once delivery, idempotent consumer** — the hook tracks its own cursor at `.subagent-inbox/.cursor`. A crash mid-write redelivers the same notification on the next prompt; a corrupted cursor self-heals to "no cursor" instead of silently masking new notifications.
- **Standalone** — `notification_hook.py` reads the inbox directory directly and does not import the `pydantic_ai_subagent_mcp` package, so the hook keeps working even if the server is uninstalled or downgraded.
- **Bounded latency and noise** — each invocation emits at most 10 notifications. A backlog drains across successive prompts.
- **Safe defaults** — exits 0 on any error so a misconfigured hook can never break your prompt submit.

If your inbox lives outside the project root, set `SUBAGENT_MCP_INBOX_DIR` in the hook environment.

### Stopping a session

Call the `stop_session` MCP tool with a `session_id` to cancel an in-flight turn and drop any queued items from the session's mailbox. The tool returns JSON with `status` (`stopped` / `already_idle` / `not_found`), `in_flight_cancelled` (`0` or `1`), and `queued_dropped` (count). Stop is idempotent: calling on a session that is already idle, missing, or just stopped is safe and returns a sensible status without raising. A cancelled in-flight turn writes a `cancelled` trailer to the session log and a `cancelled` notification to the inbox, so observers detect the stop on their normal completion paths. After a stop, the session record stays on disk and can be resumed by a fresh push, which will spin up a new worker on a new mailbox.

### Backpressure

Two independent caps protect the server from runaway producers:

- **`max_concurrent_runs`** (default `4`) — server-wide ceiling on the number of in-flight skill turns across all sessions. Background launches that arrive while the gate is fully held return `status: "saturated"` immediately so the caller can react. Foreground launches always enqueue and end up waiting on the gate inside the worker (the natural Ask-mode behavior).
- **`mailbox_max_depth`** (default `16`) — per-session ceiling on queued items waiting behind the in-flight turn. A push that would exceed the cap returns `status: "mailbox_full"` regardless of mode, so foreground callers cannot bypass the cap by simply omitting `run_in_background`.

Both knobs can be set in `.subagent-mcp.json` or overridden by `SUBAGENT_MCP_MAX_CONCURRENT_RUNS` and `SUBAGENT_MCP_MAILBOX_MAX_DEPTH`. Non-positive or unparseable values silently fall back to the defaults (a misconfigured backpressure knob must never crash the server at boot).

### Example: launch a background run, observe it, stop it

```text
# Launch a long-running skill in the background -- returns immediately
> run_skill_by_name(skill_name="research-deep", prompt="...", run_in_background=true)
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "log_path": ".subagent-sessions/550e8400-....log",
  "mailbox_depth": 1
}

# Tail the live output as the model streams
> tail_session_log(session_id="550e8400-...", offset=0)
{ "text": "Searching for ...", "next_offset": 1234 }

# When the hook is wired, completion shows up in the next prompt context as
# <subagent-mcp-notification session_id="550e8400-..." status="ok" ...>
# (or call read_inbox to drain notifications manually)

# If you change your mind, stop the session -- in-flight turn cancels,
# queued items drop, the cancelled notification fires on the normal
# completion path
> stop_session(session_id="550e8400-...")
{
  "session_id": "550e8400-...",
  "status": "stopped",
  "in_flight_cancelled": 1,
  "queued_dropped": 0
}

# A second stop is safe -- idempotent
> stop_session(session_id="550e8400-...")
{ "status": "already_idle", "in_flight_cancelled": 0, "queued_dropped": 0 }

# A fresh push to the same session spins up a new worker on a new mailbox
> run_skill_by_name(skill_name="research-deep", prompt="try again",
                    session_id="550e8400-...")
{ "status": "idle", "response": "...", ... }
```

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

## Architecture

```
Claude Code
  |
  |-- MCP (stdio) --> subagent-mcp server
  |                     |
  |                     |-- discovers skills from .claude/commands/, plugins
  |                     |-- registers each as an MCP tool
  |                     |-- on tool call:
  |                     |     |-- per-session mailbox + worker actor model
  |                     |     |-- server-wide concurrency semaphore
  |                     |     |-- creates pydantic-ai Agent with Ollama model
  |                     |     |-- streams response to {session_id}.log
  |                     |     |-- writes completion notification to inbox
  |                     |     \-- returns result (foreground) or "running" (background)
  |                     |
  |                     +-- session tools: list, get transcript, tail log, stop
  |                     |
  |                     \-- inbox tools: read_inbox (pull mode)
  |
  \-- UserPromptSubmit hook --> notification_hook.py (push mode, drains inbox)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for ADRs and
[docs/DESIGN.md](docs/DESIGN.md) for the full concurrency / backpressure /
stop semantics.

## License

GPL-3.0-or-later
