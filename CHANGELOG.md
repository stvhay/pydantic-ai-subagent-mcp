# Changelog

## Unreleased

<!-- bump: minor -->

### Added

- Asynchronous / background skill runs via `run_in_background=true` on `run_skill_by_name` (and the per-skill `skill_*` tools). A background launch enqueues a work item on the session's mailbox and returns immediately with `status` of `running` or `queued`; foreground (Ask mode) callers still block on their own turn (#19).
- Per-session actor model: each session owns a mailbox (`asyncio.Queue`) drained FIFO by a single long-lived worker task. Concurrent resumes to the same session enqueue in submission order rather than rejecting, and distinct sessions run on independent workers (#19).
- `stop_session(session_id)` MCP tool that drains queued mailbox items, cancels the in-flight turn, and tears down the session's worker and mailbox. Returns `{status, in_flight_cancelled, queued_dropped}`. Idempotent: safe to call on an already-idle, not-found, or just-stopped session. A cancelled turn writes a `cancelled` trailer to the session log and a `cancelled` notification to the inbox on the way out (#19).
- Backpressure: `max_concurrent_runs` (default `4`) is a server-wide `asyncio.Semaphore` owned by the `SessionStore` and acquired by the session worker around the actual turn execution; `mailbox_max_depth` (default `16`) is a per-session cap on queued items and is the only admission-time rejection (`status="mailbox_full"`, foreground and background alike). Both knobs are settable via `.subagent-mcp.json` or the env vars `SUBAGENT_MCP_MAX_CONCURRENT_RUNS` and `SUBAGENT_MCP_MAILBOX_MAX_DEPTH` (#19).
- Durable completion outbox at `.subagent-inbox/` (configurable via `inbox_dir`). Every turn exit path (`ok` / `error` / `cancelled`) writes one atomically-renamed `{uuid7}.json` record with `session_id`, `skill`, `model`, `status`, `timestamp`, and a short `summary`. Filename lexicographic order matches arrival order. A new `read_inbox(since, limit)` MCP tool exposes the outbox for pull-mode consumers (#19).
- Claude Code `UserPromptSubmit` hook bridge (`scripts/notification_hook.py` + `scripts/notification-hook.sh`). On every prompt submit the hook drains new inbox records, emits `<subagent-mcp-notification>` blocks into the next-turn context, and advances a cursor at `.subagent-inbox/.cursor`. The hook is standalone (does not import the package), at-least-once with an idempotent consumer, and fail-soft (always exits 0) so a misconfigured hook cannot break a prompt submit (#19).
- Session lifecycle fields: `status` (`idle` / `running` / `failed`) and `last_active` are persisted on every save and returned from `list_sessions` alongside a new `mailbox_depth` field (#19).
- **ACTION:** To receive completion notifications automatically, wire `scripts/notification-hook.sh` as a `UserPromptSubmit` hook in `.claude/settings.json`. See `README.md` for the full hook configuration (#19).

### Changed

- Document mid-stream staleness of `get_session_transcript` in `docs/DESIGN.md`; cross-link `get_session_transcript` and `tail_session_log` MCP tool descriptions so LLM callers pick the right one (#9).

### Added

- Per-turn completion trailer (`--- end ok ---` / `--- end error ---` / `--- end cancelled ---`) on streaming session logs so tail clients can detect completion, errors, and client-disconnect cancellation (#8).
- Streaming skill execution via `agent.run_stream()`, gated by a new `streaming: bool` config field (default `true`) with `SUBAGENT_MCP_STREAMING` env override
- Per-session append-only `{session_dir}/{session_id}.log` files capturing text deltas with a `--- prompt ---` / `--- response ---` transcript format
- New `tail_session_log(session_id, offset)` MCP tool for offset-based polling of live stream output
- Real-Ollama integration test (`tests/test_streaming_integration.py`) that exercises the full streaming path end-to-end; skipped when Ollama is unreachable
- Initial MCP server with skill discovery and Ollama subagent execution
- UUID-keyed session management with transcript persistence
- Built-in tools: file I/O, code search (ripgrep), shell execution
- Srclight integration for enhanced code indexing
- Claude Code `.mcp.json` for automatic server loading
- Nix flake + envrc.d development environment
- Configuration via `.subagent-mcp.json` and environment variables
- Tests for tools, server, config, session, and skills (28 total)
- `docs/ARCHITECTURE.md` with system context diagram and ADRs
- `docs/DESIGN.md` with skill discovery, session, and security patterns

### Fixed

- Build backend (`hatchling.backends` → `hatchling.build`)
- Ruff lint errors (unused imports, `timezone.utc` → `datetime.UTC`)
- Mypy strict errors (untyped dict, tool registration type mismatch)
- Env-dependent test failure in `test_load_from_file`
- `search_files` grep fallback when ripgrep is not installed
- CI workflow installs dev dependencies (`uv sync --extra dev`)
- `docs/DESIGN.md` streaming section: corrected `{uuid}.log` → `{session_id}.log` to match the code and README, and distinguished `SessionStore.tail()`'s `(text, next_offset)` tuple return from the `tail_session_log` MCP tool's `{session_id, text, next_offset}` JSON envelope
- `_run_skill_streaming` return type tightened from `tuple[str, list[Any]]` to `tuple[str, list[ModelMessage]]`, and a redundant `log_path.parent.mkdir()` was removed (the session directory is already created by `SessionStore.__init__`)
