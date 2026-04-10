# Changelog

## Unreleased

<!-- bump: minor -->

### Added

- External MCP server tool injection: subagents now expose tools from any MCP server declared in `.subagent-mcp.servers.json` (configurable via the new `mcp_servers_config` field in `.subagent-mcp.json`). The file uses the standard `{"mcpServers": {<name>: {command, args, env}}}` shape consumed by `pydantic_ai.mcp.load_mcp_servers`, with `${VAR}` / `${VAR:-default}` env-var expansion. Each subagent run wraps the agent in `async with agent:` so server subprocesses are started fresh per turn and torn down on exit. A missing or malformed config is logged and treated as empty so the server still boots. Ships with `.subagent-mcp.servers.json.example` showing srclight pre-wired (#5).
- Asynchronous / background skill runs via `run_in_background=true` on `run_skill_by_name` (and the per-skill `skill_*` tools). A background launch enqueues a work item on the session's mailbox and returns immediately with `status` of `running` or `queued`; foreground (Ask mode) callers still block on their own turn (#19).
- Per-session actor model: each session owns a mailbox (`asyncio.Queue`) drained FIFO by a single long-lived worker task. Concurrent resumes to the same session enqueue in submission order rather than rejecting, and distinct sessions run on independent workers (#19).
- `stop_session(session_id)` MCP tool that drains queued mailbox items, cancels the in-flight turn, and tears down the session's worker and mailbox. Returns `{status, in_flight_cancelled, queued_dropped}`. Idempotent: safe to call on an already-idle, not-found, or just-stopped session. A cancelled turn writes a `cancelled` trailer to the session log and a `cancelled` notification to the inbox on the way out (#19).
- Backpressure: `max_concurrent_runs` (default `4`) is a server-wide `asyncio.Semaphore` owned by the `SessionStore` and acquired by the session worker around the actual turn execution; `mailbox_max_depth` (default `16`) is a per-session cap on queued items and is the only admission-time rejection (`status="mailbox_full"`, foreground and background alike). Both knobs are settable via `.subagent-mcp.json` or the env vars `SUBAGENT_MCP_MAX_CONCURRENT_RUNS` and `SUBAGENT_MCP_MAILBOX_MAX_DEPTH` (#19).
- Durable completion outbox at `.subagent-inbox/` (configurable via `inbox_dir`). Every turn exit path (`ok` / `error` / `cancelled`) writes one atomically-renamed `{uuid7}.json` record with `session_id`, `skill`, `model`, `status`, `timestamp`, and a short `summary`. Filename lexicographic order matches arrival order. A new `read_inbox(since, limit)` MCP tool exposes the outbox for pull-mode consumers (#19).
- Claude Code `UserPromptSubmit` hook bridge (`scripts/notification_hook.py` + `scripts/notification-hook.sh`). On every prompt submit the hook drains new inbox records, emits `<subagent-mcp-notification>` blocks into the next-turn context, and advances a cursor at `.subagent-inbox/.cursor`. The hook is standalone (does not import the package), at-least-once with an idempotent consumer, and fail-soft (always exits 0) so a misconfigured hook cannot break a prompt submit (#19).
- Session lifecycle fields: `status` (`idle` / `running` / `failed`) and `last_active` are persisted on every save and returned from `list_sessions` alongside a new `mailbox_depth` field (#19).
- **ACTION:** To receive completion notifications automatically, wire `scripts/notification-hook.sh` as a `UserPromptSubmit` hook in `.claude/settings.json`. See `README.md` for the full hook configuration (#19).
- Regression tests and probe script verifying parallel `tool_calls` handling across the gemma4 model family. All four tags (e2b, e4b, 26b, 31b) emit parallel tool_calls and handle the canonical multi-tool-result history shape correctly. ADR-7 documents the finding. `@pytest.mark.live` marker infrastructure added for opt-in live endpoint tests (#26).
- Per-turn completion trailer (`--- end ok ---` / `--- end error ---` / `--- end cancelled ---`) on streaming session logs so tail clients can detect completion, errors, and client-disconnect cancellation (#8).
- Streaming unit tests closing the D1 (resume-from-disk multi-turn) and D3 (concurrent tail mid-stream) coverage gaps from the #4 final integration review. D2 (mid-stream error) was already covered by the trailer tests added in #8 (#11).
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

### Changed

- Skill discovery now reads modern Claude Code `SKILL.md` files instead of loose markdown slash commands. A skill is a directory containing `SKILL.md` whose first block is YAML frontmatter declaring at least `name` (and optionally `description`); unknown frontmatter keys (e.g. `model`, `effort`, `context`) are tolerated so benchmark-style skills load cleanly. The default search order is `./.claude/skills/` → `~/.claude/skills/` → every installed plugin under `~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/skills/`, with name-collision dedup so earlier dirs win. Plugin attribution is recovered from the cache path. Adds a `pyyaml` runtime dependency (and `types-PyYAML` for dev). Existing `.claude/commands/*.md` slash commands are no longer picked up — migrate them to `.claude/skills/<name>/SKILL.md` with frontmatter.
- Default model bumped from `gemma4:12b` to `gemma4:26b`. The 26B Gemma 4 variant handles tool calling and multi-turn skill execution more reliably than the smaller 12B model, which matters now that subagents can pull tools from external MCP servers (#5). Existing `.subagent-mcp.json` files that pin `default_model` are unaffected; only the fallback default changed.
- Document mid-stream staleness of `get_session_transcript` in `docs/DESIGN.md`; cross-link `get_session_transcript` and `tail_session_log` MCP tool descriptions so LLM callers pick the right one (#9).
- `run_skill_by_name` now resolves the skill via `next(...)` instead of building an intermediate list, a small readability cleanup with no behavior change.

### Removed

- Placeholder `web_search` built-in tool. It returned a hard-coded "not yet configured" string and only existed as a stub for future search-API integration; now that subagents can pull tools from any external MCP server (see Added → MCP server tool injection), wiring up a real search server through `.subagent-mcp.servers.json` is the supported path. Existing call sites (none in production code) need no migration -- just remove the import.
- Dead config knobs `max_iterations`, `tool_timeout`, and `srclight_enabled` from `ServerConfig`. They were declared, defaulted, and parsed from `.subagent-mcp.json` but never read by any production code path. The `srclight` MCP server is still listed as a dependency (and remains usable when wired through `.mcp.json`); only the unused config flag is gone. Existing `.subagent-mcp.json` files that still set these keys keep working — unknown keys are silently ignored by the loader — but the keys no longer do anything and can be deleted.

### Fixed

- Build backend (`hatchling.backends` → `hatchling.build`)
- Ruff lint errors (unused imports, `timezone.utc` → `datetime.UTC`)
- Mypy strict errors (untyped dict, tool registration type mismatch)
- Env-dependent test failure in `test_load_from_file`
- `search_files` grep fallback when ripgrep is not installed
- CI workflow installs dev dependencies (`uv sync --extra dev`)
- `docs/DESIGN.md` streaming section: corrected `{uuid}.log` → `{session_id}.log` to match the code and README, and distinguished `SessionStore.tail()`'s `(text, next_offset)` tuple return from the `tail_session_log` MCP tool's `{session_id, text, next_offset}` JSON envelope
- `_run_skill_streaming` return type tightened from `tuple[str, list[Any]]` to `tuple[str, list[ModelMessage]]`, and a redundant `log_path.parent.mkdir()` was removed (the session directory is already created by `SessionStore.__init__`)
