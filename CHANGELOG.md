# Changelog

## v0.1.1 (2026-04-07)



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
