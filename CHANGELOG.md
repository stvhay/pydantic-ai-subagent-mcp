# Changelog

## Unreleased

<!-- bump: patch -->

### Added

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
