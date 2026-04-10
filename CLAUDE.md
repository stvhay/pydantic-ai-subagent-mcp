# pydantic-ai-subagent-mcp

MCP server that proxies Claude Code skills to local Ollama models via a native /api/chat agent loop.

## Build & Run

```bash
uv sync                    # Install dependencies
uv run subagent-mcp        # Run the MCP server (stdio mode)
uv run pytest              # Run tests
uv run ruff check src/ tests/  # Lint
uv run mypy src/           # Type check
```

## Project structure

- `src/pydantic_ai_subagent_mcp/server.py` -- Main MCP server entry point (FastMCP, tool registration, orchestration)
- `src/pydantic_ai_subagent_mcp/agent.py` -- Multi-turn agent loop (`run_agent`), `Tool` dataclass, `AgentResult`
- `src/pydantic_ai_subagent_mcp/ollama.py` -- Native Ollama /api/chat client, NDJSON streaming, `ChatClient` Protocol
- `src/pydantic_ai_subagent_mcp/mcp_loader.py` -- External MCP server lifecycle (long-lived child sessions)
- `src/pydantic_ai_subagent_mcp/config.py` -- Configuration loading from file + env
- `src/pydantic_ai_subagent_mcp/skills.py` -- Skill discovery from Claude Code command dirs
- `src/pydantic_ai_subagent_mcp/session.py` -- UUID-keyed session store with Ollama-native messages
- `src/pydantic_ai_subagent_mcp/inbox.py` -- Durable on-disk outbox for completion notifications
- `src/pydantic_ai_subagent_mcp/tools.py` -- Built-in tools for subagent runs (files, search, shell)
- `.mcp.json` -- Claude Code MCP server configuration
- `.subagent-mcp.json` -- Default server configuration
- `flake.nix` -- Nix development environment (Python 3.14 + uv)

## Conventions

- Python 3.14+ (required for `uuid.uuid7()`)
- Use `uv` for package management, never pip directly
- Type annotations required; `mypy --strict` must pass
- Ruff for linting and formatting
- Tests in `tests/` using pytest with asyncio auto mode
- Source layout: `src/pydantic_ai_subagent_mcp/`

## Key dependencies

- `mcp` -- Model Context Protocol SDK (FastMCP for stdio server)
- `httpx` -- HTTP client for Ollama /api/chat
- `srclight` -- Code indexing MCP server (optional, for enhanced code search)

## CI vs local environment

The Nix flake provides `ripgrep` (`rg`) locally, but CI (ubuntu-latest) does not have it.
Tools like `search_files` must gracefully fall back to `grep` when `rg` is missing.
When adding tools that shell out to CLI programs, ensure fallback paths are tested
without those programs available (CI will catch this if local tests don't).

## Environment variables

- `OLLAMA_BASE_URL` -- Ollama endpoint (default: http://localhost:11434)
- `SUBAGENT_MCP_DEFAULT_MODEL` -- Default model (default: gemma4:12b)
