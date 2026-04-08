# pydantic-ai-subagent-mcp

MCP server that proxies Claude Code skills to local Ollama models via pydantic-ai agents.

## Build & Run

```bash
uv sync                    # Install dependencies
uv run subagent-mcp        # Run the MCP server (stdio mode)
uv run pytest              # Run tests
uv run ruff check src/ tests/  # Lint
uv run mypy src/           # Type check
```

## Project structure

- `src/pydantic_ai_subagent_mcp/server.py` -- Main MCP server entry point (FastMCP, tool registration)
- `src/pydantic_ai_subagent_mcp/config.py` -- Configuration loading from file + env
- `src/pydantic_ai_subagent_mcp/skills.py` -- Skill discovery from Claude Code command dirs
- `src/pydantic_ai_subagent_mcp/session.py` -- UUID-keyed session store with transcripts
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

- `pydantic-ai` -- Agent framework with Ollama support
- `mcp` -- Model Context Protocol SDK (FastMCP for stdio server)
- `srclight` -- Code indexing MCP server (optional, for enhanced code search)
- `httpx` -- HTTP client for Ollama API

## CI vs local environment

The Nix flake provides `ripgrep` (`rg`) locally, but CI (ubuntu-latest) does not have it.
Tools like `search_files` must gracefully fall back to `grep` when `rg` is missing.
When adding tools that shell out to CLI programs, ensure fallback paths are tested
without those programs available (CI will catch this if local tests don't).

## Environment variables

- `OLLAMA_BASE_URL` -- Ollama endpoint (default: http://localhost:11434)
- `SUBAGENT_MCP_DEFAULT_MODEL` -- Default model (default: gemma4:12b)
