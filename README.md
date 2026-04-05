# pydantic-ai-subagent-mcp

MCP server that proxies Claude Code skills to local Ollama models (gemma4 family) via pydantic-ai agents.

## What it does

When loaded as an MCP server in Claude Code, it:

1. **Discovers skills** from Claude Code's command directories and installed plugins
2. **Registers each skill as an MCP tool** with optional model and session parameters
3. **Delegates execution** to a pydantic-ai agent backed by Ollama, with a rich set of built-in tools (file I/O, code search, shell execution, srclight)
4. **Manages sessions** with UUID-keyed transcripts so conversations can be resumed
5. **Supports recursive sub-agents** -- the spawned agent can itself use this MCP to create sub-sub-agents

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
  "srclight_enabled": true
}
```

Environment variables `OLLAMA_BASE_URL` and `SUBAGENT_MCP_DEFAULT_MODEL` override the config file.

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
                        |
                        |-- discovers skills from .claude/commands/, plugins
                        |-- registers each as an MCP tool
                        |-- on tool call:
                        |     |-- creates pydantic-ai Agent with Ollama model
                        |     |-- provides built-in tools (files, search, shell, srclight)
                        |     |-- runs agent with skill prompt + user input
                        |     |-- persists session transcript
                        |     \-- returns result
                        |
                        \-- session management tools (list, get transcript, resume)
```

## License

GPL-3.0-or-later
