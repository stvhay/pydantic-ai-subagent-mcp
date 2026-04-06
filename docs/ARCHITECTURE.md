# Architecture

## Overview

pydantic-ai-subagent-mcp is an MCP server that bridges Claude Code skills to local
Ollama models. It sits between Claude Code (MCP client) and Ollama (LLM backend),
using pydantic-ai as the agent framework.

## System Context

```
Claude Code (MCP client)
  |
  +-- stdio --> subagent-mcp (FastMCP server)
  |               |
  |               +-- pydantic-ai Agent
  |               |    +-- OpenAIChatModel + OllamaProvider --> Ollama
  |               |    +-- Built-in tools (file I/O, search, shell)
  |               |    +-- MCPServerStdio toolsets --> srclight, self (recursive)
  |               |
  |               +-- SessionStore (JSON on disk, UUID-keyed)
  |
  +-- srclight (optional, code indexing MCP server)
```

## Key Architectural Decisions

### ADR-1: pydantic-ai as agent framework

> In the context of needing an agent framework for Ollama, facing choices between
> raw API calls, LangChain, and pydantic-ai, we chose pydantic-ai to get native
> Ollama support, typed tool definitions, and built-in MCP client capabilities,
> accepting coupling to the pydantic-ai API surface.

### ADR-2: Native message history over text concatenation

> In the context of multi-turn sessions, facing a choice between manually
> concatenating prior messages as text or using pydantic-ai's `message_history`
> parameter, we chose native message history to preserve tool call/result structure
> and let the framework manage context, accepting that session serialization must
> use `ModelMessagesTypeAdapter`.

### ADR-3: MCP client for external tool access

> In the context of giving subagents access to srclight and recursive sub-agents,
> facing a choice between custom HTTP integrations or pydantic-ai's MCP client,
> we chose `MCPServerStdio` toolsets to reuse the MCP protocol and enable the agent
> to call any MCP server, accepting subprocess management overhead.

### ADR-4: Log-based streaming over MCP progress notifications

> In the context of wanting incremental output visibility during long-running
> skill runs, facing a choice between MCP `notifications/progress` (requires
> client-side `progressToken` handling) and writing deltas to per-session log
> files, we chose the log-file approach to keep the MCP tool contract unchanged
> (callers still receive a complete result) and to make transcripts inspectable
> out-of-band, accepting that clients must poll a `tail_session_log` tool to
> read live output.

## Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `server.py` | FastMCP server, tool registration, agent orchestration |
| `config.py` | Configuration from file + environment |
| `skills.py` | Skill discovery from filesystem |
| `session.py` | Session persistence with native pydantic-ai messages; per-session `.log` files for streaming output |
| `tools.py` | Built-in tools for file I/O, search, shell |
