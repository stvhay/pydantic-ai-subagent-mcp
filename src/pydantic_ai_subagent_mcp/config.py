"""Configuration for the subagent MCP server."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ServerConfig:
    """Configuration loaded from environment and optional config file."""

    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "gemma4:12b"
    session_dir: str = ".subagent-sessions"
    inbox_dir: str = ".subagent-inbox"
    max_iterations: int = 50
    tool_timeout: float = 120.0
    srclight_enabled: bool = True
    streaming: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path | None = None) -> ServerConfig:
        """Load config from file and environment overrides."""
        data: dict[str, Any] = {}

        # Load from config file if provided or default location exists
        if config_path is None:
            config_path = Path(".subagent-mcp.json")
        if config_path.exists():
            data = json.loads(config_path.read_text())

        # Parse SUBAGENT_MCP_STREAMING env override (truthy: 1/true/yes)
        streaming_default = bool(data.get("streaming", cls.streaming))
        streaming_env = os.environ.get("SUBAGENT_MCP_STREAMING")
        if streaming_env is not None:
            streaming_default = streaming_env.strip().lower() in ("1", "true", "yes")

        # Environment overrides take precedence
        return cls(
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL", data.get("ollama_base_url", cls.ollama_base_url)
            ),
            default_model=os.environ.get(
                "SUBAGENT_MCP_DEFAULT_MODEL",
                data.get("default_model", cls.default_model),
            ),
            session_dir=data.get("session_dir", cls.session_dir),
            inbox_dir=data.get("inbox_dir", cls.inbox_dir),
            max_iterations=int(data.get("max_iterations", cls.max_iterations)),
            tool_timeout=float(data.get("tool_timeout", cls.tool_timeout)),
            srclight_enabled=data.get("srclight_enabled", cls.srclight_enabled),
            streaming=streaming_default,
            extra_env=data.get("extra_env", {}),
        )
