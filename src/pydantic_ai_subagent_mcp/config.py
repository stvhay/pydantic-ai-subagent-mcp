"""Configuration for the subagent MCP server."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServerConfig:
    """Configuration loaded from environment and optional config file."""

    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "gemma4:12b"
    session_dir: str = ".subagent-sessions"
    max_iterations: int = 50
    tool_timeout: float = 120.0
    srclight_enabled: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path | None = None) -> ServerConfig:
        """Load config from file and environment overrides."""
        data: dict = {}

        # Load from config file if provided or default location exists
        if config_path is None:
            config_path = Path(".subagent-mcp.json")
        if config_path.exists():
            data = json.loads(config_path.read_text())

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
            max_iterations=int(data.get("max_iterations", cls.max_iterations)),
            tool_timeout=float(data.get("tool_timeout", cls.tool_timeout)),
            srclight_enabled=data.get("srclight_enabled", cls.srclight_enabled),
            extra_env=data.get("extra_env", {}),
        )
