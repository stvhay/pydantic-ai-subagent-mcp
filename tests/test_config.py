"""Tests for configuration loading."""

import json
from pathlib import Path

import pytest

from pydantic_ai_subagent_mcp.config import ServerConfig


def test_default_config():
    config = ServerConfig()
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.default_model == "gemma4:12b"
    assert config.srclight_enabled is True


def test_load_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SUBAGENT_MCP_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "default_model": "gemma4:27b",
        "session_dir": "/tmp/sessions",
    }))
    config = ServerConfig.load(config_file)
    assert config.default_model == "gemma4:27b"
    assert config.session_dir == "/tmp/sessions"
    # Defaults preserved
    assert config.ollama_base_url == "http://localhost:11434"


def test_streaming_default_true():
    config = ServerConfig()
    assert config.streaming is True


def test_streaming_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SUBAGENT_MCP_STREAMING", raising=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"streaming": False}))
    config = ServerConfig.load(config_file)
    assert config.streaming is False


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
    ],
)
def test_streaming_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected: bool,
) -> None:
    # Config file says the opposite — env should win
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"streaming": not expected}))
    monkeypatch.setenv("SUBAGENT_MCP_STREAMING", env_value)
    config = ServerConfig.load(config_file)
    assert config.streaming is expected
