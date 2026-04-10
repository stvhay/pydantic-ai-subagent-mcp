"""Tests for configuration loading."""

import json
from pathlib import Path

import pytest

from pydantic_ai_subagent_mcp.config import ServerConfig


def test_default_config():
    config = ServerConfig()
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.default_model == "gemma4:26b"
    assert config.mcp_servers_config == ".subagent-mcp.servers.json"


def test_mcp_servers_config_loaded_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The mcp_servers_config path comes from .subagent-mcp.json when set."""
    monkeypatch.delenv("SUBAGENT_MCP_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"mcp_servers_config": "/etc/custom-mcp-servers.json"})
    )
    config = ServerConfig.load(config_file)
    assert config.mcp_servers_config == "/etc/custom-mcp-servers.json"


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


# -- Phase 6: backpressure knobs --


def test_backpressure_defaults() -> None:
    """The default backpressure knobs are positive integers."""
    config = ServerConfig()
    assert config.max_concurrent_runs == 4
    assert config.mailbox_max_depth == 16


def test_backpressure_loaded_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SUBAGENT_MCP_MAX_CONCURRENT_RUNS", raising=False)
    monkeypatch.delenv("SUBAGENT_MCP_MAILBOX_MAX_DEPTH", raising=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"max_concurrent_runs": 8, "mailbox_max_depth": 32})
    )
    config = ServerConfig.load(config_file)
    assert config.max_concurrent_runs == 8
    assert config.mailbox_max_depth == 32


def test_backpressure_env_overrides_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"max_concurrent_runs": 8, "mailbox_max_depth": 32})
    )
    monkeypatch.setenv("SUBAGENT_MCP_MAX_CONCURRENT_RUNS", "2")
    monkeypatch.setenv("SUBAGENT_MCP_MAILBOX_MAX_DEPTH", "5")
    config = ServerConfig.load(config_file)
    assert config.max_concurrent_runs == 2
    assert config.mailbox_max_depth == 5


@pytest.mark.parametrize("bad_value", ["-1", "0", "not-a-number", ""])
def test_backpressure_env_invalid_falls_back_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    """Garbage env values must not crash the server -- they fall through."""
    monkeypatch.setenv("SUBAGENT_MCP_MAX_CONCURRENT_RUNS", bad_value)
    monkeypatch.setenv("SUBAGENT_MCP_MAILBOX_MAX_DEPTH", bad_value)
    config = ServerConfig.load(tmp_path / "missing.json")
    assert config.max_concurrent_runs == 4
    assert config.mailbox_max_depth == 16


def test_backpressure_file_invalid_falls_back_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-positive config-file values fall back to the class default."""
    monkeypatch.delenv("SUBAGENT_MCP_MAX_CONCURRENT_RUNS", raising=False)
    monkeypatch.delenv("SUBAGENT_MCP_MAILBOX_MAX_DEPTH", raising=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"max_concurrent_runs": 0, "mailbox_max_depth": -3})
    )
    config = ServerConfig.load(config_file)
    assert config.max_concurrent_runs == 4
    assert config.mailbox_max_depth == 16


# -- shutdown timeout knob --


def test_shutdown_timeout_default() -> None:
    """The default shutdown timeout is 5.0 seconds."""
    config = ServerConfig()
    assert config.shutdown_timeout_seconds == 5.0


def test_shutdown_timeout_loaded_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SUBAGENT_MCP_SHUTDOWN_TIMEOUT", raising=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"shutdown_timeout_seconds": 10.0}))
    config = ServerConfig.load(config_file)
    assert config.shutdown_timeout_seconds == 10.0


def test_shutdown_timeout_env_overrides_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"shutdown_timeout_seconds": 10.0}))
    monkeypatch.setenv("SUBAGENT_MCP_SHUTDOWN_TIMEOUT", "3.0")
    config = ServerConfig.load(config_file)
    assert config.shutdown_timeout_seconds == 3.0


@pytest.mark.parametrize("bad_value", ["-1", "0", "not-a-number", ""])
def test_shutdown_timeout_env_invalid_falls_back_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    """Garbage env values must not crash the server -- they fall through."""
    monkeypatch.setenv("SUBAGENT_MCP_SHUTDOWN_TIMEOUT", bad_value)
    config = ServerConfig.load(tmp_path / "missing.json")
    assert config.shutdown_timeout_seconds == 5.0


def test_shutdown_timeout_file_invalid_falls_back_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-positive config-file values fall back to the class default."""
    monkeypatch.delenv("SUBAGENT_MCP_SHUTDOWN_TIMEOUT", raising=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"shutdown_timeout_seconds": 0}))
    config = ServerConfig.load(config_file)
    assert config.shutdown_timeout_seconds == 5.0
