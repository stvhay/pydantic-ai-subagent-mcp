"""Tests for the standalone notification hook script.

The hook is loaded as a module via ``importlib`` because it lives in
``scripts/`` (intentionally outside the package — the design says the
hook must be importable without depending on
``pydantic_ai_subagent_mcp``). One subprocess test verifies the actual
shebang/main wiring; one verifies the bash shim execs the python script.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK_PY = REPO_ROOT / "scripts" / "notification_hook.py"
HOOK_SH = REPO_ROOT / "scripts" / "notification-hook.sh"

# A real uuid7 stem for use in tests where the cursor's UUID-shape check
# matters. The hook validates cursor format loosely (length + dash count),
# so synthetic ids must look like canonical UUIDs.
UUID_A = "01939a55-1234-7000-8000-000000000001"
UUID_B = "01939a55-1234-7000-8000-000000000002"
UUID_C = "01939a55-1234-7000-8000-000000000003"


@pytest.fixture
def hook() -> ModuleType:
    """Import notification_hook.py as a module without installing it."""
    spec = importlib.util.spec_from_file_location("notification_hook", HOOK_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_record(
    inbox_dir: Path, nid: str, **fields: Any
) -> Path:
    """Write a notification record file with sensible defaults."""
    inbox_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "notification_id": nid,
        "session_id": "sess",
        "skill": "skill",
        "model": "gemma4:12b",
        "status": "ok",
        "timestamp": "2026-04-08T00:00:00+00:00",
        "summary": "done",
    }
    record.update(fields)
    path = inbox_dir / f"{nid}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


# -- main() unit tests ---------------------------------------------------


def test_missing_inbox_dir_returns_zero_silently(
    hook: ModuleType, tmp_path: Path
) -> None:
    """A missing inbox directory must not crash and must emit nothing."""
    out = StringIO()
    rc = hook.main(inbox_dir=tmp_path / "does-not-exist", stdout=out)
    assert rc == 0
    assert out.getvalue() == ""


def test_empty_inbox_returns_zero_silently(
    hook: ModuleType, tmp_path: Path
) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    out = StringIO()
    rc = hook.main(inbox_dir=inbox, stdout=out)
    assert rc == 0
    assert out.getvalue() == ""


def test_emits_notification_block_per_record(
    hook: ModuleType, tmp_path: Path
) -> None:
    inbox = tmp_path / "inbox"
    _write_record(inbox, UUID_A, summary="hello world", skill="code-review")
    _write_record(inbox, UUID_B, summary="task done", status="ok")

    out = StringIO()
    rc = hook.main(inbox_dir=inbox, stdout=out)

    assert rc == 0
    text = out.getvalue()
    assert text.count("<subagent-mcp-notification") == 2
    assert text.count("</subagent-mcp-notification>") == 2
    assert "hello world" in text
    assert "task done" in text
    assert 'skill="code-review"' in text


def test_notification_block_includes_all_fields(
    hook: ModuleType, tmp_path: Path
) -> None:
    inbox = tmp_path / "inbox"
    _write_record(
        inbox,
        UUID_A,
        session_id="sess-42",
        skill="explain-code",
        model="gemma4:27b",
        status="error",
        timestamp="2026-04-08T12:34:56+00:00",
        summary="boom: something went wrong",
    )
    out = StringIO()
    hook.main(inbox_dir=inbox, stdout=out)
    text = out.getvalue()
    assert 'session_id="sess-42"' in text
    assert 'skill="explain-code"' in text
    assert 'model="gemma4:27b"' in text
    assert 'status="error"' in text
    assert 'timestamp="2026-04-08T12:34:56+00:00"' in text
    assert "boom: something went wrong" in text


def test_first_run_writes_cursor_to_latest_id(
    hook: ModuleType, tmp_path: Path
) -> None:
    inbox = tmp_path / "inbox"
    _write_record(inbox, UUID_A)
    _write_record(inbox, UUID_B)
    rc = hook.main(inbox_dir=inbox, stdout=StringIO())
    assert rc == 0
    cursor = (inbox / ".cursor").read_text(encoding="utf-8").strip()
    assert cursor == UUID_B


def test_second_run_emits_only_new_records(
    hook: ModuleType, tmp_path: Path
) -> None:
    """After the first drain, only records newer than the cursor appear."""
    inbox = tmp_path / "inbox"
    _write_record(inbox, UUID_A, summary="first")

    out1 = StringIO()
    hook.main(inbox_dir=inbox, stdout=out1)
    assert "first" in out1.getvalue()

    _write_record(inbox, UUID_B, summary="second")

    out2 = StringIO()
    hook.main(inbox_dir=inbox, stdout=out2)
    assert "second" in out2.getvalue()
    assert "first" not in out2.getvalue()
    assert (inbox / ".cursor").read_text(encoding="utf-8").strip() == UUID_B


def test_no_new_records_no_output(
    hook: ModuleType, tmp_path: Path
) -> None:
    """When the cursor already covers all records, output is empty."""
    inbox = tmp_path / "inbox"
    _write_record(inbox, UUID_A)
    hook.main(inbox_dir=inbox, stdout=StringIO())  # advance cursor

    out = StringIO()
    rc = hook.main(inbox_dir=inbox, stdout=out)
    assert rc == 0
    assert out.getvalue() == ""


def test_idempotent_repeated_runs_yield_no_duplicates(
    hook: ModuleType, tmp_path: Path
) -> None:
    inbox = tmp_path / "inbox"
    _write_record(inbox, UUID_A, summary="only one")

    out1 = StringIO()
    out2 = StringIO()
    out3 = StringIO()
    hook.main(inbox_dir=inbox, stdout=out1)
    hook.main(inbox_dir=inbox, stdout=out2)
    hook.main(inbox_dir=inbox, stdout=out3)

    assert "only one" in out1.getvalue()
    assert out2.getvalue() == ""
    assert out3.getvalue() == ""


def test_corrupt_json_record_skipped(
    hook: ModuleType, tmp_path: Path
) -> None:
    """A malformed JSON file does not block the rest of the inbox."""
    inbox = tmp_path / "inbox"
    _write_record(inbox, UUID_A, summary="good1")
    (inbox / f"{UUID_B}.json").write_text("{not valid json", encoding="utf-8")
    _write_record(inbox, UUID_C, summary="good2")

    out = StringIO()
    rc = hook.main(inbox_dir=inbox, stdout=out)

    assert rc == 0
    text = out.getvalue()
    assert "good1" in text
    assert "good2" in text


def test_corrupt_cursor_treated_as_empty(
    hook: ModuleType, tmp_path: Path
) -> None:
    """A garbage cursor file self-heals: hook treats it as no cursor."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / ".cursor").write_text("garbage cursor value", encoding="utf-8")
    _write_record(inbox, UUID_A, summary="should still be emitted")

    out = StringIO()
    rc = hook.main(inbox_dir=inbox, stdout=out)

    assert rc == 0
    assert "should still be emitted" in out.getvalue()
    # Cursor should now be the valid uuid, replacing the garbage.
    assert (inbox / ".cursor").read_text(encoding="utf-8").strip() == UUID_A


def test_empty_cursor_file_treated_as_no_cursor(
    hook: ModuleType, tmp_path: Path
) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / ".cursor").write_text("", encoding="utf-8")
    _write_record(inbox, UUID_A, summary="visible")

    out = StringIO()
    hook.main(inbox_dir=inbox, stdout=out)
    assert "visible" in out.getvalue()


def test_atomic_cursor_write_leaves_no_tempfiles(
    hook: ModuleType, tmp_path: Path
) -> None:
    inbox = tmp_path / "inbox"
    _write_record(inbox, UUID_A)
    hook.main(inbox_dir=inbox, stdout=StringIO())

    leftover = sorted(p.name for p in inbox.iterdir())
    # Expect: the json record + .cursor only. No .cursor.*.tmp.
    assert ".cursor" in leftover
    assert not any(name.endswith(".tmp") for name in leftover)


def test_xml_special_chars_escaped_in_summary(
    hook: ModuleType, tmp_path: Path
) -> None:
    """Summaries containing < > & " do not break the framing."""
    inbox = tmp_path / "inbox"
    _write_record(
        inbox,
        UUID_A,
        summary='<script>alert("x")</script> & more',
    )
    out = StringIO()
    hook.main(inbox_dir=inbox, stdout=out)
    text = out.getvalue()
    # Raw < > & should be escaped so they can't break the outer tag.
    assert "&lt;script&gt;" in text
    assert "&amp;" in text
    assert "&quot;" in text
    # Outer tag still parses cleanly.
    assert text.count("<subagent-mcp-notification") == 1
    assert text.count("</subagent-mcp-notification>") == 1


def test_read_limit_caps_first_run_emission(
    hook: ModuleType, tmp_path: Path
) -> None:
    """A backlog larger than READ_LIMIT drains across runs, not at once."""
    inbox = tmp_path / "inbox"
    # Create READ_LIMIT + 5 records.
    base = "01939a55-1234-7000-8000-{:012d}"
    ids = [base.format(i) for i in range(hook.READ_LIMIT + 5)]
    for nid in ids:
        _write_record(inbox, nid)

    out = StringIO()
    hook.main(inbox_dir=inbox, stdout=out)

    # First run emits the most recent READ_LIMIT records (tail view).
    assert out.getvalue().count("<subagent-mcp-notification") == hook.READ_LIMIT
    # Cursor is at the latest record overall.
    assert (inbox / ".cursor").read_text(encoding="utf-8").strip() == ids[-1]


def test_dotfiles_in_inbox_dir_ignored(
    hook: ModuleType, tmp_path: Path
) -> None:
    """Hidden files (e.g. .cursor, tempfiles) must not be parsed as records."""
    inbox = tmp_path / "inbox"
    _write_record(inbox, UUID_A, summary="real")
    # A hidden file that would otherwise match *.json
    (inbox / ".hidden.json").write_text(
        json.dumps({"notification_id": "x", "summary": "ghost"}),
        encoding="utf-8",
    )

    out = StringIO()
    hook.main(inbox_dir=inbox, stdout=out)
    text = out.getvalue()
    assert "real" in text
    assert "ghost" not in text


# -- subprocess / shim integration tests ---------------------------------


def test_subprocess_invocation_via_python(tmp_path: Path) -> None:
    """Run the script as a subprocess to exercise the __main__ wiring."""
    inbox = tmp_path / "inbox"
    _write_record(inbox, UUID_A, summary="from subprocess")

    result = subprocess.run(
        [sys.executable, str(HOOK_PY)],
        env={**os.environ, "SUBAGENT_MCP_INBOX_DIR": str(inbox)},
        capture_output=True,
        text=True,
        input="",  # simulate Claude Code passing event JSON on stdin
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "from subprocess" in result.stdout
    assert (inbox / ".cursor").read_text(encoding="utf-8").strip() == UUID_A


def test_subprocess_invocation_via_shell_shim(tmp_path: Path) -> None:
    """The bash shim must successfully exec the python script."""
    inbox = tmp_path / "inbox"
    _write_record(inbox, UUID_A, summary="via shim")

    result = subprocess.run(
        ["bash", str(HOOK_SH)],
        env={**os.environ, "SUBAGENT_MCP_INBOX_DIR": str(inbox)},
        capture_output=True,
        text=True,
        input="",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "via shim" in result.stdout


def test_subprocess_no_inbox_dir_returns_zero(tmp_path: Path) -> None:
    """No inbox dir, no notifications, no error -- safe default."""
    result = subprocess.run(
        [sys.executable, str(HOOK_PY)],
        env={
            **os.environ,
            "SUBAGENT_MCP_INBOX_DIR": str(tmp_path / "definitely-not-here"),
        },
        capture_output=True,
        text=True,
        input="",
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
