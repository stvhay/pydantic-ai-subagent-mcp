#!/usr/bin/env python3
"""Compute and update project version from CHANGELOG.md.

Reads the current version from pyproject.toml and src/__init__.py.
Determines bump type from <!-- bump: TYPE --> comment in CHANGELOG.md.

Usage:
    python compute_version.py          # Print next version
    python compute_version.py --ci     # CI mode: rewrite Unreleased header, update version files
    python compute_version.py --update # Update version files only (no changelog rewrite)
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PYPROJECT = Path("pyproject.toml")
INIT_FILE = Path("src/pydantic_ai_subagent_mcp/__init__.py")
CHANGELOG = Path("CHANGELOG.md")

VERSION_RE = re.compile(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', re.MULTILINE)
INIT_VERSION_RE = re.compile(r'^__version__\s*=\s*"(\d+\.\d+\.\d+)"', re.MULTILINE)
BUMP_RE = re.compile(r"<!--\s*bump:\s*(major|minor|patch)\s*-->")


def read_current_version() -> str:
    """Read version from pyproject.toml."""
    content = PYPROJECT.read_text()
    match = VERSION_RE.search(content)
    if not match:
        print("Error: could not find version in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    return match.group(1)


def read_bump_type() -> str | None:
    """Read bump type from CHANGELOG.md <!-- bump: TYPE --> comment."""
    if not CHANGELOG.exists():
        return None
    content = CHANGELOG.read_text()
    match = BUMP_RE.search(content)
    return match.group(1) if match else None


def compute_next_version(current: str, bump: str) -> str:
    """Compute next version given current version and bump type."""
    parts = [int(x) for x in current.split(".")]
    if bump == "major":
        parts = [parts[0] + 1, 0, 0]
    elif bump == "minor":
        parts = [parts[0], parts[1] + 1, 0]
    elif bump == "patch":
        parts = [parts[0], parts[1], parts[2] + 1]
    return ".".join(str(p) for p in parts)


def update_pyproject(new_version: str) -> None:
    """Update version in pyproject.toml using regex replacement."""
    content = PYPROJECT.read_text()
    updated = VERSION_RE.sub(f'version = "{new_version}"', content)
    PYPROJECT.write_text(updated)


def update_init(new_version: str) -> None:
    """Update __version__ in __init__.py."""
    if not INIT_FILE.exists():
        return
    content = INIT_FILE.read_text()
    updated = INIT_VERSION_RE.sub(f'__version__ = "{new_version}"', content)
    INIT_FILE.write_text(updated)


def rewrite_changelog(new_version: str) -> None:
    """Replace '## Unreleased' with '## vX.Y.Z' and remove bump comment."""
    content = CHANGELOG.read_text()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content = re.sub(
        r"## Unreleased",
        f"## v{new_version} ({today})",
        content,
    )
    content = BUMP_RE.sub("", content)
    CHANGELOG.write_text(content)


def main() -> None:
    ci_mode = "--ci" in sys.argv
    update_mode = "--update" in sys.argv

    current = read_current_version()
    bump = read_bump_type()

    if bump is None:
        print(f"Current: {current} (no bump type found in CHANGELOG.md)")
        sys.exit(0)

    next_ver = compute_next_version(current, bump)

    if not ci_mode and not update_mode:
        print(next_ver)
        return

    print(f"Bumping {current} -> {next_ver} ({bump})")
    update_pyproject(next_ver)
    update_init(next_ver)

    if ci_mode:
        rewrite_changelog(next_ver)

    print(f"Updated to v{next_ver}")


if __name__ == "__main__":
    main()
