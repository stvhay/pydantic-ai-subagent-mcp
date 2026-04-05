"""Discover Claude Code skills from the current directory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    """A discovered Claude Code skill."""

    name: str
    description: str
    source_path: Path
    plugin_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source_path": str(self.source_path),
            "plugin_name": self.plugin_name,
        }


def discover_skills(search_dirs: list[Path] | None = None) -> list[Skill]:
    """Find all skills available to Claude Code.

    Searches:
    1. .claude/commands/ in the current directory (project commands)
    2. ~/.claude/commands/ (user commands)
    3. Plugin skill directories from .claude/settings.json
    """
    skills: list[Skill] = []

    if search_dirs is None:
        search_dirs = _default_search_dirs()

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        # Markdown-based skills (Claude Code slash commands)
        for md_file in search_dir.rglob("*.md"):
            skill = _parse_markdown_skill(md_file, search_dir)
            if skill:
                skills.append(skill)

    return skills


def _default_search_dirs() -> list[Path]:
    """Build the default list of directories to search for skills."""
    dirs: list[Path] = []

    # Project-level commands
    project_cmds = Path(".claude/commands")
    if project_cmds.exists():
        dirs.append(project_cmds)

    # User-level commands
    user_cmds = Path.home() / ".claude" / "commands"
    if user_cmds.exists():
        dirs.append(user_cmds)

    # Plugin skills from settings
    plugin_dirs = _discover_plugin_skill_dirs()
    dirs.extend(plugin_dirs)

    return dirs


def _discover_plugin_skill_dirs() -> list[Path]:
    """Find skill directories from installed Claude Code plugins."""
    dirs: list[Path] = []
    settings_path = Path(".claude/settings.json")
    if not settings_path.exists():
        return dirs

    try:
        settings = json.loads(settings_path.read_text())
        # Look for plugin references in permissions or other config
        plugin_dirs_config = settings.get("pluginDirs", [])
        for pd in plugin_dirs_config:
            plugin_path = Path(pd)
            skills_dir = plugin_path / "skills"
            if skills_dir.exists():
                dirs.append(skills_dir)
    except (json.JSONDecodeError, KeyError):
        pass

    return dirs


def _parse_markdown_skill(md_path: Path, base_dir: Path) -> Skill | None:
    """Parse a markdown file as a skill definition."""
    try:
        content = md_path.read_text()
    except OSError:
        return None

    # Use filename (without extension) as skill name
    rel_path = md_path.relative_to(base_dir)
    name = str(rel_path.with_suffix("")).replace("/", ":")

    # Extract description from first paragraph or heading
    description = _extract_description(content)

    # Determine plugin name from path
    plugin_name = _extract_plugin_name(md_path)

    return Skill(
        name=name,
        description=description,
        source_path=md_path,
        plugin_name=plugin_name,
    )


def _extract_description(content: str) -> str:
    """Extract a one-line description from markdown content."""
    lines = content.strip().splitlines()
    for line in lines:
        stripped = line.strip()
        # Skip frontmatter
        if stripped == "---":
            continue
        # Skip headings, grab first real paragraph line
        if stripped and not stripped.startswith("#"):
            # Clean up markdown formatting
            desc = re.sub(r"[*_`]", "", stripped)
            return desc[:200]
    return "No description available"


def _extract_plugin_name(md_path: Path) -> str | None:
    """Try to extract plugin name from path."""
    parts = md_path.parts
    for i, part in enumerate(parts):
        if part == "plugins" and i + 1 < len(parts):
            return parts[i + 1]
        if part == "skills" and i > 0:
            return parts[i - 1]
    return None
