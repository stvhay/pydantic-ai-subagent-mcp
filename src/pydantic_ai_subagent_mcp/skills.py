"""Discover Claude Code skills from the standard locations.

A "skill" is a directory containing a ``SKILL.md`` file whose first
block is YAML frontmatter declaring at least ``name`` and
``description``. This is the modern Claude Code skill format used by
both user skills (``~/.claude/skills/``) and plugins
(``~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/skills/``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    """Find every ``SKILL.md`` reachable from ``search_dirs``.

    Skills are de-duplicated by ``name`` in the order the search
    directories are visited, so earlier directories take precedence.
    With the default search dirs that means project skills shadow user
    skills, which in turn shadow plugin-cache skills.
    """
    if search_dirs is None:
        search_dirs = _default_search_dirs()

    skills: list[Skill] = []
    seen: set[str] = set()

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for skill_md in search_dir.rglob("SKILL.md"):
            skill = _parse_skill_md(skill_md)
            if skill is None:
                continue
            if skill.name in seen:
                continue
            seen.add(skill.name)
            skills.append(skill)

    return skills


def _default_search_dirs() -> list[Path]:
    """Build the default list of directories to search for ``SKILL.md`` files.

    Order matters: earlier dirs win on name collisions.

    1. ``./.claude/skills/`` (project)
    2. ``~/.claude/skills/`` (user)
    3. ``~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/skills/`` (every installed plugin)
    """
    dirs: list[Path] = []

    project_skills = Path(".claude/skills")
    if project_skills.exists():
        dirs.append(project_skills)

    user_skills = Path.home() / ".claude" / "skills"
    if user_skills.exists():
        dirs.append(user_skills)

    plugins_cache = Path.home() / ".claude" / "plugins" / "cache"
    if plugins_cache.exists():
        # marketplace / plugin / version / skills
        for skills_dir in sorted(plugins_cache.glob("*/*/*/skills")):
            if skills_dir.is_dir():
                dirs.append(skills_dir)

    return dirs


_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def _parse_skill_md(skill_md_path: Path) -> Skill | None:
    """Parse a ``SKILL.md`` into a ``Skill``, returning ``None`` on failure.

    The file must open with a YAML frontmatter block delimited by
    ``---`` lines, and the frontmatter must define a non-empty ``name``.
    Unknown frontmatter keys are tolerated. A missing ``description``
    falls through to the empty string rather than rejecting the skill.
    """
    try:
        content = skill_md_path.read_text()
    except OSError:
        return None

    match = _FRONTMATTER_RE.match(content)
    if not match:
        return None

    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(frontmatter, dict):
        return None

    name = frontmatter.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    description = frontmatter.get("description")
    if not isinstance(description, str):
        description = ""

    return Skill(
        name=name.strip(),
        description=description.strip(),
        source_path=skill_md_path,
        plugin_name=_extract_plugin_name(skill_md_path),
    )


def _extract_plugin_name(skill_md_path: Path) -> str | None:
    """Return the owning plugin's name when the skill lives in a plugin cache.

    Recognises the standard layout
    ``.../plugins/cache/<marketplace>/<plugin>/<version>/skills/<skill>/SKILL.md``
    and returns ``<plugin>``. For project or user skills (no plugin
    cache in the path), returns ``None``.
    """
    parts = skill_md_path.parts
    for i in range(1, len(parts) - 2):
        if parts[i] == "cache" and parts[i - 1] == "plugins":
            return parts[i + 2]
    return None
