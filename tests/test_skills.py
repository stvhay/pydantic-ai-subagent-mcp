"""Tests for skill discovery."""

from pathlib import Path

from pydantic_ai_subagent_mcp.skills import discover_skills


def _write_skill(
    base: Path,
    skill_dir: str,
    name: str,
    description: str,
    extra_frontmatter: str = "",
    body: str = "Body text.",
) -> Path:
    """Write a SKILL.md under base/skill_dir/SKILL.md and return its path."""
    target = base / skill_dir
    target.mkdir(parents=True, exist_ok=True)
    extra = f"\n{extra_frontmatter}" if extra_frontmatter else ""
    skill_md = target / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: {description}{extra}\n---\n\n{body}\n"
    )
    return skill_md


def test_discover_skill_md(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "greet", "greet", "Say hello to the user.")
    _write_skill(skills_dir, "review", "review", "Review code changes.")

    skills = discover_skills([skills_dir])

    assert {s.name for s in skills} == {"greet", "review"}
    assert {s.description for s in skills} == {
        "Say hello to the user.",
        "Review code changes.",
    }


def test_description_comes_from_frontmatter(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(
        skills_dir,
        "deploy",
        "deploy",
        "Deploy the application to production.",
        body="# Deploy\n\nDetailed body that should be ignored.",
    )

    skills = discover_skills([skills_dir])

    assert len(skills) == 1
    assert skills[0].description == "Deploy the application to production."


def test_extra_frontmatter_keys_tolerated(tmp_path: Path) -> None:
    """Skills with model/effort/etc. frontmatter (like benchmark skills) load."""
    skills_dir = tmp_path / "skills"
    _write_skill(
        skills_dir,
        "bench",
        "bench-sonnet-high",
        "Benchmark scorer",
        extra_frontmatter="model: sonnet\neffort: high\ncontext: fork",
    )

    skills = discover_skills([skills_dir])

    assert len(skills) == 1
    assert skills[0].name == "bench-sonnet-high"


def test_skips_files_without_frontmatter(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    bad = skills_dir / "broken"
    bad.mkdir(parents=True)
    (bad / "SKILL.md").write_text("# No frontmatter here\n\nJust prose.\n")

    assert discover_skills([skills_dir]) == []


def test_skips_files_with_blank_name(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    bad = skills_dir / "nameless"
    bad.mkdir(parents=True)
    (bad / "SKILL.md").write_text("---\nname: \ndescription: foo\n---\n")

    assert discover_skills([skills_dir]) == []


def test_dedup_by_name_first_dir_wins(tmp_path: Path) -> None:
    """When the same skill name appears in two search dirs, the first wins."""
    project = tmp_path / "project" / "skills"
    user = tmp_path / "user" / "skills"
    _write_skill(project, "shared", "shared", "Project version.")
    _write_skill(user, "shared", "shared", "User version.")

    skills = discover_skills([project, user])

    assert len(skills) == 1
    assert skills[0].description == "Project version."


def test_plugin_name_from_cache_path(tmp_path: Path) -> None:
    """A SKILL.md inside plugins/cache/<m>/<plugin>/<v>/skills/<s>/ tags the plugin."""
    plugin_skills = (
        tmp_path
        / "plugins"
        / "cache"
        / "my-marketplace"
        / "my-plugin"
        / "1.0.0"
        / "skills"
    )
    _write_skill(plugin_skills, "frobnicate", "frobnicate", "Frob the nicate.")

    skills = discover_skills([plugin_skills])

    assert len(skills) == 1
    assert skills[0].plugin_name == "my-plugin"


def test_plugin_name_none_for_user_skill(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "solo", "solo", "Standalone skill.")

    skills = discover_skills([skills_dir])

    assert skills[0].plugin_name is None


def test_default_search_dirs_handles_missing(monkeypatch, tmp_path: Path) -> None:
    """Calling discover_skills() with no args is safe even if HOME has nothing."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    # No project/user/plugin dirs exist under tmp_path -- should not raise
    assert discover_skills() == []
