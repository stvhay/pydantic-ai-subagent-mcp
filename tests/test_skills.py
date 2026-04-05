"""Tests for skill discovery."""

from pathlib import Path

from pydantic_ai_subagent_mcp.skills import discover_skills


def test_discover_markdown_skills(tmp_path: Path):
    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()
    (commands_dir / "greet.md").write_text("# Greet\n\nSay hello to the user.\n")
    (commands_dir / "review.md").write_text("# Review\n\nReview code changes.\n")

    skills = discover_skills([commands_dir])
    assert len(skills) == 2
    names = {s.name for s in skills}
    assert "greet" in names
    assert "review" in names


def test_skill_description_extraction(tmp_path: Path):
    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()
    (commands_dir / "deploy.md").write_text(
        "# Deploy\n\nDeploy the application to production.\n\n## Steps\n..."
    )

    skills = discover_skills([commands_dir])
    assert len(skills) == 1
    assert "Deploy the application" in skills[0].description


def test_nested_skills(tmp_path: Path):
    commands_dir = tmp_path / "commands"
    sub = commands_dir / "ops"
    sub.mkdir(parents=True)
    (sub / "restart.md").write_text("Restart the service gracefully.\n")

    skills = discover_skills([commands_dir])
    assert len(skills) == 1
    assert skills[0].name == "ops:restart"
