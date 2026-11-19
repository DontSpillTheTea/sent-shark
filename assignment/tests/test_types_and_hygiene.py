from pathlib import Path
from typing import get_args

from assignment.models import JudgmentRecord
from assignment.types import EntailmentLabel, TierName


def test_custom_literal_types_match_model_choices():
    assert set(get_args(EntailmentLabel)) == {choice[0] for choice in JudgmentRecord.LABEL_CHOICES}
    assert set(get_args(TierName)) == {choice[0] for choice in JudgmentRecord.TIER_CHOICES}


def test_root_gitignore_blocks_internal_and_agent_directories():
    gitignore_path = Path(__file__).resolve().parents[2] / ".gitignore"
    gitignore_lines = {
        line.strip()
        for line in gitignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert ".superpowers/" in gitignore_lines
    assert ".local_docs/" in gitignore_lines
    assert ".env" in gitignore_lines


def test_root_dockerignore_excludes_local_bloat_and_preserves_env_examples():
    dockerignore_path = Path(__file__).resolve().parents[2] / ".dockerignore"
    assert dockerignore_path.exists(), "Missing .dockerignore at repository root"

    lines = [
        line.strip()
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    line_set = set(lines)
    required_excludes = {
        ".git/",
        ".venv/",
        ".local_docs/",
        ".superpowers/",
        "__pycache__/",
        ".pytest_cache/",
        "*.pyc",
        "db.sqlite3*",
        ".env",
        ".env.*",
    }
    missing = required_excludes - line_set
    assert not missing, f".dockerignore missing required patterns: {sorted(missing)}"

    # Example env files must be un-ignored AFTER .env.* so Docker builds keep setup templates
    assert "!.env.example" in line_set
    assert lines.index("!.env.example") > lines.index(".env.*")
