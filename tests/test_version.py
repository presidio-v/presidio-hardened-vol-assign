"""Smoke tests for the CLI scaffold, and version consistency across metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from presidio_vol_assign import __version__
from presidio_vol_assign.cli import app

runner = CliRunner()

REPO_ROOT = Path(__file__).parent.parent

pytestmark_repo = pytest.mark.skipif(
    not (REPO_ROOT / "pyproject.toml").exists(),
    reason="metadata files are only present in a source checkout",
)


def test_version_is_a_semver_string() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), __version__


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytestmark_repo
def test_pyproject_version_matches_package() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match, "no version field in pyproject.toml"
    assert match.group(1) == __version__


@pytestmark_repo
def test_citation_version_matches_package() -> None:
    text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version: (.+)$", text, re.MULTILINE)
    assert match, "no version field in CITATION.cff"
    assert match.group(1).strip() == __version__


@pytestmark_repo
def test_zenodo_version_matches_package() -> None:
    data = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert data["version"] == __version__


@pytestmark_repo
def test_changelog_has_an_entry_for_the_current_version() -> None:
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{__version__}]" in text
    assert f"## [{__version__}] — unreleased" not in text, (
        "the released version is still marked unreleased in the CHANGELOG"
    )


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "assign" in result.output
    assert "metrics" in result.output


def test_serve_is_registered() -> None:
    result = runner.invoke(app, ["--help"])
    assert "serve" in result.output
