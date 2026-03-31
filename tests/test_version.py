"""Smoke tests for the CLI scaffold."""

from typer.testing import CliRunner

from presidio_vol_assign import __version__
from presidio_vol_assign.cli import app

runner = CliRunner()


def test_version_string() -> None:
    assert __version__ == "0.1.0"


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "assign" in result.output
    assert "metrics" in result.output
