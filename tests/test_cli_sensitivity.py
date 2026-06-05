"""Integration test for the `pva sensitivity` command."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from presidio_vol_assign.cli import app

runner = CliRunner()

FIXTURES = Path(__file__).parent / "fixtures"
PEOPLE = str(FIXTURES / "people_valid.csv")
CENTERS = str(FIXTURES / "centers_valid.csv")
FAST = ["--pop-size", "8", "--generations", "4", "--seed", "42"]


def test_sensitivity_humanitarian_writes_csv(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "sensitivity",
            "--model",
            "humanitarian",
            "--people",
            PEOPLE,
            "--centers",
            CENTERS,
            "--factors",
            "-0.1,0,0.1",
            "--solver",
            "nsga2",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    csv = next(tmp_path.glob("sensitivity_*.csv"))
    df = pd.read_csv(csv)
    assert set(df.columns) == {"factor", "solver", "nns", "mid", "sm", "hv", "cpu_time_sec"}
    assert sorted(df["factor"].unique().tolist()) == [-0.1, 0.0, 0.1]
    # security-event log emitted
    assert (tmp_path / "pva.log").exists()


def test_sensitivity_missing_inputs_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["sensitivity", "--model", "humanitarian", "--output", str(tmp_path)] + FAST
    )
    assert result.exit_code == 1


def test_sensitivity_bad_factors_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "sensitivity",
            "--people",
            PEOPLE,
            "--centers",
            CENTERS,
            "--factors",
            "not,numbers",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    assert result.exit_code == 1
