"""Integration test for the `pva ablation` command."""

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


def test_ablation_humanitarian_writes_csv(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "ablation",
            "--model",
            "humanitarian",
            "--people",
            PEOPLE,
            "--centers",
            CENTERS,
            "--solver",
            "nsga2",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    csv = next(tmp_path.glob("ablation_*.csv"))
    df = pd.read_csv(csv)
    assert {"dropped", "delta_dropped", "delta_hv"} <= set(df.columns)
    # One row per humanitarian objective.
    assert sorted(df["dropped"].tolist()) == ["z1", "z2", "z3"]
    assert (tmp_path / "pva.log").exists()


def test_ablation_missing_inputs_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["ablation", "--model", "humanitarian", "--output", str(tmp_path)] + FAST
    )
    assert result.exit_code == 1
