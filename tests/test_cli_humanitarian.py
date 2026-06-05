"""Integration tests for the humanitarian model via the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from presidio_vol_assign.cli import app

runner = CliRunner()

FIXTURES = Path(__file__).parent / "fixtures"
PEOPLE = str(FIXTURES / "people_valid.csv")
CENTERS = str(FIXTURES / "centers_valid.csv")

FAST = ["--pop-size", "8", "--generations", "4", "--seed", "42"]


def test_humanitarian_happy_path(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "assign",
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

    pareto = next(tmp_path.glob("pareto_nsga2_*.csv"))
    df = pd.read_csv(pareto)
    # Three-objective front: z1, z2, z3 all present
    assert {"solver", "solution_id", "z1", "z2", "z3"} <= set(df.columns)
    assert len(df) >= 1

    assigns = next(tmp_path.glob("assignments_nsga2_*.csv"))
    adf = pd.read_csv(assigns)
    assert set(adf.columns) == {
        "solution_id",
        "person_id",
        "center_id",
        "fairness",
        "transport",
        "overcrowding",
    }

    metrics_file = next(tmp_path.glob("metrics_nsga2_*.json"))
    data = json.loads(metrics_file.read_text())
    assert data["solver"] == "nsga2"
    assert data["nns"] >= 1


def test_humanitarian_metrics_roundtrip(tmp_path: Path) -> None:
    runner.invoke(
        app,
        [
            "assign",
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
    pareto = next(tmp_path.glob("pareto_nsga2_*.csv"))
    result = runner.invoke(app, ["metrics", "--pareto", str(pareto)])
    assert result.exit_code == 0, result.output
    assert "HV" in result.output


def test_humanitarian_missing_inputs_errors(tmp_path: Path) -> None:
    # --model humanitarian but ED inputs supplied → should error clearly
    result = runner.invoke(
        app,
        ["assign", "--model", "humanitarian", "--output", str(tmp_path)] + FAST,
    )
    assert result.exit_code == 1
    assert "requires" in result.output.lower() or "requires" in (result.stderr or "").lower()


def test_unknown_model_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "assign",
            "--model",
            "nope",
            "--people",
            PEOPLE,
            "--centers",
            CENTERS,
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    assert result.exit_code == 1


def test_ed_model_still_default(tmp_path: Path) -> None:
    # No --model flag → ed-staffing; humanitarian inputs absent is fine.
    vol = str(FIXTURES / "volunteers_valid.csv")
    eds = str(FIXTURES / "eds_valid.csv")
    result = runner.invoke(
        app,
        [
            "assign",
            "--volunteers",
            vol,
            "--eds",
            eds,
            "--solver",
            "nsga2",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    assert result.exit_code == 0, result.output
    pareto = next(tmp_path.glob("pareto_nsga2_*.csv"))
    df = pd.read_csv(pareto)
    # ED model is 2-objective: no z3 column
    assert "z3" not in df.columns
