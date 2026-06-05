"""Integration tests for the `pva benchmark` command."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from presidio_vol_assign.cli import app

runner = CliRunner()

FAST = ["--instances", "1", "--pop-size", "6", "--generations", "2", "--seed", "7"]


@pytest.mark.slow
def test_benchmark_ed_small_writes_summary(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--model",
            "ed-staffing",
            "--size",
            "small",
            "--solver",
            "both",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert "Benchmark summary" in result.output

    csv_files = list(tmp_path.glob("benchmark_*.csv"))
    json_files = list(tmp_path.glob("benchmark_*.json"))
    assert len(csv_files) == 1 and len(json_files) == 1

    df = pd.read_csv(csv_files[0])
    assert {"nsga2", "nrga"} == set(df["solver"])

    # SDLC #4/#5: on-run audit + security-event log emitted for benchmark too
    log = tmp_path / "pva.log"
    assert log.exists()
    assert "loaded" in log.read_text()


@pytest.mark.slow
def test_benchmark_check_repro_reports_rep(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--model",
            "ed-staffing",
            "--size",
            "small",
            "--solver",
            "nsga2",
            "--check-repro",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    assert result.exit_code == 0, result.output
    data = json.loads(next(tmp_path.glob("benchmark_*.json")).read_text())
    assert data[0]["rep"] == 1.0


def test_benchmark_unknown_model_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["benchmark", "--model", "nope", "--size", "small", "--output", str(tmp_path)] + FAST
    )
    assert result.exit_code == 1


def test_benchmark_bad_size_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["benchmark", "--model", "ed-staffing", "--size", "gigantic", "--output", str(tmp_path)]
        + FAST,
    )
    assert result.exit_code == 1
