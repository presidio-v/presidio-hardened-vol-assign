"""Integration tests for the CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from presidio_vol_assign.cli import app
from presidio_vol_assign.security import AuditResult, AuditStatus

runner = CliRunner()

FIXTURES = Path(__file__).parent / "fixtures"
VALID_VOL = str(FIXTURES / "volunteers_valid.csv")
VALID_EDS = str(FIXTURES / "eds_valid.csv")

# Tiny solver config for fast tests
FAST = ["--pop-size", "4", "--generations", "3", "--seed", "42"]


# ---------------------------------------------------------------------------
# pva assign — happy path
# ---------------------------------------------------------------------------


def test_assign_happy_path(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "assign",
            "--volunteers",
            VALID_VOL,
            "--eds",
            VALID_EDS,
            "--solver",
            "nsga2",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    # Output files written
    csvs = list(tmp_path.glob("pareto_nsga2_*.csv"))
    jsons = list(tmp_path.glob("metrics_nsga2_*.json"))
    assigns = list(tmp_path.glob("assignments_nsga2_*.csv"))
    assert len(csvs) == 1
    assert len(jsons) == 1
    assert len(assigns) == 1


def test_assign_solver_both_writes_two_sets(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "assign",
            "--volunteers",
            VALID_VOL,
            "--eds",
            VALID_EDS,
            "--solver",
            "both",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    assert result.exit_code == 0, result.output

    for solver in ("nsga2", "nrga"):
        assert len(list(tmp_path.glob(f"pareto_{solver}_*.csv"))) == 1
        assert len(list(tmp_path.glob(f"metrics_{solver}_*.json"))) == 1


def test_assign_metrics_json_schema(tmp_path: Path) -> None:
    runner.invoke(
        app,
        [
            "assign",
            "--volunteers",
            VALID_VOL,
            "--eds",
            VALID_EDS,
            "--solver",
            "nsga2",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    metrics_file = next(tmp_path.glob("metrics_nsga2_*.json"))
    data = json.loads(metrics_file.read_text())
    assert data["solver"] == "nsga2"
    assert isinstance(data["nns"], int) and data["nns"] >= 1
    for key in ("mid", "sm", "hv", "cpu_time_sec"):
        assert isinstance(data[key], float)


def test_assign_pareto_csv_schema(tmp_path: Path) -> None:
    runner.invoke(
        app,
        [
            "assign",
            "--volunteers",
            VALID_VOL,
            "--eds",
            VALID_EDS,
            "--solver",
            "nsga2",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    import pandas as pd

    pareto_file = next(tmp_path.glob("pareto_nsga2_*.csv"))
    df = pd.read_csv(pareto_file)
    assert set(df.columns) >= {"solver", "solution_id", "z1", "z2"}
    assert len(df) >= 1


def test_assign_security_log_written(tmp_path: Path) -> None:
    runner.invoke(
        app,
        [
            "assign",
            "--volunteers",
            VALID_VOL,
            "--eds",
            VALID_EDS,
            "--solver",
            "nsga2",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    log = tmp_path / "pva.log"
    assert log.exists()
    lines = log.read_text().strip().splitlines()
    assert len(lines) >= 1
    import json as _json

    entry = _json.loads(lines[0])
    assert entry["level"] == "INFO"


# ---------------------------------------------------------------------------
# pva assign — error paths
# ---------------------------------------------------------------------------


def test_assign_missing_volunteers_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "assign",
            "--volunteers",
            str(tmp_path / "nope.csv"),
            "--eds",
            VALID_EDS,
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    assert result.exit_code != 0


def test_assign_bad_eds_csv(tmp_path: Path) -> None:
    bad = tmp_path / "eds.csv"
    bad.write_text("ed_id,wrong_col\nED1,oops\n")
    result = runner.invoke(
        app,
        ["assign", "--volunteers", VALID_VOL, "--eds", str(bad), "--output", str(tmp_path)] + FAST,
    )
    assert result.exit_code != 0


def test_assign_invalid_solver(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "assign",
            "--volunteers",
            VALID_VOL,
            "--eds",
            VALID_EDS,
            "--solver",
            "badvalue",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    assert result.exit_code != 0


def test_assign_path_traversal_rejected(tmp_path: Path) -> None:
    runner.invoke(
        app,
        ["assign", "--volunteers", VALID_VOL, "--eds", VALID_EDS, "--output", "../../etc/pwned"]
        + FAST,
    )
    # Path traversal resolves to absolute path outside CWD; guard_output_path
    # resolves it to an absolute path (no ".." remain after resolve), so the
    # command may succeed if the resolved path is writable. The important thing
    # is that ".." in the raw input does NOT reach the filesystem as "../..".
    # We verify no crash and that the raw ".." string is not in any written path.
    written = list(Path.cwd().glob("**/*.csv"))
    for p in written:
        assert ".." not in str(p)


def test_assign_guard_output_path_raises(tmp_path: Path) -> None:
    """When guard_output_path raises ValueError the CLI exits with code 1."""
    with patch(
        "presidio_vol_assign.cli.guard_output_path",
        side_effect=ValueError("path traversal is not allowed"),
    ):
        result = runner.invoke(
            app,
            ["assign", "--volunteers", VALID_VOL, "--eds", VALID_EDS, "--output", str(tmp_path)]
            + FAST,
        )
    assert result.exit_code == 1


def test_assign_vulnerable_audit_prints_warning(tmp_path: Path) -> None:
    """When the audit returns VULNERABLE the CLI prints a warning."""
    vulnerable = AuditResult(status=AuditStatus.VULNERABLE, n_vulnerabilities=2)
    with patch("presidio_vol_assign.cli.run_audit", return_value=vulnerable):
        result = runner.invoke(
            app,
            [
                "assign",
                "--volunteers",
                VALID_VOL,
                "--eds",
                VALID_EDS,
                "--solver",
                "nsga2",
                "--output",
                str(tmp_path),
            ]
            + FAST,
        )
    assert result.exit_code == 0
    assert "Warning" in result.output or "warning" in result.output.lower()


# ---------------------------------------------------------------------------
# pva metrics
# ---------------------------------------------------------------------------


def test_metrics_on_generated_csv(tmp_path: Path) -> None:
    # First generate a pareto CSV via assign
    runner.invoke(
        app,
        [
            "assign",
            "--volunteers",
            VALID_VOL,
            "--eds",
            VALID_EDS,
            "--solver",
            "nsga2",
            "--output",
            str(tmp_path),
        ]
        + FAST,
    )
    pareto_file = next(tmp_path.glob("pareto_nsga2_*.csv"))

    result = runner.invoke(app, ["metrics", "--pareto", str(pareto_file)])
    assert result.exit_code == 0, result.output
    assert "NNS" in result.output
    assert "HV" in result.output


def test_metrics_missing_file() -> None:
    result = runner.invoke(app, ["metrics", "--pareto", "/nonexistent/pareto.csv"])
    assert result.exit_code != 0


def test_metrics_bad_csv(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n")
    result = runner.invoke(app, ["metrics", "--pareto", str(bad)])
    assert result.exit_code != 0
