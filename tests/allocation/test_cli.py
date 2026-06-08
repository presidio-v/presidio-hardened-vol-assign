"""Integration tests for `pva allocate` and `pva alloc-metrics` via Typer CliRunner."""

from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from presidio_vol_assign.cli import app


class TestAllocateCommand:
    def test_help_renders(self):
        runner = CliRunner()
        r = runner.invoke(app, ["allocate", "--help"])
        assert r.exit_code == 0
        assert "Resourcefulness" in r.stdout

    def test_happy_path_4obj(self, csv_dir):
        runner = CliRunner()
        out = csv_dir / "out"
        r = runner.invoke(
            app,
            [
                "allocate",
                "--people",
                str(csv_dir / "people.csv"),
                "--centers",
                str(csv_dir / "centers.csv"),
                "--travel",
                str(csv_dir / "travel.csv"),
                "--n-dir",
                "4",
                "--solver",
                "nsga2",
                "--objectives",
                "4",
                "--pop-size",
                "20",
                "--generations",
                "10",
                "--seed",
                "42",
                "--output",
                str(out),
            ],
        )
        assert r.exit_code == 0, f"stdout: {r.stdout}\nexc: {r.exception}"
        # Three output files plus the log
        files = sorted(p.name for p in out.iterdir())
        assert any(f.startswith("pareto_alloc_nsga2_4obj_") and f.endswith(".csv") for f in files)
        assert any(f.startswith("metrics_alloc_nsga2_4obj_") and f.endswith(".json") for f in files)
        assert any(
            f.startswith("allocations_alloc_nsga2_4obj_") and f.endswith(".csv") for f in files
        )

    def test_invalid_solver_rejected(self, csv_dir):
        runner = CliRunner()
        r = runner.invoke(
            app,
            [
                "allocate",
                "--people",
                str(csv_dir / "people.csv"),
                "--centers",
                str(csv_dir / "centers.csv"),
                "--travel",
                str(csv_dir / "travel.csv"),
                "--n-dir",
                "4",
                "--solver",
                "lemonade",
            ],
        )
        assert r.exit_code != 0


class TestFISRulesOption:
    def test_allocate_with_fis_rules_drop(self, csv_dir):
        runner = CliRunner()
        spec = csv_dir / "fis_rules.json"
        spec.write_text(json.dumps({"fis1": [0, 1], "fis2b_rpd": [2]}))
        out = csv_dir / "out_fis_dropped"
        r = runner.invoke(
            app,
            [
                "allocate",
                "--people",
                str(csv_dir / "people.csv"),
                "--centers",
                str(csv_dir / "centers.csv"),
                "--travel",
                str(csv_dir / "travel.csv"),
                "--n-dir",
                "4",
                "--solver",
                "nsga2",
                "--objectives",
                "4",
                "--pop-size",
                "15",
                "--generations",
                "5",
                "--seed",
                "42",
                "--fis-rules",
                str(spec),
                "--output",
                str(out),
            ],
        )
        assert r.exit_code == 0, f"stdout: {r.stdout}\nexc: {r.exception}"

    def test_allocate_invalid_fis_rules_path(self, csv_dir):
        runner = CliRunner()
        r = runner.invoke(
            app,
            [
                "allocate",
                "--people",
                str(csv_dir / "people.csv"),
                "--centers",
                str(csv_dir / "centers.csv"),
                "--travel",
                str(csv_dir / "travel.csv"),
                "--n-dir",
                "4",
                "--fis-rules",
                str(csv_dir / "missing_spec.json"),
            ],
        )
        assert r.exit_code != 0


class TestWeightSweepCommand:
    def test_help_renders(self):
        runner = CliRunner()
        r = runner.invoke(app, ["allocate-weight-sweep", "--help"])
        assert r.exit_code == 0
        assert "H3b" in r.stdout

    def test_happy_path_writes_manifest(self, csv_dir):
        runner = CliRunner()
        out = csv_dir / "out_sweep"
        r = runner.invoke(
            app,
            [
                "allocate-weight-sweep",
                "--people",
                str(csv_dir / "people.csv"),
                "--centers",
                str(csv_dir / "centers.csv"),
                "--travel",
                str(csv_dir / "travel.csv"),
                "--n-dir",
                "4",
                "--solver",
                "nsga2",
                "--objectives",
                "4",
                "--pop-size",
                "15",
                "--generations",
                "5",
                "--seed",
                "42",
                "--n-samples",
                "3",
                "--bound",
                "0.1",
                "--lhs-seed",
                "7",
                "--output",
                str(out),
            ],
        )
        assert r.exit_code == 0, f"stdout: {r.stdout}\nexc: {r.exception}"
        manifest = out / "weight_sweep_manifest.csv"
        assert manifest.exists()
        df = pd.read_csv(manifest)
        assert len(df) == 3

    def test_sweep_rejects_solver_all(self, csv_dir):
        runner = CliRunner()
        r = runner.invoke(
            app,
            [
                "allocate-weight-sweep",
                "--people",
                str(csv_dir / "people.csv"),
                "--centers",
                str(csv_dir / "centers.csv"),
                "--travel",
                str(csv_dir / "travel.csv"),
                "--n-dir",
                "4",
                "--solver",
                "all",
            ],
        )
        assert r.exit_code != 0


class TestAllocMetricsCommand:
    def test_alloc_metrics_recompute_round_trip(self, csv_dir):
        runner = CliRunner()
        out = csv_dir / "out"
        r = runner.invoke(
            app,
            [
                "allocate",
                "--people",
                str(csv_dir / "people.csv"),
                "--centers",
                str(csv_dir / "centers.csv"),
                "--travel",
                str(csv_dir / "travel.csv"),
                "--n-dir",
                "4",
                "--solver",
                "nsga2",
                "--objectives",
                "4",
                "--pop-size",
                "15",
                "--generations",
                "5",
                "--seed",
                "7",
                "--output",
                str(out),
            ],
        )
        assert r.exit_code == 0
        pareto_csv = next(p for p in out.iterdir() if p.name.startswith("pareto_alloc_"))
        r2 = runner.invoke(app, ["alloc-metrics", "--pareto", str(pareto_csv)])
        assert r2.exit_code == 0, f"stdout: {r2.stdout}\nexc: {r2.exception}"
        assert "NNS" in r2.stdout
        assert "HV" in r2.stdout
