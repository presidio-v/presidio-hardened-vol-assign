"""Tests for the Wilcoxon rank-sum HV significance layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from presidio_vol_assign.benchmark import BenchmarkRow
from presidio_vol_assign.stats import (
    MIN_SAMPLES,
    wilcoxon_hv_tests,
    write_hv_tests_csv,
)


def _row(solver: str, hv_samples: list[float], size: str = "small") -> BenchmarkRow:
    """A BenchmarkRow carrying only the fields the stats layer reads."""
    n = len(hv_samples)
    mean = sum(hv_samples) / n if n else 0.0
    return BenchmarkRow(
        model="humanitarian",
        size=size,
        solver=solver,
        n_instances=n,
        nns_mean=1.0,
        nns_std=0.0,
        mid_mean=0.0,
        mid_std=0.0,
        sm_mean=0.0,
        sm_std=0.0,
        hv_mean=mean,
        hv_std=0.0,
        cpu_mean=0.0,
        cpu_std=0.0,
        hv_samples=hv_samples,
    )


def test_no_tests_when_single_solver() -> None:
    rows = [_row("nsga2", [0.4] * MIN_SAMPLES)]
    assert wilcoxon_hv_tests(rows) == []


def test_no_tests_when_too_few_samples() -> None:
    rows = [_row("nsga2", [0.4] * 3), _row("greedy", [0.3] * 3)]
    assert wilcoxon_hv_tests(rows) == []


def test_reference_prefers_greedy_baseline() -> None:
    rows = [
        _row("nsga2", [0.50] * MIN_SAMPLES),
        _row("nrga", [0.48] * MIN_SAMPLES),
        _row("greedy", [0.30] * MIN_SAMPLES),
    ]
    tests = wilcoxon_hv_tests(rows)
    # Every comparison is against the greedy baseline.
    assert tests
    assert {t.reference for t in tests} == {"greedy"}
    assert {t.solver for t in tests} == {"nsga2", "nrga"}


def test_significant_difference_detected_and_direction() -> None:
    # Framework clearly dominates the baseline on HV → significant, framework better.
    rows = [
        _row("nsga2", [0.60, 0.61, 0.59, 0.62, 0.60, 0.63]),
        _row("greedy", [0.30, 0.31, 0.29, 0.32, 0.30, 0.28]),
    ]
    (t,) = wilcoxon_hv_tests(rows)
    assert t.solver == "nsga2" and t.reference == "greedy"
    assert t.better == "nsga2"
    assert t.significant is True
    assert t.p_value < 0.05


def test_no_significant_difference_for_identical_distributions() -> None:
    rows = [
        _row("nsga2", [0.40, 0.42, 0.41, 0.43, 0.39, 0.44]),
        _row("greedy", [0.40, 0.42, 0.41, 0.43, 0.39, 0.44]),
    ]
    (t,) = wilcoxon_hv_tests(rows)
    assert t.significant is False
    assert t.better == "tie"


def test_write_hv_tests_csv(tmp_path: Path) -> None:
    rows = [
        _row("nsga2", [0.60] * MIN_SAMPLES),
        _row("greedy", [0.30] * MIN_SAMPLES),
    ]
    tests = wilcoxon_hv_tests(rows)
    path = write_hv_tests_csv(tests, tmp_path)
    assert path.exists()
    # Must NOT collide with the benchmark_*.csv summary glob.
    assert path.name.startswith("stats_")
    assert not path.name.startswith("benchmark_")
    df = pd.read_csv(path)
    assert {"size", "solver", "reference", "p_value", "better", "significant"} <= set(df.columns)
