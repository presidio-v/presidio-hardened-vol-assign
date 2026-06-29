"""Tests for the benchmark harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from presidio_vol_assign.benchmark import (
    BenchmarkRow,
    generate_instance,
    resolve_sizes,
    run_benchmark,
    write_benchmark_summary,
)
from presidio_vol_assign.models import HumanitarianProblem, ProblemInstance, RunConfig


def _tiny_cfg(solver: str = "both") -> RunConfig:
    return RunConfig(solver=solver, pop_size=6, generations=2, seed=7)


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------


def test_generate_humanitarian_sizes() -> None:
    prob = generate_instance("humanitarian", "small", seed=1)
    assert isinstance(prob, HumanitarianProblem)
    assert prob.n_centers == 5
    assert prob.n_people == 150
    # Capacity must cover demand (feasibility invariant)
    assert sum(c.capacity for c in prob.centers) >= sum(p.group_size for p in prob.people)


def test_generate_ed_sizes() -> None:
    prob = generate_instance("ed-staffing", "large", seed=1)
    assert isinstance(prob, ProblemInstance)
    assert prob.n_vacancies == 10
    assert prob.n_volunteers == 150


def test_generation_is_deterministic() -> None:
    a = generate_instance("humanitarian", "small", seed=123)
    b = generate_instance("humanitarian", "small", seed=123)
    assert [p.vulnerability for p in a.people] == [p.vulnerability for p in b.people]
    assert [c.capacity for c in a.centers] == [c.capacity for c in b.centers]


def test_generation_differs_by_seed() -> None:
    a = generate_instance("humanitarian", "small", seed=1)
    b = generate_instance("humanitarian", "small", seed=2)
    assert [p.vulnerability for p in a.people] != [p.vulnerability for p in b.people]


def test_generate_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        generate_instance("nope", "small", seed=1)


def test_generate_unknown_size_raises() -> None:
    with pytest.raises(ValueError, match="unknown size"):
        generate_instance("humanitarian", "huge", seed=1)


def test_resolve_sizes() -> None:
    assert resolve_sizes("both") == ["small", "large"]
    assert resolve_sizes("small") == ["small"]
    with pytest.raises(ValueError):
        resolve_sizes("nonsense")


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_benchmark_ed_small() -> None:
    rows = run_benchmark("ed-staffing", ["small"], n_instances=2, config=_tiny_cfg())
    solvers = {r.solver for r in rows}
    assert solvers == {"nsga2", "nrga"}
    for r in rows:
        assert r.n_instances == 2
        assert r.nns_mean >= 1
        assert r.rep is None  # no repro check requested


@pytest.mark.slow
def test_run_benchmark_repro_flag() -> None:
    rows = run_benchmark(
        "ed-staffing", ["small"], n_instances=2, config=_tiny_cfg("nsga2"), check_repro=True
    )
    assert all(r.rep == 1.0 for r in rows)


@pytest.mark.slow
def test_run_benchmark_humanitarian_smoke() -> None:
    rows = run_benchmark("humanitarian", ["small"], n_instances=1, config=_tiny_cfg("nsga2"))
    assert len(rows) == 1
    assert rows[0].model == "humanitarian"
    assert rows[0].size == "small"


@pytest.mark.slow
def test_run_benchmark_include_baseline_adds_greedy_row() -> None:
    rows = run_benchmark(
        "ed-staffing", ["small"], n_instances=2, config=_tiny_cfg("nsga2"), include_baseline=True
    )
    solvers = {r.solver for r in rows}
    assert solvers == {"nsga2", "greedy"}
    # Per-instance HV samples are retained for the stats layer.
    for r in rows:
        assert len(r.hv_samples) == 2


@pytest.mark.slow
def test_run_benchmark_include_exact_adds_exact_row() -> None:
    rows = run_benchmark(
        "ed-staffing", ["small"], n_instances=2, config=_tiny_cfg("nsga2"), include_exact=True
    )
    assert {r.solver for r in rows} == {"nsga2", "exact"}


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def test_write_benchmark_summary(tmp_path: Path) -> None:
    rows = [
        BenchmarkRow(
            model="ed-staffing",
            size="small",
            solver="nsga2",
            n_instances=2,
            nns_mean=8.0,
            nns_std=0.0,
            mid_mean=0.5,
            mid_std=0.01,
            sm_mean=0.02,
            sm_std=0.0,
            hv_mean=0.47,
            hv_std=0.01,
            cpu_mean=0.01,
            cpu_std=0.0,
            rep=1.0,
            hv_samples=[0.46, 0.48],
        )
    ]
    csv_path, json_path = write_benchmark_summary(rows, tmp_path)
    assert csv_path.exists() and json_path.exists()

    import json

    import pandas as pd

    df = pd.read_csv(csv_path)
    assert {"model", "size", "solver", "nns_mean", "hv_mean", "rep"} <= set(df.columns)
    # Per-instance sample arrays are stripped from the Table-3 summary.
    assert "hv_samples" not in df.columns
    data = json.loads(json_path.read_text())
    assert data[0]["solver"] == "nsga2"
    assert "hv_samples" not in data[0]
