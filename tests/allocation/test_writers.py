"""Unit tests for allocation/writers.py — round-trip CSV and metrics JSON."""

from __future__ import annotations

import json

import pytest

from presidio_vol_assign.allocation.models import (
    Allocation,
    AllocationMetrics,
    AllocationParetoFront,
    AllocationSolution,
    AllocationSolverType,
)
from presidio_vol_assign.allocation.writers import (
    load_allocation_pareto_csv,
    write_allocation_csv,
    write_allocation_metrics_json,
    write_allocation_pareto_csv,
)


def _front_4obj() -> AllocationParetoFront:
    sols = [
        AllocationSolution(
            allocations=[
                Allocation("P0", "C0", ulpp=20, trd=10, rpd=15, cail_contrib=30),
                Allocation("P1", "C1", ulpp=40, trd=50, rpd=55, cail_contrib=20),
            ],
            objectives_count=4,
            mn_ulpp=30,
            mn_trd=30,
            mn_rpd=35,
            mn_cail=25,
        ),
        AllocationSolution(
            allocations=[
                Allocation("P2", "C0", ulpp=50, trd=10, rpd=15, cail_contrib=30),
                Allocation("P3", "C1", ulpp=10, trd=50, rpd=55, cail_contrib=20),
            ],
            objectives_count=4,
            mn_ulpp=30,
            mn_trd=30,
            mn_rpd=35,
            mn_cail=25,
        ),
    ]
    return AllocationParetoFront(
        solver=AllocationSolverType.NSGA3,
        objectives_count=4,
        solutions=sols,
        cpu_time_sec=2.5,
    )


def _front_3obj() -> AllocationParetoFront:
    sols = [
        AllocationSolution(
            allocations=[Allocation("P0", "C0", ulpp=20, til=25, cail_contrib=30)],
            objectives_count=3,
            mn_ulpp=20,
            mn_til=25,
            mn_cail=30,
        )
    ]
    return AllocationParetoFront(
        solver=AllocationSolverType.NSGA2,
        objectives_count=3,
        solutions=sols,
        cpu_time_sec=1.0,
    )


class TestPareto4ObjRoundTrip:
    def test_write_and_load_4obj(self, tmp_path):
        front = _front_4obj()
        path = write_allocation_pareto_csv(front, tmp_path)
        loaded = load_allocation_pareto_csv(path)
        assert loaded.solver == AllocationSolverType.NSGA3
        assert loaded.objectives_count == 4
        assert len(loaded.solutions) == 2
        assert loaded.solutions[0].mn_trd == 30
        assert loaded.solutions[0].mn_rpd == 35


class TestPareto3ObjRoundTrip:
    def test_write_and_load_3obj(self, tmp_path):
        front = _front_3obj()
        path = write_allocation_pareto_csv(front, tmp_path)
        loaded = load_allocation_pareto_csv(path)
        assert loaded.objectives_count == 3
        assert loaded.solutions[0].mn_til == 25


class TestAllocationsCSV:
    def test_write_allocations(self, tmp_path):
        front = _front_4obj()
        path = write_allocation_csv(front, tmp_path)
        assert path.exists()
        # 2 solutions × 2 allocations each + header
        lines = path.read_text().splitlines()
        assert len(lines) == 5


class TestMetricsJSON:
    def test_write_metrics(self, tmp_path):
        m = AllocationMetrics(
            solver=AllocationSolverType.NSGA3,
            objectives_count=4,
            nns=10,
            mid=42.5,
            sm=0.123,
            hv=2_500_000.0,
            cpu_time_sec=3.7,
        )
        path = write_allocation_metrics_json(m, tmp_path)
        data = json.loads(path.read_text())
        assert data["solver"] == "nsga3"
        assert data["objectives_count"] == 4
        assert data["nns"] == 10
        assert data["mid"] == 42.5


class TestLoaderErrors:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_allocation_pareto_csv(tmp_path / "nope.csv")
