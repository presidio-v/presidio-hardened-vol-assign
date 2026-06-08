"""Unit tests for the 4-obj → 3-obj projection helpers (H1 analysis)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from presidio_vol_assign.allocation.models import (
    AllocationParetoFront,
    AllocationSolution,
    AllocationSolverType,
    Weights,
)
from presidio_vol_assign.allocation.projection import (
    H1Summary,
    project_pareto_4_to_3,
    spearman_trd_rpd,
    summarise_h1,
)
from presidio_vol_assign.allocation.solvers import solve


def _make_4obj_front(*tuples) -> AllocationParetoFront:
    """Build a 4-obj front from raw fitness tuples (no allocations)."""
    sols = [
        AllocationSolution(
            allocations=[],
            objectives_count=4,
            mn_ulpp=t[0],
            mn_trd=t[1],
            mn_rpd=t[2],
            mn_cail=t[3],
        )
        for t in tuples
    ]
    return AllocationParetoFront(
        solver=AllocationSolverType.NSGA2,
        objectives_count=4,
        solutions=sols,
    )


class TestSpearmanTrdRpd:
    def test_perfectly_correlated(self):
        front = _make_4obj_front(
            (10, 10, 10, 50), (20, 20, 20, 50), (30, 30, 30, 50), (40, 40, 40, 50)
        )
        assert spearman_trd_rpd(front) == pytest.approx(1.0)

    def test_anticorrelated(self):
        front = _make_4obj_front(
            (10, 10, 40, 50), (20, 20, 30, 50), (30, 30, 20, 50), (40, 40, 10, 50)
        )
        assert spearman_trd_rpd(front) == pytest.approx(-1.0)

    def test_constant_axis_returns_nan(self):
        # All TRD identical → rank correlation undefined
        front = _make_4obj_front((10, 50, 10, 50), (20, 50, 20, 50), (30, 50, 30, 50))
        assert math.isnan(spearman_trd_rpd(front))

    def test_singleton_returns_nan(self):
        front = _make_4obj_front((10, 20, 30, 40))
        assert math.isnan(spearman_trd_rpd(front))

    def test_3obj_front_rejected(self):
        front = AllocationParetoFront(
            solver=AllocationSolverType.NSGA2,
            objectives_count=3,
            solutions=[],
        )
        with pytest.raises(ValueError, match="4-obj"):
            spearman_trd_rpd(front)


class TestProjectPareto4To3:
    def test_projection_recomputes_til(self, problem, base_config):
        # Solve and project on the fixture problem
        cfg = dataclasses.replace(base_config, objectives=4, pop_size=20, generations=10)
        front = solve(problem, cfg)
        projected = project_pareto_4_to_3(front, problem, Weights())
        assert len(projected) == len(front.solutions)
        # Each projection has 3 components in [0, 100]
        for ps in projected:
            assert len(ps.projected_fitness) == 3
            for v in ps.projected_fitness:
                assert 0.0 <= v <= 100.0

    def test_3obj_front_rejected(self, problem, base_config):
        cfg = dataclasses.replace(base_config, objectives=3, pop_size=15, generations=5)
        front = solve(problem, cfg)
        with pytest.raises(ValueError, match="4-obj"):
            project_pareto_4_to_3(front, problem, Weights())

    def test_dominance_flags_internal(self):
        # A clearly dominated solution should be flagged
        front = _make_4obj_front(
            (10, 10, 10, 10),  # dominates everyone
            (50, 50, 50, 50),
            (90, 90, 90, 90),
        )
        # Use empty travel; mn_til defaults to 0 so projection becomes
        # (mn_ulpp, 0, mn_cail) which preserves the dominance structure
        # along ULPP and CAIL axes.
        from presidio_vol_assign.allocation.models import AllocationProblem

        empty_problem = AllocationProblem(people=[], centers=[], travel={}, n_dir=1)
        # Bypass n_dir validation by constructing directly
        empty_problem.n_dir = 1  # type: ignore[misc]
        projected = project_pareto_4_to_3(front, empty_problem, Weights())
        # First (10,10,10) is non-dominated; second and third are dominated
        assert projected[0].dominated_in_3obj is False
        assert projected[1].dominated_in_3obj is True
        assert projected[2].dominated_in_3obj is True


class TestSummariseH1:
    def test_empty_front(self, problem):
        front = AllocationParetoFront(
            solver=AllocationSolverType.NSGA2,
            objectives_count=4,
            solutions=[],
        )
        s = summarise_h1(front, problem, Weights())
        assert isinstance(s, H1Summary)
        assert s.n_solutions == 0
        assert s.n_dominated_in_3obj == 0
        assert s.fraction_dominated == 0.0
        assert s.confirms_h1 is False
        assert math.isnan(s.spearman_rho)

    def test_h1_confirmation_path(self, problem, base_config):
        cfg = dataclasses.replace(base_config, objectives=4, pop_size=40, generations=30, seed=42)
        front = solve(problem, cfg)
        s = summarise_h1(front, problem, Weights())
        assert s.n_solutions == len(front.solutions)
        assert 0.0 <= s.fraction_dominated <= 1.0
        # confirms_h1 is a boolean derived from thresholds; check it's typed correctly
        assert isinstance(s.confirms_h1, bool)
