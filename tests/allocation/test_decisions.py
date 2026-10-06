"""Tests for decision extraction and stability metrics (Paper B, RQ1)."""

from __future__ import annotations

import math

import pytest

from presidio_vol_assign.allocation.decisions import (
    canonical_decision,
    canonical_decision_fixed,
    decision_stability,
    pairs_of,
)
from presidio_vol_assign.allocation.models import (
    Allocation,
    AllocationParetoFront,
    AllocationSolution,
    AllocationSolverType,
)
from presidio_vol_assign.allocation.solvers import precompute_fis_cache


def _sol(ulpp, trd, rpd, cail) -> AllocationSolution:
    return AllocationSolution(
        allocations=[Allocation(person_id="P0", center_id="C0")],
        objectives_count=4,
        mn_ulpp=ulpp,
        mn_trd=trd,
        mn_rpd=rpd,
        mn_cail=cail,
    )


def _front(solutions) -> AllocationParetoFront:
    return AllocationParetoFront(
        solver=AllocationSolverType.NSGA2, objectives_count=4, solutions=list(solutions)
    )


def test_canonical_decision_picks_min_normalised_sum() -> None:
    dominant = _sol(0.1, 0.1, 0.1, 0.1)
    front = _front([_sol(0.9, 0.9, 0.9, 0.9), dominant, _sol(0.5, 0.5, 0.5, 0.5)])
    assert canonical_decision(front) is dominant


def test_canonical_decision_empty_front_fails_closed() -> None:
    with pytest.raises(ValueError, match="empty front"):
        canonical_decision(_front([]))


def test_canonical_decision_fixed_picks_min_norm() -> None:
    near = _sol(0.1, 0.1, 0.1, 0.1)
    front = _front([_sol(0.9, 0.9, 0.9, 0.9), near, _sol(0.4, 0.4, 0.4, 0.4)])
    assert canonical_decision_fixed(front) is near


def test_canonical_decision_fixed_empty_fails_closed() -> None:
    with pytest.raises(ValueError, match="empty front"):
        canonical_decision_fixed(_front([]))


def test_pairs_of_maps_ids_to_indices(problem) -> None:
    solution = AllocationSolution(
        allocations=[
            Allocation(person_id="P2", center_id="C1"),
            Allocation(person_id="P5", center_id="C0"),
        ],
        objectives_count=4,
        mn_ulpp=0.0,
        mn_cail=0.0,
    )
    assert pairs_of(solution, problem) == [(2, 1), (5, 0)]


def test_identical_decisions_are_perfectly_stable(problem, base_config) -> None:
    cache = precompute_fis_cache(problem, base_config)
    pairs = [(0, 0), (1, 1), (2, 2), (3, 0)]
    m = decision_stability(pairs, pairs, cache, base_config.objectives, problem.n_centers)
    assert m["objective_drift"] == 0.0
    assert m["quality_loss"] == 0.0
    assert m["allocation_churn"] == 0.0
    assert m["load_rank_stability"] == 1.0


def test_changed_decision_registers_churn_and_drift(problem, base_config) -> None:
    cache = precompute_fis_cache(problem, base_config)
    clean = [(0, 0), (1, 1), (2, 2), (3, 0)]
    perturbed = [(0, 1), (1, 1), (2, 2), (3, 0)]  # person 0 re-routed C0 -> C1
    m = decision_stability(clean, perturbed, cache, base_config.objectives, problem.n_centers)
    assert m["allocation_churn"] == pytest.approx(0.25)
    assert m["objective_drift"] > 0.0
    assert math.isfinite(m["quality_loss"])
