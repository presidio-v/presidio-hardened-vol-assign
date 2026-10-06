"""Tests for allocation-front reproducibility signatures (Paper B, RQ2)."""

from __future__ import annotations

from presidio_vol_assign.allocation.models import (
    Allocation,
    AllocationParetoFront,
    AllocationSolution,
    AllocationSolverType,
)
from presidio_vol_assign.allocation.repro import (
    allocation_front_signature,
    rep_score,
)


def _solution(ulpp: float, trd: float, rpd: float, cail: float) -> AllocationSolution:
    allocs = [
        Allocation(person_id="P0", center_id="C1"),
        Allocation(person_id="P2", center_id="C0"),
    ]
    return AllocationSolution(
        allocations=allocs,
        objectives_count=4,
        mn_ulpp=ulpp,
        mn_trd=trd,
        mn_rpd=rpd,
        mn_cail=cail,
    )


def _front(solutions) -> AllocationParetoFront:
    return AllocationParetoFront(
        solver=AllocationSolverType.NSGA2,
        objectives_count=4,
        solutions=list(solutions),
    )


def test_signature_is_deterministic() -> None:
    front = _front([_solution(0.1, 0.2, 0.3, 0.4)])
    assert allocation_front_signature(front) == allocation_front_signature(front)


def test_signature_is_order_invariant_over_solutions() -> None:
    a = _solution(0.1, 0.2, 0.3, 0.4)
    b = _solution(0.5, 0.6, 0.7, 0.8)
    assert allocation_front_signature(_front([a, b])) == allocation_front_signature(_front([b, a]))


def test_signature_changes_when_an_objective_changes() -> None:
    base = allocation_front_signature(_front([_solution(0.1, 0.2, 0.3, 0.4)]))
    drift = allocation_front_signature(_front([_solution(0.1, 0.2, 0.3, 0.400001)]))
    assert base != drift


def test_signature_changes_when_an_assignment_changes() -> None:
    front = _front([_solution(0.1, 0.2, 0.3, 0.4)])
    base = allocation_front_signature(front)
    front.solutions[0].allocations[0].center_id = "C2"
    assert allocation_front_signature(front) != base


def test_rep_score_all_identical_is_one() -> None:
    sig = allocation_front_signature(_front([_solution(0.1, 0.2, 0.3, 0.4)]))
    assert rep_score([sig, sig, sig]) == 1.0


def test_rep_score_any_divergence_is_zero() -> None:
    assert rep_score(["a", "a", "b"]) == 0.0


def test_rep_score_empty_fails_closed() -> None:
    assert rep_score([]) == 0.0
