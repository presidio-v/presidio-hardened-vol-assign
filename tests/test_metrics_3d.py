"""Tests for the N-dimensional generalisation of the metrics module (v0.2.0).

Exercises 3-objective fronts (the humanitarian model's objective space) and the
reproducibility primitives.
"""

from __future__ import annotations

import math

import pytest

from presidio_vol_assign.metrics import (
    _hv,
    _mid,
    _nns,
    _sm,
    compute_metrics,
    front_signature,
    fronts_signature,
    reproducibility_score,
)
from presidio_vol_assign.models import (
    Assignment,
    ParetoFront,
    SkillType,
    Solution,
    SolverType,
)


def _sol3(z1: float, z2: float, z3: float) -> Solution:
    a = Assignment(volunteer_id="P1", ed_id="C1", vacancy_type=SkillType.TRIAGE)
    return Solution(assignments=[a], objectives=(z1, z2, z3))


# ---------------------------------------------------------------------------
# Solution objective vector + backward-compatible views
# ---------------------------------------------------------------------------


def test_solution_objectives_three() -> None:
    s = _sol3(0.1, 0.2, 0.3)
    assert s.objectives == (0.1, 0.2, 0.3)
    assert s.n_objectives == 3
    # z1/z2 remain views on the first two objectives
    assert s.z1 == 0.1
    assert s.z2 == 0.2


def test_solution_two_objective_path_sets_objectives() -> None:
    a = Assignment(volunteer_id="V1", ed_id="ED1", vacancy_type=SkillType.TRIAGE)
    s = Solution(assignments=[a], z1=0.4, z2=0.6)
    assert s.objectives == (0.4, 0.6)
    assert s.n_objectives == 2


# ---------------------------------------------------------------------------
# MID / SM / HV in 3-D
# ---------------------------------------------------------------------------


def test_nns_3d() -> None:
    assert _nns([_sol3(0.2, 0.2, 0.2), _sol3(0.3, 0.1, 0.5)]) == 2


def test_mid_3d_single() -> None:
    # distance from (0.2, 0.2, 0.2) to origin = sqrt(0.12)
    assert _mid([_sol3(0.2, 0.2, 0.2)]) == pytest.approx(math.sqrt(0.12))


def test_hv_3d_single_point() -> None:
    # (0.2,0.2,0.2) vs ref (1,1,1) → (0.8)^3 = 0.512
    assert _hv([_sol3(0.2, 0.2, 0.2)]) == pytest.approx(0.512)


def test_hv_3d_point_outside_reference_ignored() -> None:
    assert _hv([_sol3(1.2, 0.5, 0.5)]) == pytest.approx(0.0)


def test_sm_3d_single_is_zero() -> None:
    assert _sm([_sol3(0.2, 0.2, 0.2)]) == 0.0


def test_sm_3d_uneven_greater_than_even() -> None:
    tight = _sm([_sol3(0.1, 0.1, 0.8), _sol3(0.11, 0.11, 0.79), _sol3(0.8, 0.8, 0.1)])
    even = _sm([_sol3(0.1, 0.1, 0.8), _sol3(0.45, 0.45, 0.45), _sol3(0.8, 0.8, 0.1)])
    assert tight > even


def test_compute_metrics_3d_front() -> None:
    front = ParetoFront(
        solver=SolverType.NSGA2,
        solutions=[_sol3(0.2, 0.2, 0.2), _sol3(0.3, 0.1, 0.5), _sol3(0.1, 0.4, 0.3)],
        cpu_time_sec=2.0,
    )
    m = compute_metrics(front)
    assert m.nns == 3
    assert m.mid > 0
    assert 0.0 < m.hv < 1.0
    assert m.sm >= 0.0


# ---------------------------------------------------------------------------
# Reproducibility primitives
# ---------------------------------------------------------------------------


def test_front_signature_order_invariant() -> None:
    f1 = ParetoFront(SolverType.NSGA2, [_sol3(0.2, 0.2, 0.2), _sol3(0.3, 0.1, 0.5)], 1.0)
    f2 = ParetoFront(SolverType.NSGA2, [_sol3(0.3, 0.1, 0.5), _sol3(0.2, 0.2, 0.2)], 9.0)
    # Same points in different order (and different cpu time) → same signature
    assert front_signature(f1) == front_signature(f2)


def test_front_signature_sensitive_to_values() -> None:
    f1 = ParetoFront(SolverType.NSGA2, [_sol3(0.2, 0.2, 0.2)], 1.0)
    f2 = ParetoFront(SolverType.NSGA2, [_sol3(0.2, 0.2, 0.2000001)], 1.0)
    assert front_signature(f1) != front_signature(f2)


def test_reproducibility_score() -> None:
    assert reproducibility_score(["a", "a", "a"]) == 1.0
    assert reproducibility_score(["a", "b"]) == 0.0
    assert reproducibility_score([]) == 0.0


def test_fronts_signature_combines_solvers() -> None:
    f_nsga = ParetoFront(SolverType.NSGA2, [_sol3(0.2, 0.2, 0.2)], 1.0)
    f_nrga = ParetoFront(SolverType.NRGA, [_sol3(0.3, 0.1, 0.5)], 1.0)
    sig_a = fronts_signature([f_nsga, f_nrga])
    sig_b = fronts_signature([f_nsga, f_nrga])
    assert sig_a == sig_b
    # Solver order matters in the combined signature
    assert fronts_signature([f_nrga, f_nsga]) != sig_a
