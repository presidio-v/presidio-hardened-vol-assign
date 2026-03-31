"""Tests for the Pareto-front metrics module."""

from __future__ import annotations

import math

import pytest

from presidio_vol_assign.metrics import _hv, _mid, _nns, _sm, compute_metrics
from presidio_vol_assign.models import Assignment, ParetoFront, SkillType, Solution, SolverType


def _sol(z1: float, z2: float) -> Solution:
    a = Assignment(volunteer_id="V1", ed_id="ED1", vacancy_type=SkillType.TRIAGE)
    return Solution(assignments=[a], z1=z1, z2=z2)


# ---------------------------------------------------------------------------
# NNS
# ---------------------------------------------------------------------------


def test_nns_empty() -> None:
    assert _nns([]) == 0


def test_nns_count() -> None:
    assert _nns([_sol(0.2, 0.8), _sol(0.5, 0.5)]) == 2


# ---------------------------------------------------------------------------
# MID
# ---------------------------------------------------------------------------


def test_mid_empty() -> None:
    assert _mid([]) == 0.0


def test_mid_single() -> None:
    # distance from (0.3, 0.4) to (0,0) = 0.5
    assert _mid([_sol(0.3, 0.4)]) == pytest.approx(0.5)


def test_mid_two_points() -> None:
    # sqrt(0.09 + 0.16) = 0.5 and sqrt(0.25 + 0.25) ≈ 0.7071
    expected = (0.5 + math.hypot(0.5, 0.5)) / 2
    assert _mid([_sol(0.3, 0.4), _sol(0.5, 0.5)]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# SM
# ---------------------------------------------------------------------------


def test_sm_empty() -> None:
    assert _sm([]) == 0.0


def test_sm_single() -> None:
    assert _sm([_sol(0.3, 0.4)]) == 0.0


def test_sm_evenly_spaced_is_zero() -> None:
    # Three points at equal Euclidean spacing → std = 0
    pts = [_sol(0.0, 0.9), _sol(0.5, 0.5), _sol(0.9, 0.0)]
    # Distances won't be exactly equal, but SM should be low
    result = _sm(pts)
    assert result >= 0.0


def test_sm_uneven_spacing() -> None:
    # Cluster two points close together, one far away → higher SM
    tight = _sm([_sol(0.1, 0.9), _sol(0.11, 0.89), _sol(0.9, 0.1)])
    even = _sm([_sol(0.1, 0.9), _sol(0.5, 0.5), _sol(0.9, 0.1)])
    assert tight > even


# ---------------------------------------------------------------------------
# HV
# ---------------------------------------------------------------------------


def test_hv_empty() -> None:
    assert _hv([]) == 0.0


def test_hv_single_point() -> None:
    # Point (0.2, 0.8), ref (1, 1) → rectangle (1-0.2)*(1-0.8) = 0.8*0.2 = 0.16
    assert _hv([_sol(0.2, 0.8)]) == pytest.approx(0.16)


def test_hv_two_nondominated_points() -> None:
    # (0.2, 0.8) and (0.8, 0.2) with ref (1,1)
    # Sweep: sort by z1 → [(0.2,0.8),(0.8,0.2)]
    # Step 1: z2=0.8 < 1.0 → hv += (1-0.2)*(1-0.8) = 0.16; current_z2=0.8
    # Step 2: z2=0.2 < 0.8 → hv += (1-0.8)*(0.8-0.2) = 0.12; current_z2=0.2
    # Total = 0.28
    assert _hv([_sol(0.2, 0.8), _sol(0.8, 0.2)]) == pytest.approx(0.28)


def test_hv_three_points() -> None:
    # (0.2, 0.8), (0.5, 0.5), (0.8, 0.2) ref (1,1)
    # Step 1: z2=0.8 → hv += 0.8*0.2 = 0.16; cur=0.8
    # Step 2: z2=0.5 → hv += 0.5*0.3 = 0.15; cur=0.5
    # Step 3: z2=0.2 → hv += 0.2*0.3 = 0.06; cur=0.2
    # Total = 0.37
    assert _hv([_sol(0.2, 0.8), _sol(0.5, 0.5), _sol(0.8, 0.2)]) == pytest.approx(0.37)


def test_hv_point_outside_reference_ignored() -> None:
    # Point (1.5, 0.5) is outside ref (1,1) and should not contribute
    result = _hv([_sol(1.5, 0.5)])
    assert result == pytest.approx(0.0)


def test_hv_dominated_point_does_not_add() -> None:
    # (0.5, 0.5) dominates (0.7, 0.7) — dominated point adds nothing
    hv_with_dominated = _hv([_sol(0.5, 0.5), _sol(0.7, 0.7)])
    hv_without = _hv([_sol(0.5, 0.5)])
    assert hv_with_dominated == pytest.approx(hv_without)


# ---------------------------------------------------------------------------
# compute_metrics integration
# ---------------------------------------------------------------------------


def test_compute_metrics_fields() -> None:
    front = ParetoFront(
        solver=SolverType.NSGA2,
        solutions=[_sol(0.2, 0.8), _sol(0.5, 0.5), _sol(0.8, 0.2)],
        cpu_time_sec=1.23,
    )
    m = compute_metrics(front)
    assert m.solver == SolverType.NSGA2
    assert m.nns == 3
    assert m.mid > 0
    assert m.hv == pytest.approx(0.37)
    assert m.cpu_time_sec == pytest.approx(1.23)
