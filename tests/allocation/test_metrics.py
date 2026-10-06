"""Unit tests for allocation/metrics.py — NNS, MID, SM, HV correctness."""

from __future__ import annotations

import pytest

from presidio_vol_assign.allocation.metrics import compute_allocation_metrics
from presidio_vol_assign.allocation.models import (
    Allocation,
    AllocationParetoFront,
    AllocationSolution,
    AllocationSolverType,
)


def _front(*fitness_tuples, n_obj=4) -> AllocationParetoFront:
    """Build a front from a sequence of fitness tuples."""
    sols = []
    for ft in fitness_tuples:
        if n_obj == 4:
            sols.append(
                AllocationSolution(
                    allocations=[Allocation("p", "c")],
                    objectives_count=4,
                    mn_ulpp=ft[0],
                    mn_trd=ft[1],
                    mn_rpd=ft[2],
                    mn_cail=ft[3],
                )
            )
        else:
            sols.append(
                AllocationSolution(
                    allocations=[Allocation("p", "c")],
                    objectives_count=3,
                    mn_ulpp=ft[0],
                    mn_til=ft[1],
                    mn_cail=ft[2],
                )
            )
    return AllocationParetoFront(
        solver=AllocationSolverType.NSGA2,
        objectives_count=n_obj,
        solutions=sols,
        cpu_time_sec=1.0,
    )


class TestEmptyFront:
    def test_empty_front_metrics(self):
        front = AllocationParetoFront(
            solver=AllocationSolverType.NSGA2, objectives_count=4, solutions=[]
        )
        m = compute_allocation_metrics(front)
        assert m.nns == 0
        assert m.mid == 0.0
        assert m.sm == 0.0
        assert m.hv == 0.0


class TestNNS:
    def test_nns_counts_solutions(self):
        front = _front((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2))
        m = compute_allocation_metrics(front)
        assert m.nns == 3


class TestMID:
    def test_mid_is_mean_distance_to_origin(self):
        # Points at (3,4,0,0) and (0,0,0,5): distances 5 and 5; mean 5
        front = _front((3, 4, 0, 0), (0, 0, 0, 5))
        m = compute_allocation_metrics(front)
        assert m.mid == pytest.approx(5.0)


class TestHV:
    def test_hv_dominated_point_is_full_box_minus_corner(self):
        # ref=(100,100,100,100); single point at (50,50,50,50) dominates
        # the volume from (50,50,50,50) to (100,100,100,100): 50^4 = 6.25e6
        front = _front((50, 50, 50, 50))
        m = compute_allocation_metrics(front)
        assert m.hv == pytest.approx(50 * 50 * 50 * 50)

    def test_hv_drops_solutions_outside_reference(self):
        # Ref-point box is (0..100)^4; a solution at (200,200,200,200) is outside.
        front = _front((50, 50, 50, 50), (200, 200, 200, 200))
        m = compute_allocation_metrics(front)
        # The outside point is dropped; HV equals single-point case
        assert m.hv == pytest.approx(50 * 50 * 50 * 50)

    def test_hv_3obj(self):
        # 3-obj single point at (50, 50, 50): volume 50^3 = 125000
        front = _front((50, 50, 50), n_obj=3)
        m = compute_allocation_metrics(front)
        assert m.hv == pytest.approx(125000.0)

    def test_ref_point_dimension_mismatch_raises(self):
        front = _front((50, 50, 50, 50))
        with pytest.raises(ValueError, match="ref_point dimension"):
            compute_allocation_metrics(front, ref_point=(100.0, 100.0))


class TestSM:
    def test_sm_zero_for_singleton(self):
        front = _front((50, 50, 50, 50))
        m = compute_allocation_metrics(front)
        assert m.sm == 0.0

    def test_sm_positive_for_uneven_spacing(self):
        # Points along a line with varying gaps
        front = _front((0, 0, 0, 0), (1, 0, 0, 0), (10, 0, 0, 0))
        m = compute_allocation_metrics(front)
        # Distances 1 and 9; std-dev / mean of |d-d_bar| / d_bar = (4+4)/(2*5) = 0.8
        assert m.sm > 0.5
