"""Pareto-front quality metrics for the allocation module.

Implements ATRes Eqs. (18)–(20) plus NNS:

    NNS  Number of Non-dominated Solutions          (count)
    MID  Mean Ideal Distance (Eq. 18)               (lower = closer to ideal)
    SM   Spacing Metric (Eq. 19)                    (lower = more uniform)
    HV   Hypervolume (Eq. 20)                       (higher = better coverage)

Hypervolume is computed via pymoo's WFG-based indicator for any
dimensionality ≥ 2. The legacy 2D sweep-line implementation in
`presidio_vol_assign.metrics` is left intact for the volunteer-assignment
side of the package; allocation never falls back to it.

Reference point default: (100, 100, ..., 100). All FIS outputs in this
model live in [0, 100], so this anchor dominates every feasible solution.
Callers can pass a tighter reference point (e.g. computed from the union
of all algorithms' fronts on the same instance, per ATRes §5) when they
want HV values comparable across runs.

Public API:
    compute_allocation_metrics(front, ref_point=None) -> AllocationMetrics
"""

from __future__ import annotations

import math

import numpy as np
from pymoo.indicators.hv import HV

from presidio_vol_assign.allocation.models import (
    AllocationMetrics,
    AllocationParetoFront,
    AllocationSolution,
)


def compute_allocation_metrics(
    front: AllocationParetoFront,
    ref_point: tuple[float, ...] | None = None,
) -> AllocationMetrics:
    """Compute NNS, MID, SM, and HV for an `AllocationParetoFront`."""
    n_obj = front.objectives_count
    if ref_point is None:
        ref_point = tuple(100.0 for _ in range(n_obj))
    if len(ref_point) != n_obj:
        raise ValueError(
            f"ref_point dimension {len(ref_point)} does not match objectives_count {n_obj}"
        )

    return AllocationMetrics(
        solver=front.solver,
        objectives_count=n_obj,
        nns=_nns(front.solutions),
        mid=_mid(front.solutions),
        sm=_sm(front.solutions),
        hv=_hv(front.solutions, ref_point),
        cpu_time_sec=front.cpu_time_sec,
    )


# ---------------------------------------------------------------------------
# Individual metrics — exported for tests
# ---------------------------------------------------------------------------


def _nns(solutions: list[AllocationSolution]) -> int:
    return len(solutions)


def _mid(solutions: list[AllocationSolution]) -> float:
    """Mean Ideal Distance — mean Euclidean distance from each solution to
    the origin (0, 0, ...) per ATRes Eq. (18).

    Returns 0.0 for empty fronts.
    """
    if not solutions:
        return 0.0
    distances = [math.sqrt(sum(v * v for v in s.fitness)) for s in solutions]
    return float(np.mean(distances))


def _sm(solutions: list[AllocationSolution]) -> float:
    """Spacing Metric per ATRes Eq. (19): coefficient-of-variation form.

        SM = sum_i |d_i - d_bar| / ((n - 1) · d_bar)

    where d_i is the Euclidean distance between consecutive solutions on
    the front (sorted by first objective; ties broken by subsequent
    objectives), and d_bar is their mean. Lower = more uniform spacing.

    Returns 0.0 for fronts with fewer than two solutions, or when d_bar=0
    (all solutions coincide in objective space).
    """
    if len(solutions) < 2:
        return 0.0
    sorted_sols = sorted(solutions, key=lambda s: s.fitness)
    dists = []
    for i in range(1, len(sorted_sols)):
        a = np.asarray(sorted_sols[i].fitness, dtype=float)
        b = np.asarray(sorted_sols[i - 1].fitness, dtype=float)
        dists.append(float(np.linalg.norm(a - b)))

    d_bar = float(np.mean(dists))
    if d_bar == 0.0:
        return 0.0
    abs_dev = [abs(d - d_bar) for d in dists]
    return float(sum(abs_dev) / ((len(dists)) * d_bar))


def _hv(
    solutions: list[AllocationSolution],
    ref_point: tuple[float, ...],
) -> float:
    """Hypervolume per ATRes Eq. (20), via pymoo's WFG indicator.

    Solutions outside the reference box (any objective ≥ corresponding
    ref-point coordinate) are dropped before computation; pymoo would
    otherwise compute HV against a non-dominating reference, returning 0.

    Returns 0.0 for empty fronts or fronts wholly outside the reference.
    """
    if not solutions:
        return 0.0
    ref = np.asarray(ref_point, dtype=float)
    pts = np.asarray([list(s.fitness) for s in solutions], dtype=float)
    inside = pts < ref  # element-wise
    keep = inside.all(axis=1)
    pts = pts[keep]
    if pts.size == 0:
        return 0.0
    indicator = HV(ref_point=ref)
    return float(indicator(pts))
