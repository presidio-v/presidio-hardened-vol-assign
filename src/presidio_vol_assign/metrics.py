"""Pareto-front quality metrics matching paper Table 3.

All functions are pure (no side effects).

Public API:
    compute_metrics(front: ParetoFront) -> Metrics
"""

from __future__ import annotations

import math

import numpy as np

from presidio_vol_assign.models import Metrics, ParetoFront, Solution


def compute_metrics(front: ParetoFront) -> Metrics:
    """Compute NNS, MID, SM, and HV for a ParetoFront."""
    return Metrics(
        solver=front.solver,
        nns=_nns(front.solutions),
        mid=_mid(front.solutions),
        sm=_sm(front.solutions),
        hv=_hv(front.solutions),
        cpu_time_sec=front.cpu_time_sec,
    )


# ---------------------------------------------------------------------------
# Individual metric functions (also exported for testing)
# ---------------------------------------------------------------------------


def _nns(solutions: list[Solution]) -> int:
    """Number of Non-dominated Solutions."""
    return len(solutions)


def _mid(solutions: list[Solution]) -> float:
    """Mean Ideal Distance — mean Euclidean distance from each solution to (0, 0)."""
    if not solutions:
        return 0.0
    distances = [math.hypot(s.z1, s.z2) for s in solutions]
    return float(np.mean(distances))


def _sm(solutions: list[Solution]) -> float:
    """Spacing Metric — std-dev of consecutive distances on the sorted front.

    Solutions are sorted by z1 before computing adjacent distances.
    Returns 0.0 for fronts with fewer than two solutions.
    """
    if len(solutions) < 2:
        return 0.0
    sorted_sols = sorted(solutions, key=lambda s: s.z1)
    dists = [
        math.hypot(
            sorted_sols[i].z1 - sorted_sols[i - 1].z1, sorted_sols[i].z2 - sorted_sols[i - 1].z2
        )
        for i in range(1, len(sorted_sols))
    ]
    return float(np.std(dists))


def _hv(solutions: list[Solution], ref: tuple[float, float] = (1.0, 1.0)) -> float:
    """2-D Hypervolume dominated by the front relative to reference point *ref*.

    Uses a sweep-line algorithm (O(n log n)).
    Reference point default (1, 1) assumes both objectives are in [0, 1].
    Returns 0.0 for empty fronts or fronts that do not dominate the reference.
    """
    if not solutions:
        return 0.0

    # Keep only solutions that are within the reference box
    pts = [(s.z1, s.z2) for s in solutions if s.z1 < ref[0] and s.z2 < ref[1]]
    if not pts:
        return 0.0

    # Sort by z1 ascending; break ties by z2 ascending
    pts.sort(key=lambda p: (p[0], p[1]))

    hv = 0.0
    current_z2 = ref[1]

    for z1, z2 in pts:
        if z2 < current_z2:
            hv += (ref[0] - z1) * (current_z2 - z2)
            current_z2 = z2

    return float(hv)
