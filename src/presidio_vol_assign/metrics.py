"""Pareto-front quality metrics, generalised to any number of objectives.

All functions are pure (no side effects) and read each solution's canonical
``objectives`` vector, so they work for both the 2-objective ED-staffing model
and the 3-objective humanitarian model.

Metric definitions:
    NNS — Number of Non-dominated Solutions (front size).
    MID — Mean Ideal Distance: mean Euclidean distance from each solution to the
          ideal point (default = origin).
    SM  — Spacing Metric (Schott): standard deviation of each solution's
          nearest-neighbour distance to the rest of the front. Dimension-
          agnostic; lower = more uniform spread. (This replaces the v0.1.0
          sort-by-z1 consecutive-distance definition, which only worked in 2-D.)
    HV  — Hypervolume dominated relative to a reference point (default = all
          ones), computed with DEAP's n-dimensional implementation (pymoo
          fallback when DEAP's compiled module is unavailable).
    REP — Reproducibility: 1.0 when repeated seeded runs yield bit-for-bit
          identical fronts (see ``reproducibility_score`` / ``front_signature``).

Public API:
    compute_metrics(front) -> Metrics
    front_signature(front) -> str
    reproducibility_score(signatures) -> float
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np

try:  # deap's compiled n-d hypervolume; some deap wheels omit this module
    from deap.tools._hypervolume import hv as _deap_hv

    def _hypervolume(pts: np.ndarray, ref: np.ndarray) -> float:
        return float(_deap_hv.hypervolume(pts, ref))

except ImportError:  # fall back to pymoo (already a dependency) so fresh
    from pymoo.indicators.hv import HV as _PymooHV  # installs stay reproducible

    def _hypervolume(pts: np.ndarray, ref: np.ndarray) -> float:
        return float(_PymooHV(ref_point=ref)(pts))


from presidio_vol_assign.models import Metrics, ParetoFront, Solution


def compute_metrics(
    front: ParetoFront,
    *,
    ideal_point: tuple[float, ...] | None = None,
    reference_point: tuple[float, ...] | None = None,
) -> Metrics:
    """Compute NNS, MID, SM, and HV for a ParetoFront.

    ``ideal_point`` / ``reference_point`` default to the origin and the all-ones
    point of the front's objective dimension respectively (both objective
    families normalise objectives to [0, 1]).
    """
    sols = front.solutions
    return Metrics(
        solver=front.solver,
        nns=_nns(sols),
        mid=_mid(sols, ideal=ideal_point),
        sm=_sm(sols),
        hv=_hv(sols, ref=reference_point),
        cpu_time_sec=front.cpu_time_sec,
    )


# ---------------------------------------------------------------------------
# Individual metric functions (also exported for testing)
# ---------------------------------------------------------------------------


def _nns(solutions: list[Solution]) -> int:
    """Number of Non-dominated Solutions."""
    return len(solutions)


def _mid(solutions: list[Solution], ideal: tuple[float, ...] | None = None) -> float:
    """Mean Ideal Distance — mean Euclidean distance to the ideal point.

    Defaults to the origin of the appropriate dimension.
    """
    if not solutions:
        return 0.0
    vectors = [s.objectives for s in solutions]
    if ideal is None:
        ideal = (0.0,) * len(vectors[0])
    distances = [math.dist(v, ideal) for v in vectors]
    return float(np.mean(distances))


def _sm(solutions: list[Solution]) -> float:
    """Schott spacing metric — std-dev of nearest-neighbour distances.

    For each solution, take the Euclidean distance to its closest neighbour on
    the front; SM is the standard deviation of those distances. Works in any
    dimension. Returns 0.0 for fronts with fewer than two solutions.
    """
    if len(solutions) < 2:
        return 0.0
    vectors = [s.objectives for s in solutions]
    nearest: list[float] = []
    for i, vi in enumerate(vectors):
        nn = min(math.dist(vi, vj) for j, vj in enumerate(vectors) if j != i)
        nearest.append(nn)
    return float(np.std(nearest))


def _hv(solutions: list[Solution], ref: tuple[float, ...] | None = None) -> float:
    """Hypervolume dominated by the front relative to reference point *ref*.

    Uses DEAP's n-dimensional hypervolume, or the pymoo fallback when DEAP's
    compiled module is unavailable. ``ref`` defaults to the all-ones
    point (both objective families have objectives in [0, 1]). Only solutions
    strictly inside the reference box contribute; returns 0.0 for empty fronts
    or fronts that do not dominate the reference point.
    """
    if not solutions:
        return 0.0
    vectors = [s.objectives for s in solutions]
    dim = len(vectors[0])
    if ref is None:
        ref = (1.0,) * dim
    pts = [v for v in vectors if all(v[k] < ref[k] for k in range(dim))]
    if not pts:
        return 0.0
    return _hypervolume(np.array(pts, dtype=float), np.array(ref, dtype=float))


# ---------------------------------------------------------------------------
# Reproducibility (REP)
# ---------------------------------------------------------------------------


def front_signature(front: ParetoFront, precision: int = 9) -> str:
    """Return a stable SHA-256 signature of a front's objective vectors.

    Objective vectors are rounded to *precision* decimals and sorted, so the
    signature is invariant to solution ordering but sensitive to any numeric
    drift — the basis for the bit-for-bit reproducibility check.
    """
    rounded = sorted(tuple(round(x, precision) for x in s.objectives) for s in front.solutions)
    payload = repr(rounded).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fronts_signature(fronts: Sequence[ParetoFront], precision: int = 9) -> str:
    """Combined signature over an ordered collection of fronts (e.g. both solvers)."""
    combined = "|".join(f"{f.solver.value}:{front_signature(f, precision)}" for f in fronts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def reproducibility_score(signatures: Sequence[str]) -> float:
    """1.0 if all run signatures are identical (bit-for-bit), else 0.0."""
    if not signatures:
        return 0.0
    first = signatures[0]
    return 1.0 if all(s == first for s in signatures) else 0.0
