"""4-objective → 3-objective projection and H1 analysis helpers.

H1 of the MDPI extended paper claims that the 4-objective formulation
exposes Pareto trade-offs that the 3-objective ATRes formulation cannot
represent. To test it, we need two derived quantities for each 4-obj
Pareto front:

    1. The Spearman rank correlation between TRD and RPD across the
       front. If |ρ| ≥ 0.9 the two axes carry redundant information and
       the split adds nothing.

    2. The 3-objective projection: for every allocation in every 4-obj
       solution, recompute TIL via the original ATRes FIS2_TIL pathway
       (compute_rws → evaluate_fis2_til) and average to get Mn_TIL. Then
       check Pareto dominance among those projected solutions; any 4-obj
       solution whose 3-obj projection is dominated is one the 3-obj
       formulation could not have surfaced.

H1 is confirmed when ≥20% of 4-obj solutions are non-recoverable in
3-obj space *and* |Spearman ρ| < 0.5 — the operationalisation in
`hypothesis-rq.md`.

Public API:
    project_pareto_4_to_3(front, problem, weights) -> list[ProjectedSolution]
    spearman_trd_rpd(front) -> float
    summarise_h1(front, problem, weights) -> H1Summary
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import spearmanr

from presidio_vol_assign.allocation.fis import compute_rws, evaluate_fis2_til
from presidio_vol_assign.allocation.models import (
    AllocationParetoFront,
    AllocationProblem,
    AllocationSolution,
    Weights,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectedSolution:
    """A 4-obj solution alongside its 3-obj projection and dominance flag.

    Attributes:
        source: The original 4-obj `AllocationSolution`.
        projected_fitness: (mn_ulpp, mn_til, mn_cail) recomputed via the
            ATRes 3-obj pathway.
        dominated_in_3obj: True iff at least one other projected solution
            in the same set strictly dominates this one in 3-obj space.
    """

    source: AllocationSolution
    projected_fitness: tuple[float, float, float]
    dominated_in_3obj: bool


@dataclass(frozen=True)
class H1Summary:
    """Aggregate statistics for the H1 hypothesis test on a single front.

    Attributes:
        n_solutions: Number of solutions in the source 4-obj front.
        spearman_rho: Spearman rank correlation between TRD and RPD across
            the 4-obj front (NaN if fewer than 2 distinct points).
        n_dominated_in_3obj: Count of projected solutions that another
            projection strictly dominates.
        fraction_dominated: n_dominated_in_3obj / n_solutions.
        confirms_h1: True iff fraction_dominated >= 0.20 AND
            |spearman_rho| < 0.5 (operationalisation per
            `hypothesis-rq.md`).
    """

    n_solutions: int
    spearman_rho: float
    n_dominated_in_3obj: int
    fraction_dominated: float
    confirms_h1: bool


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def project_pareto_4_to_3(
    front: AllocationParetoFront,
    problem: AllocationProblem,
    weights: Weights,
) -> list[ProjectedSolution]:
    """Recompute each 4-obj solution's TIL via ATRes FIS2_TIL and tag dominance.

    For each allocation (person → center) in each solution, the ATRes 3-obj
    pathway computes:

        RWS_{j,i} = compute_rws(travel, weights)        # sign-corrected Eq. 5
        TIL_{j,i} = evaluate_fis2_til(TD_{j,i}, RWS_{j,i})

    Mn_TIL is the mean over all allocations in that solution. Mn_ULPP and
    Mn_CAIL are read directly from the solution.

    Dominance is then evaluated within the projected set: solution P_i is
    dominated iff some P_j has all three objectives ≤ P_i's with at least
    one strictly less.

    Raises:
        ValueError: If the front's `objectives_count` is not 4. The
            projection is only defined for 4-obj fronts.
    """
    if front.objectives_count != 4:
        raise ValueError(
            f"project_pareto_4_to_3 requires a 4-obj front; got {front.objectives_count}"
        )

    travel = problem.travel
    projected: list[tuple[float, float, float]] = []
    for sol in front.solutions:
        til_values: list[float] = []
        for alloc in sol.allocations:
            t = travel.get((alloc.person_id, alloc.center_id))
            if t is None:
                # Loaded fronts (from CSV) carry empty allocations; skip
                # silently and let the caller decide if that is acceptable.
                continue
            rws = compute_rws(t, weights)
            til_values.append(evaluate_fis2_til(t.travel_duration, rws))
        mn_til = float(sum(til_values) / len(til_values)) if til_values else 0.0
        projected.append((sol.mn_ulpp, mn_til, sol.mn_cail))

    dominated_flags = _dominance_flags(projected)
    return [
        ProjectedSolution(
            source=front.solutions[i],
            projected_fitness=projected[i],
            dominated_in_3obj=dominated_flags[i],
        )
        for i in range(len(front.solutions))
    ]


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------


def spearman_trd_rpd(front: AllocationParetoFront) -> float:
    """Spearman rank correlation between Mn_TRD and Mn_RPD across the front.

    Returns NaN for fronts with fewer than 2 distinct points or when the
    rank statistic is undefined (e.g. all TRD or all RPD identical).
    """
    if front.objectives_count != 4:
        raise ValueError(f"spearman_trd_rpd requires a 4-obj front; got {front.objectives_count}")
    if len(front.solutions) < 2:
        return float("nan")
    trd = [s.mn_trd for s in front.solutions]
    rpd = [s.mn_rpd for s in front.solutions]
    if len(set(trd)) < 2 or len(set(rpd)) < 2:
        return float("nan")
    rho, _ = spearmanr(trd, rpd)
    return float(rho)


# ---------------------------------------------------------------------------
# H1 summary
# ---------------------------------------------------------------------------


def summarise_h1(
    front: AllocationParetoFront,
    problem: AllocationProblem,
    weights: Weights,
    fraction_threshold: float = 0.20,
    rho_threshold: float = 0.5,
) -> H1Summary:
    """Combine projection and Spearman analyses into a single H1 verdict.

    H1 is confirmed when both conditions hold:
        - fraction of 4-obj solutions whose 3-obj projection is dominated
          by another projection ≥ `fraction_threshold` (default 0.20), AND
        - |Spearman ρ(TRD, RPD)| < `rho_threshold` (default 0.5).

    The two thresholds are deliberately separate: high dominance fraction
    alone could come from a few outlier solutions; low |ρ| alone could
    come from noise. Confirming both gives a stronger empirical signal.
    """
    n = len(front.solutions)
    if n == 0:
        return H1Summary(0, float("nan"), 0, 0.0, False)

    rho = spearman_trd_rpd(front)
    projected = project_pareto_4_to_3(front, problem, weights)
    n_dom = sum(1 for ps in projected if ps.dominated_in_3obj)
    frac = n_dom / n
    confirms = (frac >= fraction_threshold) and (not math.isnan(rho) and abs(rho) < rho_threshold)
    return H1Summary(
        n_solutions=n,
        spearman_rho=rho,
        n_dominated_in_3obj=n_dom,
        fraction_dominated=frac,
        confirms_h1=confirms,
    )


# ---------------------------------------------------------------------------
# Internal helper — Pareto dominance flags
# ---------------------------------------------------------------------------


def _dominance_flags(points: list[tuple[float, float, float]]) -> list[bool]:
    """Return a list of bools: True at index i iff some other point dominates points[i].

    A point a dominates b iff every coordinate a_k <= b_k and at least one
    a_k < b_k. Naive O(n^2) check; n is small for our Pareto fronts so the
    constant overhead matters more than asymptotic cleverness.
    """
    n = len(points)
    flags = [False] * n
    for i in range(n):
        ai = points[i]
        for j in range(n):
            if i == j:
                continue
            aj = points[j]
            le = all(aj[k] <= ai[k] for k in range(3))
            lt = any(aj[k] < ai[k] for k in range(3))
            if le and lt:
                flags[i] = True
                break
    return flags
