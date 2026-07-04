"""Decision extraction and stability metrics for the turbulence study (Paper B, RQ1).

A multi-objective solver returns a *front*; a deployed system must commit to one
allocation. This module owns that committal rule (`canonical_decision`) and the
metrics that compare a decision made on degraded inputs against the decision made
on clean inputs — both scored on the clean ground truth through the single shared
evaluator. The committal rule is fixed (equal-weight, min–max normalised sum) so
"decision stability" means one thing across the whole study.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from presidio_vol_assign.allocation.models import (
    AllocationParetoFront,
    AllocationProblem,
    AllocationSolution,
)
from presidio_vol_assign.allocation.solvers import FISCache, evaluate_pairs


def canonical_decision(front: AllocationParetoFront) -> AllocationSolution:
    """Commit to one solution: the min equal-weight, min–max-normalised objective sum.

    Fail closed: an empty front has no decision to commit — raise rather than
    return a silent default that a caller might mistake for a real allocation.
    Ties resolve to the lowest index, so the choice is deterministic.
    """
    solutions = front.solutions
    if not solutions:
        raise ValueError("cannot extract a decision from an empty front")
    fitnesses = np.array([s.fitness for s in solutions], dtype=float)
    lo = fitnesses.min(axis=0)
    hi = fitnesses.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)  # constant objective -> no contribution
    scores = ((fitnesses - lo) / span).sum(axis=1)
    return solutions[int(np.argmin(scores))]


def pairs_of(
    solution: AllocationSolution,
    problem: AllocationProblem,
) -> list[tuple[int, int]]:
    """Map a solution's (person_id, center_id) allocations to index pairs."""
    person_idx = {p.person_id: i for i, p in enumerate(problem.people)}
    center_idx = {c.center_id: j for j, c in enumerate(problem.centers)}
    return [(person_idx[a.person_id], center_idx[a.center_id]) for a in solution.allocations]


def _loads(pairs: list[tuple[int, int]], n_centers: int) -> list[int]:
    load = [0] * n_centers
    for _, center in pairs:
        load[center] += 1
    return load


def _safe_spearman(a: list[int], b: list[int]) -> float:
    # Spearman is undefined when either vector is constant; report NaN, do not
    # silently coerce to a correlation that was never measured.
    if len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def decision_stability(
    clean_pairs: list[tuple[int, int]],
    perturbed_pairs: list[tuple[int, int]],
    clean_cache: FISCache,
    objectives: int,
    n_centers: int,
) -> dict[str, float]:
    """Compare a perturbed-input decision to the clean decision, both on clean truth.

    Returns:
        objective_drift: Euclidean distance between the two realised objective
            vectors (both scored on the clean cache).
        quality_loss: signed sum of realised objectives, perturbed minus clean;
            positive means the turbulence-driven decision is genuinely worse
            (all objectives are minimised).
        allocation_churn: fraction of clean-directed people whose assignment
            changed (re-routed or no longer directed).
        load_rank_stability: Spearman rho between clean and perturbed per-centre
            loads (NaN when a load vector is constant).
    """
    clean_obj = np.asarray(evaluate_pairs(clean_pairs, clean_cache, objectives))
    perturbed_obj = np.asarray(evaluate_pairs(perturbed_pairs, clean_cache, objectives))

    clean_map = dict(clean_pairs)
    perturbed_map = dict(perturbed_pairs)
    changed = sum(1 for person, center in clean_map.items() if perturbed_map.get(person) != center)
    churn = changed / len(clean_map) if clean_map else 0.0

    return {
        "objective_drift": float(np.linalg.norm(perturbed_obj - clean_obj)),
        "quality_loss": float(perturbed_obj.sum() - clean_obj.sum()),
        "allocation_churn": churn,
        "load_rank_stability": _safe_spearman(
            _loads(clean_pairs, n_centers), _loads(perturbed_pairs, n_centers)
        ),
    }
