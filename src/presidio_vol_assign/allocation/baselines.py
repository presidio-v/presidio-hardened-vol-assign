"""Crisp greedy baseline allocator — the non-fuzzy foil for Paper B, RQ1.

This module owns one decision rule: rank the directed people by a crisp
priority and send each to the crisp-cheapest centre, with no fuzzy inference
anywhere. It is deterministic (no RNG), so its behaviour under input
turbulence is a clean reference against which the fuzzy-MOEA system's graceful
degradation is measured. Objectives are scored through the one shared
evaluation point (``solvers.evaluate_pairs``), so the baseline and the
evolutionary solver are judged on identical ground truth — the crisp rule only
changes *how the decision is made*, never how it is graded.
"""

from __future__ import annotations

import numpy as np

from presidio_vol_assign.allocation.models import (
    Allocation,
    AllocationConfig,
    AllocationProblem,
    AllocationSolution,
    Person,
    ReliefCenter,
    TravelInfo,
)
from presidio_vol_assign.allocation.solvers import FISCache, evaluate_pairs

# Pinned normalisers (match the plausible ranges in ``allocation.turbulence``).
_RTR_MAX = 72.0
_TD_MAX = 180.0


def _crisp_priority(person: Person) -> float:
    """Crisp urgency of a person in [0, 1] — higher is served first.

    A plain mean of normalised crisp criteria; deliberately no FIS, so any
    robustness the fuzzy system shows over this baseline is attributable to
    the fuzzy front-end, not to a different criteria set.
    """
    remaining = min(max(person.resource_time_remaining, 0.0), _RTR_MAX)
    urgency = 1.0 - remaining / _RTR_MAX
    return (
        person.age_score
        + person.disability_status.score
        + person.injury_level.score
        + person.living_status.score
        + min(max(person.infrastructure_damage_level, 0.0), 100.0) / 100.0
        + urgency
    ) / 6.0


def _crisp_cost(center: ReliefCenter, travel: TravelInfo) -> float:
    """Crisp cost of routing a person to *center* in [0, 1] — lower is better."""
    return (
        min(max(travel.travel_duration, 0.0), _TD_MAX) / _TD_MAX
        + min(max(center.center_occupancy_rate, 0.0), 100.0) / 100.0
        + min(max(center.resource_depletion_rate, 0.0), 100.0) / 100.0
        + travel.road_condition.score
        + travel.possible_hazard.score
    ) / 5.0


def crisp_greedy_pairs(
    problem: AllocationProblem,
    config: AllocationConfig,
) -> list[tuple[int, int]]:
    """Decide a feasible allocation crisply: top-priority people, cheapest centre.

    Returns ``n_dir`` (person_idx, center_idx) pairs. Deterministic: ties in
    priority and in cost break toward the lower index, so the same inputs
    always yield the same decision.
    """
    people = problem.people
    centers = problem.centers
    n_dir = problem.n_dir
    if n_dir > len(people):  # fail closed on a malformed instance
        raise ValueError("n_dir exceeds the number of people")

    ranked = sorted(
        range(len(people)),
        key=lambda i: (-_crisp_priority(people[i]), i),
    )
    selected = ranked[:n_dir]

    pairs: list[tuple[int, int]] = []
    for i in selected:
        person = people[i]
        best_j = 0
        best_cost = float("inf")
        for j, center in enumerate(centers):
            cost = _crisp_cost(center, problem.travel[(person.person_id, center.center_id)])
            if cost < best_cost:  # strict: keeps the lowest-index centre on ties
                best_cost = cost
                best_j = j
        pairs.append((i, best_j))
    return pairs


def crisp_greedy_solution(
    problem: AllocationProblem,
    config: AllocationConfig,
    cache: FISCache,
) -> AllocationSolution:
    """Crisp greedy decision, scored against *cache* via the shared evaluator.

    The caller supplies the cache, so the same decision can be graded on the
    clean ground truth even when it was decided on a perturbed instance.
    """
    pairs = crisp_greedy_pairs(problem, config)
    fitness = evaluate_pairs(pairs, cache, config.objectives)
    allocations = [
        Allocation(
            person_id=problem.people[i].person_id,
            center_id=problem.centers[j].center_id,
        )
        for i, j in pairs
    ]
    if config.objectives == 4:
        mn_ulpp, mn_trd, mn_rpd, mn_cail = fitness
        return AllocationSolution(
            allocations=allocations,
            objectives_count=4,
            mn_ulpp=mn_ulpp,
            mn_trd=mn_trd,
            mn_rpd=mn_rpd,
            mn_cail=mn_cail,
        )
    mn_ulpp, mn_til, mn_cail = fitness
    return AllocationSolution(
        allocations=allocations,
        objectives_count=3,
        mn_ulpp=mn_ulpp,
        mn_til=mn_til,
        mn_cail=mn_cail,
    )


def exact_weighted_sum_pairs(
    cache: FISCache,
    n_dir: int,
    objectives: int,
    weights: tuple[float, ...] | None = None,
) -> list[tuple[int, int]]:
    """Exact minimiser of a weighted objective sum for the separable static model.

    Every objective in ``evaluate_pairs`` is a mean over directed people of a
    per-person (ULPP) or per-(person, centre) cache entry, with no capacity
    coupling. A weighted sum therefore decomposes per person: each person's
    best centre is independent of everyone else's, and the optimum directs the
    ``n_dir`` people with the lowest best-case cost. This is exact, not a
    heuristic, and deterministic: ties break toward the lower index.

    It is the reference decider the stochastic MOEA is audited against; it
    stops being exact the moment an objective depends on the assigned load.
    """
    if objectives == 4:
        pair_terms = (cache.trd, cache.rpd, cache.cail)
    else:
        pair_terms = (cache.til, cache.cail)
    w = tuple(weights) if weights is not None else (1.0,) * objectives
    if len(w) != objectives or any(x < 0 for x in w):
        raise ValueError("weights must be one non-negative value per objective")
    m = cache.ulpp.shape[0]
    if not 0 < n_dir <= m:  # fail closed on a malformed instance
        raise ValueError("n_dir must be in (0, number of people]")

    pair_cost = sum(wk * term for wk, term in zip(w[1:], pair_terms, strict=True))
    best_center = np.argmin(pair_cost, axis=1)  # first minimum -> lowest index
    person_cost = w[0] * cache.ulpp + pair_cost[np.arange(m), best_center]
    selected = np.lexsort((np.arange(m), person_cost))[:n_dir]
    return [(int(j), int(best_center[j])) for j in selected]
