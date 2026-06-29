"""Humanitarian domain — allocation of affected people to relief centres.

Three objectives, integer encoding (one gene per person = the index of its
assigned centre). Capacity is modelled *softly*: any assignment is feasible, and
over-capacity centres are penalised through the third objective rather than
repaired, matching the paper's "minimise overcrowding" formulation.

Chromosome encoding:
    A list of length n_people; gene i in [0, n_centers - 1] is the centre that
    person i is allocated to. Every chromosome is feasible, so uniform-style
    crossover and integer-reset mutation apply without a repair step.

Objective evaluation:
    Per-(person, centre) FIS scores (fairness, transport) depend only on the
    pairing and are pre-computed once. Overcrowding depends on the whole
    allocation (centre utilisation) and is read from a pre-computed
    utilisation -> overcrowding lookup table, so the evolutionary loop stays
    fast and deterministic.

    Z1 = mean unfairness over people            (minimise)
    Z2 = mean transport infeasibility over people (minimise)
    Z3 = load-weighted mean centre overcrowding  (minimise)
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from deap import tools

from presidio_vol_assign.domains.base import Domain
from presidio_vol_assign.fis_humanitarian import (
    evaluate_fairness,
    evaluate_overcrowding,
    evaluate_transport,
)
from presidio_vol_assign.models import (
    CenterAssignment,
    HumanitarianProblem,
    Solution,
)

_LUT_POINTS = 201  # utilisation grid resolution over [0, 2]


@dataclass
class _HumCache:
    """Pre-computed per-instance data handed to evaluate / to_solution."""

    pairs: dict[tuple[int, int], tuple[float, float]]  # (person, centre) -> (fairness, transport)
    group_sizes: list[int]
    capacities: list[int]
    util_lut: np.ndarray  # overcrowding sampled over utilisation in [0, 2]


def _overcrowding_from_lut(utilisation: float, lut: np.ndarray) -> float:
    """Look up overcrowding for a utilisation ratio via the pre-sampled grid."""
    clamped = min(max(utilisation, 0.0), 2.0)
    idx = int(round(clamped / 2.0 * (len(lut) - 1)))
    return float(lut[idx])


def _center_overcrowding(
    individual: list[int], cache: _HumCache, n_centers: int
) -> tuple[list[int], list[float]]:
    """Return per-centre (load, overcrowding) for an allocation."""
    loads = [0] * n_centers
    for pi, cj in enumerate(individual):
        loads[cj] += cache.group_sizes[pi]
    overcrowding = [
        _overcrowding_from_lut(load / cache.capacities[cj], cache.util_lut) if load else 0.0
        for cj, load in enumerate(loads)
    ]
    return loads, overcrowding


class HumanitarianDomain(Domain):
    """Three-objective humanitarian people-to-centre allocation model."""

    name = "humanitarian"
    objective_names = ("z1", "z2", "z3")
    reference_point = (1.0, 1.0, 1.0)
    ideal_point = (0.0, 0.0, 0.0)
    weights = (-1.0, -1.0, -1.0)
    fitness_attr = "PVAFitness3"
    individual_attr = "PVAIndividual3"
    required_inputs = ("people", "centers")
    assignment_fieldnames = (
        "person_id",
        "center_id",
        "fairness",
        "transport",
        "overcrowding",
    )

    def __init__(self) -> None:
        # Set in precompute(); needed by mutate(), which the engine calls
        # without the problem in scope. One domain instance per run.
        self._n_centers = 0

    # ------------------------------------------------------------------
    # I/O hooks
    # ------------------------------------------------------------------

    def load(self, primary: Path, secondary: Path) -> HumanitarianProblem:
        from presidio_vol_assign.validation import load_humanitarian_problem

        return load_humanitarian_problem(primary, secondary)

    def assignment_row(self, assignment: Any) -> dict[str, Any]:
        return {
            "person_id": assignment.person_id,
            "center_id": assignment.center_id,
            "fairness": round(assignment.fairness, 6),
            "transport": round(assignment.transport, 6),
            "overcrowding": round(assignment.overcrowding, 6),
        }

    # ------------------------------------------------------------------
    # Evolutionary hooks
    # ------------------------------------------------------------------

    def precompute(self, problem: HumanitarianProblem) -> _HumCache:
        self._n_centers = problem.n_centers

        pairs: dict[tuple[int, int], tuple[float, float]] = {}
        for pi, person in enumerate(problem.people):
            for cj, center in enumerate(problem.centers):
                dist = person.distance_to(center.center_id)
                fairness = evaluate_fairness(person.vulnerability, center.service_level, dist)
                transport = evaluate_transport(dist, person.mobility, center.road_accessibility)
                pairs[(pi, cj)] = (fairness, transport)

        util_lut = np.array(
            [evaluate_overcrowding(u) for u in np.linspace(0.0, 2.0, _LUT_POINTS)],
            dtype=float,
        )
        return _HumCache(
            pairs=pairs,
            group_sizes=[p.group_size for p in problem.people],
            capacities=[c.capacity for c in problem.centers],
            util_lut=util_lut,
        )

    def perturb(self, cache: _HumCache, factor: float) -> _HumCache:
        scale = 1.0 + factor
        pairs = {
            key: (min(max(f * scale, 0.0), 1.0), min(max(t * scale, 0.0), 1.0))
            for key, (f, t) in cache.pairs.items()
        }
        return _HumCache(
            pairs=pairs,
            group_sizes=cache.group_sizes,
            capacities=cache.capacities,
            util_lut=np.clip(cache.util_lut * scale, 0.0, 1.0),
        )

    def init_individual(self, problem: HumanitarianProblem, individual_cls: type) -> list:
        n_centers = problem.n_centers
        return individual_cls(_random.randrange(n_centers) for _ in range(problem.n_people))

    def baseline_population(
        self, problem: HumanitarianProblem, cache: _HumCache, individual_cls: type
    ) -> list:
        """Weighted-sum greedy allocations across the objective simplex.

        For each weight vector ``(w1, w2, w3)`` on the simplex, allocate people
        one at a time — most-vulnerable first — to the centre minimising
        ``w1·fairness + w2·transport + w3·marginal_overcrowding``, where the
        overcrowding term reflects the centre's utilisation *after* adding this
        person's group. Running loads make the construction capacity-aware
        without a repair step. Deterministic (no RNG); one candidate per weight.
        """
        from presidio_vol_assign.baselines import weight_simplex

        n_people = problem.n_people
        n_centers = problem.n_centers
        # Most-vulnerable-first, stable on ties → deterministic processing order.
        order = sorted(range(n_people), key=lambda pi: -problem.people[pi].vulnerability)

        population: list = []
        for w1, w2, w3 in weight_simplex(3, steps=6):
            genome = [0] * n_people
            loads = [0] * n_centers
            for pi in order:
                group = cache.group_sizes[pi]
                best_cj, best_cost = 0, float("inf")
                for cj in range(n_centers):
                    fairness, transport = cache.pairs[(pi, cj)]
                    util = (loads[cj] + group) / cache.capacities[cj]
                    overcrowding = _overcrowding_from_lut(util, cache.util_lut)
                    cost = w1 * fairness + w2 * transport + w3 * overcrowding
                    if cost < best_cost:
                        best_cj, best_cost = cj, cost
                genome[pi] = best_cj
                loads[best_cj] += group
            population.append(individual_cls(genome))
        return population

    def mate(self, ind1: list, ind2: list) -> tuple[list, list]:
        return tools.cxTwoPoint(ind1, ind2)

    def mutate(self, ind: list) -> tuple[list]:
        return tools.mutUniformInt(ind, low=0, up=self._n_centers - 1, indpb=0.05)

    def evaluate(
        self, individual: list, cache: _HumCache, problem: HumanitarianProblem
    ) -> tuple[float, ...]:
        n_people = problem.n_people
        fairness_sum = 0.0
        transport_sum = 0.0
        for pi, cj in enumerate(individual):
            f, t = cache.pairs[(pi, cj)]
            fairness_sum += f
            transport_sum += t

        loads, overcrowding = _center_overcrowding(individual, cache, problem.n_centers)
        total_load = sum(loads)
        if total_load == 0:
            z3 = 0.0
        else:
            z3 = sum(load * oc for load, oc in zip(loads, overcrowding)) / total_load

        return (fairness_sum / n_people, transport_sum / n_people, z3)

    def to_solution(
        self, individual: list, cache: _HumCache, problem: HumanitarianProblem
    ) -> Solution:
        _loads, overcrowding = _center_overcrowding(individual, cache, problem.n_centers)
        assignments: list[CenterAssignment] = []
        for pi, cj in enumerate(individual):
            fairness, transport = cache.pairs[(pi, cj)]
            assignments.append(
                CenterAssignment(
                    person_id=problem.people[pi].person_id,
                    center_id=problem.centers[cj].center_id,
                    fairness=fairness,
                    transport=transport,
                    overcrowding=overcrowding[cj],
                )
            )
        return Solution(assignments=assignments, objectives=tuple(individual.fitness.values))
