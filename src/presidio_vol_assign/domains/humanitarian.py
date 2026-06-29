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

    def __init__(
        self,
        *,
        hard_capacity: bool = False,
        max_distance: float | None = None,
        mobility_threshold: float = 3.0,
    ) -> None:
        # Set in precompute(); needed by mutate(), which the engine calls
        # without the problem in scope. One domain instance per run.
        self._n_centers = 0

        # Hard-constraint mode (default off → original soft-capacity behaviour).
        # When on, every genome is repaired to a capacity-feasible assignment
        # (see _repair), and people whose mobility is below ``mobility_threshold``
        # are not placed beyond ``max_distance`` km.
        self._hard_capacity = hard_capacity
        self._max_distance = max_distance
        self._mobility_threshold = mobility_threshold
        # Repair structures, built in precompute() only when hard_capacity is on.
        self._allowed: list[frozenset[int]] = []  # transport-allowed centres per person
        self._allowed_order: list[list[int]] = []  # candidate centres, nearest first
        self._priority_order: list[int] = []  # people, most-vulnerable first

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
        if self._hard_capacity:
            self._build_repair_structures(problem)

        return _HumCache(
            pairs=pairs,
            group_sizes=[p.group_size for p in problem.people],
            capacities=[c.capacity for c in problem.centers],
            util_lut=util_lut,
        )

    # ------------------------------------------------------------------
    # Hard-constraint repair (only used when hard_capacity is on)
    # ------------------------------------------------------------------

    def _build_repair_structures(self, problem: HumanitarianProblem) -> None:
        """Precompute per-person transport-allowed centres and processing order."""
        n_centers = problem.n_centers
        self._allowed = []
        self._allowed_order = []
        for person in problem.people:
            dists = [person.distance_to(c.center_id) for c in problem.centers]
            by_distance = sorted(range(n_centers), key=lambda cj: dists[cj])
            limited = self._max_distance is not None and person.mobility < self._mobility_threshold
            if limited:
                allowed = {cj for cj in range(n_centers) if dists[cj] <= self._max_distance}
            else:
                allowed = set(range(n_centers))
            if not allowed:  # no centre within reach → documented nearest-feasible fallback
                allowed = set(range(n_centers))
            # Allowed centres first (nearest-first), then the rest as a last resort.
            order = [cj for cj in by_distance if cj in allowed]
            order += [cj for cj in by_distance if cj not in allowed]
            self._allowed.append(frozenset(allowed))
            self._allowed_order.append(order)
        # Claim capacity most-constrained-first so transport-limited people (who
        # can reach the fewest centres) secure a feasible centre before flexible
        # people exhaust it; break ties by vulnerability (most vulnerable first).
        self._priority_order = sorted(
            range(problem.n_people),
            key=lambda pi: (len(self._allowed[pi]), -problem.people[pi].vulnerability),
        )

    def _repair(self, individual: list[int], cache: _HumCache) -> list[int]:
        """Return a capacity-feasible, transport-respecting assignment.

        People are placed most-vulnerable first; each takes its genome-preferred
        centre when that centre is transport-allowed and still has room, otherwise
        the nearest allowed centre with spare capacity. If every allowed centre is
        full (rare; only under capacity contention), it falls back to the centre
        with the most remaining capacity. Deterministic.
        """
        remaining = list(cache.capacities)
        assign = [0] * len(individual)
        for pi in self._priority_order:
            group = cache.group_sizes[pi]
            pref = individual[pi]
            if pref in self._allowed[pi]:
                candidates = [pref] + [cj for cj in self._allowed_order[pi] if cj != pref]
            else:
                candidates = self._allowed_order[pi]
            placed = False
            for cj in candidates:
                if remaining[cj] >= group:
                    assign[pi] = cj
                    remaining[cj] -= group
                    placed = True
                    break
            if not placed:  # capacity-contention fallback: most room available
                cj = max(range(self._n_centers), key=lambda c: remaining[c])
                assign[pi] = cj
                remaining[cj] -= group
        return assign

    def _effective(self, individual: list[int], cache: _HumCache) -> list[int]:
        """The assignment actually scored: the repaired genome in hard mode, else as-is."""
        if self._hard_capacity:
            return self._repair(individual, cache)
        return list(individual)

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

    def exact_baseline_population(
        self, problem: HumanitarianProblem, cache: _HumCache, individual_cls: type
    ) -> list:
        """Exact weighted-sum MILP per weight (``scipy.optimize.milp``).

        For each weight vector ``(w1, w2, w3)`` on the simplex, solve to
        optimality::

            min  w1·Σ fairness·x + w2·Σ transport·x + w3·Σ overload
            s.t. Σ_c x[p,c] = 1                       (each person to one centre)
                 Σ_p group[p]·x[p,c] − overload[c] ≤ capacity[c]
                 x ∈ {0,1},  overload ≥ 0

        ``z1``/``z2`` are linear in the assignment and modelled exactly; centre
        balance uses a **linear capacity-overload surrogate** so the programme
        stays an exact MILP (the true FIS ``z3`` is recomputed for reporting by
        the engine's ``evaluate``). For weights with ``w3 = 0`` this is the exact
        ``z1+z2`` optimum. Deterministic; globally optimal per scalarisation.
        """
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        from presidio_vol_assign.baselines import weight_simplex

        n_people = problem.n_people
        n_centers = problem.n_centers
        groups = cache.group_sizes
        caps = cache.capacities
        n_x = n_people * n_centers  # x[p, c] flattened C-order as p*n_centers + c
        n_vars = n_x + n_centers  # + one overload var per centre

        # Per-pair fairness / transport as (n_people, n_centers) matrices.
        fair = np.zeros((n_people, n_centers))
        trans = np.zeros((n_people, n_centers))
        for (pi, cj), (f, t) in cache.pairs.items():
            fair[pi, cj] = f
            trans[pi, cj] = t

        # Each person assigned to exactly one centre.
        eq_rows = [pi for pi in range(n_people) for _ in range(n_centers)]
        eq_cols = [pi * n_centers + cj for pi in range(n_people) for cj in range(n_centers)]
        a_eq = coo_matrix((np.ones(len(eq_rows)), (eq_rows, eq_cols)), shape=(n_people, n_vars))
        eq_con = LinearConstraint(a_eq, lb=1, ub=1)

        # Σ_p group[p]·x[p,c] − overload[c] ≤ capacity[c].
        ov_rows: list[int] = []
        ov_cols: list[int] = []
        ov_data: list[float] = []
        for cj in range(n_centers):
            for pi in range(n_people):
                ov_rows.append(cj)
                ov_cols.append(pi * n_centers + cj)
                ov_data.append(float(groups[pi]))
            ov_rows.append(cj)
            ov_cols.append(n_x + cj)
            ov_data.append(-1.0)
        a_ov = coo_matrix((ov_data, (ov_rows, ov_cols)), shape=(n_centers, n_vars))
        ov_con = LinearConstraint(
            a_ov, lb=-np.inf, ub=np.array([float(caps[cj]) for cj in range(n_centers)])
        )

        integrality = np.zeros(n_vars)
        integrality[:n_x] = 1  # x binary; overload continuous
        ub = np.empty(n_vars)
        ub[:n_x] = 1.0
        ub[n_x:] = np.inf
        bounds = Bounds(lb=np.zeros(n_vars), ub=ub)

        population: list = []
        for w1, w2, w3 in weight_simplex(3, steps=6):
            c_obj = np.empty(n_vars)
            c_obj[:n_x] = (w1 * fair + w2 * trans).ravel()
            c_obj[n_x:] = w3
            res = milp(
                c=c_obj, constraints=[eq_con, ov_con], integrality=integrality, bounds=bounds
            )
            if not res.success or res.x is None:
                continue
            xs = np.asarray(res.x[:n_x]).reshape(n_people, n_centers)
            genome = [int(np.argmax(xs[pi])) for pi in range(n_people)]
            population.append(individual_cls(genome))

        if not population:  # pragma: no cover - assignment MILP is always feasible
            raise ValueError("exact MILP failed to produce any solution")
        return population

    def mate(self, ind1: list, ind2: list) -> tuple[list, list]:
        return tools.cxTwoPoint(ind1, ind2)

    def mutate(self, ind: list) -> tuple[list]:
        return tools.mutUniformInt(ind, low=0, up=self._n_centers - 1, indpb=0.05)

    def evaluate(
        self, individual: list, cache: _HumCache, problem: HumanitarianProblem
    ) -> tuple[float, ...]:
        effective = self._effective(individual, cache)
        n_people = problem.n_people
        fairness_sum = 0.0
        transport_sum = 0.0
        for pi, cj in enumerate(effective):
            f, t = cache.pairs[(pi, cj)]
            fairness_sum += f
            transport_sum += t

        loads, overcrowding = _center_overcrowding(effective, cache, problem.n_centers)
        total_load = sum(loads)
        if total_load == 0:
            z3 = 0.0
        else:
            z3 = sum(load * oc for load, oc in zip(loads, overcrowding)) / total_load

        return (fairness_sum / n_people, transport_sum / n_people, z3)

    def to_solution(
        self, individual: list, cache: _HumCache, problem: HumanitarianProblem
    ) -> Solution:
        effective = self._effective(individual, cache)
        _loads, overcrowding = _center_overcrowding(effective, cache, problem.n_centers)
        assignments: list[CenterAssignment] = []
        for pi, cj in enumerate(effective):
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
