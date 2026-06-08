"""NSGA-II, NRGA, and NSGA-III solvers for the allocation model.

Implements the metaheuristic layer for the people-to-relief-centers
allocation problem described in Rabiei, Arias-Aranda, Stantchev (ATRes
2026, in press) and extended for the MDPI Applied Sciences paper.

Chromosome encoding (ATRes Fig. 2, flattened):

    individual[0 : n_dir]            person indices — partial permutation
                                     of range(n_people), no duplicates
    individual[n_dir : 2*n_dir]      real values in [0, 1), each encoding
                                     a center via int(r * n_centers)

The order of (person, center) pairs is not semantically significant — the
decoder treats them as a set of pairs. Variation operators preserve only
the partial-permutation invariant on the person slice.

Objective evaluation:

    Mn_ULPP  = mean ULPP over directed people     (always present)
    Mn_TIL   = mean TIL  over directed (p, c) pairs   (3-obj mode only)
    Mn_TRD   = mean TRD  over directed (p, c) pairs   (4-obj mode only)
    Mn_RPD   = mean RPD  over directed (p, c) pairs   (4-obj mode only)
    Mn_CAIL  = mean CAIL over directed (p, c) pairs   (always present)

ATRes Eq. (3), (7), (10) reduce to per-directed-person means under the
constraint that each directed person has exactly one assignment.

NSGA-II vs. NRGA vs. NSGA-III differ only in the survivor selection
operator; the variation operators and population-management loop are
identical. NSGA-III uses Das-Dennis structured reference points
(p divisions, default p=4 → 35 points in 4D simplex per Deb & Jain 2014).

Public API:

    precompute_fis_cache(problem, config)      -> FISCache
    decode_chromosome(individual, n_dir, ...)  -> list[(person_idx, center_idx)]
    evaluate_chromosome(individual, cache, ...) -> tuple[float, ...]
    solve(problem, config)                     -> list[AllocationParetoFront]
"""

from __future__ import annotations

import random as _random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from deap import base, creator, tools

from presidio_vol_assign.allocation.fis import (
    compute_rws,
    compute_vs,
    evaluate_fis1_ulpp,
    evaluate_fis2_til,
    evaluate_fis2a_trd,
    evaluate_fis2b_rpd,
    evaluate_fis3_cail,
)
from presidio_vol_assign.allocation.models import (
    Allocation,
    AllocationConfig,
    AllocationParetoFront,
    AllocationProblem,
    AllocationSolution,
    AllocationSolverType,
)
from presidio_vol_assign.solvers import sel_nrga  # reuse the proven NRGA selector

# ---------------------------------------------------------------------------
# DEAP creator setup — separate fitness types for 3-obj and 4-obj runs
# ---------------------------------------------------------------------------

if not hasattr(creator, "AllocFitness3"):
    creator.create("AllocFitness3", base.Fitness, weights=(-1.0, -1.0, -1.0))
if not hasattr(creator, "AllocFitness4"):
    creator.create("AllocFitness4", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
if not hasattr(creator, "AllocIndividual3"):
    creator.create("AllocIndividual3", list, fitness=creator.AllocFitness3)
if not hasattr(creator, "AllocIndividual4"):
    creator.create("AllocIndividual4", list, fitness=creator.AllocFitness4)


# ---------------------------------------------------------------------------
# FIS pre-computation cache
# ---------------------------------------------------------------------------


@dataclass
class FISCache:
    """Pre-computed FIS scores indexed by person and (person, center).

    Built once per `solve()` call. Evaluating a chromosome becomes O(n_dir)
    array lookups, avoiding repeated scikit-fuzzy ControlSystemSimulation
    calls inside the evolutionary loop.

    Attributes:
        ulpp: shape (m,) — ULPP_j for each person j.
        til:  shape (m, n) — TIL_{j,i} (populated only when objectives==3).
        trd:  shape (m, n) — TRD_{j,i} (populated only when objectives==4).
        rpd:  shape (m, n) — RPD_{j,i} (populated only when objectives==4).
        cail: shape (m, n) — CAIL_{j,i} (always populated).
    """

    ulpp: np.ndarray
    til: np.ndarray
    trd: np.ndarray
    rpd: np.ndarray
    cail: np.ndarray


def precompute_fis_cache(problem: AllocationProblem, config: AllocationConfig) -> FISCache:
    """Pre-compute all FIS evaluations for a problem instance.

    Cost: O(m + 4·m·n) FIS calls for 4-obj mode, O(m + 3·m·n) for 3-obj.
    Run once before the evolutionary loop; results are O(1) lookups
    indexed by person and center indices in the encoded chromosome.
    """
    m = problem.n_people
    n = problem.n_centers
    ulpp = np.zeros(m, dtype=float)
    til = np.zeros((m, n), dtype=float)
    trd = np.zeros((m, n), dtype=float)
    rpd = np.zeros((m, n), dtype=float)
    cail = np.zeros((m, n), dtype=float)

    weights = config.weights
    person_idx = {p.person_id: i for i, p in enumerate(problem.people)}
    center_idx = {c.center_id: j for j, c in enumerate(problem.centers)}

    # ULPP — depends only on person attributes
    for i, person in enumerate(problem.people):
        vs = compute_vs(person, weights)
        ulpp[i] = evaluate_fis1_ulpp(
            vs, person.infrastructure_damage_level, person.resource_time_remaining
        )

    # Per-pair indices: TIL (3-obj), TRD/RPD (4-obj), CAIL (always)
    for (pid, cid), travel in problem.travel.items():
        i = person_idx[pid]
        j = center_idx[cid]
        center = problem.centers[j]

        cail[i, j] = evaluate_fis3_cail(
            center.center_occupancy_rate,
            center.resource_depletion_rate,
            travel.travel_duration,
        )
        if config.objectives == 3:
            rws = compute_rws(travel, weights)
            til[i, j] = evaluate_fis2_til(travel.travel_duration, rws)
        else:
            trd[i, j] = evaluate_fis2a_trd(
                travel.road_condition.score, travel.possible_hazard.score
            )
            rpd[i, j] = evaluate_fis2b_rpd(travel.travel_duration)

    return FISCache(ulpp=ulpp, til=til, trd=trd, rpd=rpd, cail=cail)


# ---------------------------------------------------------------------------
# Chromosome encoding / decoding
# ---------------------------------------------------------------------------


def decode_chromosome(
    individual: list,
    n_dir: int,
    n_centers: int,
) -> list[tuple[int, int]]:
    """Map a (2*n_dir)-length chromosome to (person_idx, center_idx) pairs.

    The first n_dir entries are person indices; the remaining n_dir are
    real-valued [0, 1) center encodings, mapped to integer center indices
    via floor(r * n_centers) clamped to n_centers - 1 (to handle the
    boundary case r ≈ 1.0 exactly).
    """
    pairs: list[tuple[int, int]] = []
    for k in range(n_dir):
        person_idx = int(individual[k])
        center_real = float(individual[n_dir + k])
        center_idx = min(int(center_real * n_centers), n_centers - 1)
        pairs.append((person_idx, center_idx))
    return pairs


def evaluate_chromosome(
    individual: list,
    cache: FISCache,
    n_dir: int,
    n_centers: int,
    objectives: int,
) -> tuple[float, ...]:
    """Compute the objective tuple for a chromosome.

    Returns (Mn_ULPP, Mn_TRD, Mn_RPD, Mn_CAIL) for 4-obj or
            (Mn_ULPP, Mn_TIL, Mn_CAIL) for 3-obj.
    """
    pairs = decode_chromosome(individual, n_dir, n_centers)
    persons = np.fromiter((p for p, _ in pairs), dtype=int, count=n_dir)
    centers = np.fromiter((c for _, c in pairs), dtype=int, count=n_dir)

    mn_ulpp = float(cache.ulpp[persons].mean())
    mn_cail = float(cache.cail[persons, centers].mean())

    if objectives == 4:
        mn_trd = float(cache.trd[persons, centers].mean())
        mn_rpd = float(cache.rpd[persons, centers].mean())
        return (mn_ulpp, mn_trd, mn_rpd, mn_cail)

    mn_til = float(cache.til[persons, centers].mean())
    return (mn_ulpp, mn_til, mn_cail)


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------


def _make_individual(n_people: int, n_dir: int, individual_cls: type) -> list:
    """Sample one valid individual: partial permutation + uniform reals."""
    persons = _random.sample(range(n_people), n_dir)
    centers = [_random.random() for _ in range(n_dir)]
    return individual_cls(list(persons) + list(centers))


# ---------------------------------------------------------------------------
# Variation operators
# ---------------------------------------------------------------------------


def _crossover(ind1: list, ind2: list, n_people: int, n_dir: int) -> tuple[list, list]:
    """Uniform crossover with repair on the person slice; cxBlend on centers.

    Person slice: for each child, pick from parent1 or parent2 at each
    position with probability 0.5; then repair duplicates by replacing
    with a random unused person index.

    Center slice: DEAP's cxBlend with alpha=0.5, results clipped to [0, 1).
    """
    p1_persons = list(ind1[:n_dir])
    p2_persons = list(ind2[:n_dir])

    def _uniform_persons(a: list[int], b: list[int]) -> list[int]:
        child = [a[k] if _random.random() < 0.5 else b[k] for k in range(n_dir)]
        seen: set[int] = set()
        out: list[int] = []
        for v in child:
            if v not in seen:
                out.append(v)
                seen.add(v)
        unused = list(set(range(n_people)) - seen)
        _random.shuffle(unused)
        while len(out) < n_dir:
            out.append(unused.pop())
        return out

    c1_persons = _uniform_persons(p1_persons, p2_persons)
    c2_persons = _uniform_persons(p2_persons, p1_persons)

    p1_centers = list(ind1[n_dir:])
    p2_centers = list(ind2[n_dir:])
    c1_centers = []
    c2_centers = []
    alpha = 0.5
    for a, b in zip(p1_centers, p2_centers):
        gamma = (1.0 + 2.0 * alpha) * _random.random() - alpha
        v1 = (1.0 - gamma) * a + gamma * b
        v2 = gamma * a + (1.0 - gamma) * b
        c1_centers.append(min(max(v1, 0.0), 1.0 - 1e-9))
        c2_centers.append(min(max(v2, 0.0), 1.0 - 1e-9))

    ind1[:n_dir] = c1_persons
    ind1[n_dir:] = c1_centers
    ind2[:n_dir] = c2_persons
    ind2[n_dir:] = c2_centers
    del ind1.fitness.values
    del ind2.fitness.values
    return ind1, ind2


def _mutate(individual: list, n_people: int, n_dir: int, indpb: float = 0.05) -> tuple[list]:
    """Per-gene mutation.

    Person slice: with probability `indpb`, replace the gene with a random
    unused person index. Center slice: with probability `indpb`, perturb
    by N(0, 0.1) clipped to [0, 1).
    """
    used = set(int(x) for x in individual[:n_dir])
    unused_pool = list(set(range(n_people)) - used)
    _random.shuffle(unused_pool)

    for k in range(n_dir):
        if _random.random() < indpb and unused_pool:
            new_p = unused_pool.pop()
            old_p = int(individual[k])
            individual[k] = new_p
            used.discard(old_p)
            used.add(new_p)
            unused_pool.append(old_p)

    for k in range(n_dir):
        if _random.random() < indpb:
            v = float(individual[n_dir + k])
            v = v + _random.gauss(0.0, 0.1)
            individual[n_dir + k] = min(max(v, 0.0), 1.0 - 1e-9)

    del individual.fitness.values
    return (individual,)


# ---------------------------------------------------------------------------
# Evolutionary loop
# ---------------------------------------------------------------------------

_CXPB = 0.7
_MUTPB = 0.2


def _evolve(
    problem: AllocationProblem,
    config: AllocationConfig,
    cache: FISCache,
    individual_cls: type,
    selector: Any,
) -> list:
    """Run a (mu+lambda) evolutionary loop with the given selection operator.

    Returns the final population after `config.generations` generations.
    """
    if config.seed is not None:
        _random.seed(config.seed)
        np.random.seed(config.seed)

    n_people = problem.n_people
    n_dir = problem.n_dir
    n_centers = problem.n_centers
    objectives = config.objectives

    population = [_make_individual(n_people, n_dir, individual_cls) for _ in range(config.pop_size)]
    for ind in population:
        ind.fitness.values = evaluate_chromosome(ind, cache, n_dir, n_centers, objectives)

    for _ in range(config.generations):
        offspring = [individual_cls(list(ind)) for ind in population]
        for ind, parent in zip(offspring, population):
            ind.fitness.values = parent.fitness.values

        # Crossover (paired)
        for i in range(1, len(offspring), 2):
            if _random.random() < _CXPB:
                _crossover(offspring[i - 1], offspring[i], n_people, n_dir)

        # Mutation
        for ind in offspring:
            if _random.random() < _MUTPB:
                _mutate(ind, n_people, n_dir)

        # Re-evaluate any modified offspring
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate_chromosome(ind, cache, n_dir, n_centers, objectives)

        population = selector(population + offspring, config.pop_size)

    return population


# ---------------------------------------------------------------------------
# Pareto-front extraction
# ---------------------------------------------------------------------------


def _extract_pareto_front(
    population: list,
    cache: FISCache,
    problem: AllocationProblem,
    config: AllocationConfig,
    solver_type: AllocationSolverType,
    cpu_time_sec: float,
) -> AllocationParetoFront:
    first_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    solutions: list[AllocationSolution] = []
    for ind in first_front:
        solutions.append(_individual_to_solution(ind, cache, problem, config))
    if config.objectives == 4:
        solutions.sort(key=lambda s: (s.mn_ulpp, s.mn_trd, s.mn_rpd, s.mn_cail))
    else:
        solutions.sort(key=lambda s: (s.mn_ulpp, s.mn_til, s.mn_cail))
    return AllocationParetoFront(
        solver=solver_type,
        objectives_count=config.objectives,
        solutions=solutions,
        cpu_time_sec=cpu_time_sec,
    )


def _individual_to_solution(
    individual: list,
    cache: FISCache,
    problem: AllocationProblem,
    config: AllocationConfig,
) -> AllocationSolution:
    n_dir = problem.n_dir
    n_centers = problem.n_centers
    pairs = decode_chromosome(individual, n_dir, n_centers)
    fitness = individual.fitness.values
    allocations: list[Allocation] = []
    for p_idx, c_idx in pairs:
        person = problem.people[p_idx]
        center = problem.centers[c_idx]
        alloc = Allocation(
            person_id=person.person_id,
            center_id=center.center_id,
            ulpp=float(cache.ulpp[p_idx]),
            cail_contrib=float(cache.cail[p_idx, c_idx]),
        )
        if config.objectives == 4:
            alloc.trd = float(cache.trd[p_idx, c_idx])
            alloc.rpd = float(cache.rpd[p_idx, c_idx])
        else:
            alloc.til = float(cache.til[p_idx, c_idx])
        allocations.append(alloc)

    if config.objectives == 4:
        return AllocationSolution(
            allocations=allocations,
            objectives_count=4,
            mn_ulpp=float(fitness[0]),
            mn_trd=float(fitness[1]),
            mn_rpd=float(fitness[2]),
            mn_cail=float(fitness[3]),
        )
    return AllocationSolution(
        allocations=allocations,
        objectives_count=3,
        mn_ulpp=float(fitness[0]),
        mn_til=float(fitness[1]),
        mn_cail=float(fitness[2]),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def solve(
    problem: AllocationProblem,
    config: AllocationConfig,
    cache: FISCache | None = None,
) -> AllocationParetoFront:
    """Run a single solver and return its Pareto front.

    The CLI layer is responsible for invoking solve() multiple times when
    the user requests several solvers; this keeps the seed / FIS-cache /
    timing semantics clean per run.

    `cache` (optional) lets a caller reuse a pre-computed `FISCache`
    across many runs that share the same `(problem, weights, objectives,
    fis_overrides)` configuration — typical for the experiment matrix
    where FIS evaluation is the dominant cost. Pass None to recompute.
    """
    if cache is None:
        cache = precompute_fis_cache(problem, config)

    individual_cls = (
        creator.AllocIndividual4 if config.objectives == 4 else creator.AllocIndividual3
    )
    solver_type = (
        config.solver
        if isinstance(config.solver, AllocationSolverType)
        else AllocationSolverType(config.solver)
    )

    selector = _make_selector(solver_type, config)

    t0 = time.monotonic()
    population = _evolve(problem, config, cache, individual_cls, selector)
    elapsed = time.monotonic() - t0
    return _extract_pareto_front(population, cache, problem, config, solver_type, elapsed)


def _make_selector(solver_type: AllocationSolverType, config: AllocationConfig) -> Any:
    """Build the survivor-selection operator for the chosen MOEA."""
    if solver_type == AllocationSolverType.NSGA2:
        return tools.selNSGA2
    if solver_type == AllocationSolverType.NRGA:
        return sel_nrga
    if solver_type == AllocationSolverType.NSGA3:
        ref_points = tools.uniform_reference_points(
            nobj=config.objectives, p=config.nsga3_divisions
        )

        def _select_nsga3(individuals: list, k: int) -> list:
            return tools.selNSGA3(individuals, k, ref_points)

        return _select_nsga3
    raise ValueError(f"Unsupported solver: {solver_type}")
