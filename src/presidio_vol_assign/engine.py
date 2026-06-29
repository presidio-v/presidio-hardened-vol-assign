"""Domain-agnostic evolutionary engine for NSGA-II and NRGA.

The engine owns everything that is independent of the problem family: the
(mu + lambda) generational loop, crossover/mutation scheduling, non-dominated
survival selection, and Pareto-front extraction. The per-domain pieces — genome
construction, genetic operators, objective evaluation, and solution
reconstruction — are supplied by a :class:`~presidio_vol_assign.domains.base.Domain`.

The loop is intentionally identical in structure and RNG-call order to the
original v0.1.0 ``solvers._evolve`` so that seeded runs remain reproducible and
the ED-staffing model's behaviour is unchanged.

NSGA-II vs NRGA differ only in survivor selection:
    NSGA-II: ``tools.selNSGA2`` (crowding distance breaks ties within a front)
    NRGA:    :func:`sel_nrga` (uniform random selection within the last front)
"""

from __future__ import annotations

import random as _random
import time
from copy import deepcopy

import numpy as np
from deap import base, creator, tools

from presidio_vol_assign.domains.base import Domain
from presidio_vol_assign.models import ParetoFront, ProblemInstance, RunConfig, SolverType

_CXPB = 0.7  # crossover probability
_MUTPB = 0.2  # mutation probability


# ---------------------------------------------------------------------------
# DEAP creator type registration (idempotent)
# ---------------------------------------------------------------------------


def ensure_creator_types(domain: Domain) -> tuple[type, type]:
    """Register (once) the DEAP fitness + individual types for *domain*.

    Returns the (fitness_cls, individual_cls) pair. Safe to call repeatedly;
    DEAP raises if a name is re-created, so existing types are reused.
    """
    if not hasattr(creator, domain.fitness_attr):
        creator.create(domain.fitness_attr, base.Fitness, weights=domain.weights)
    fitness_cls = getattr(creator, domain.fitness_attr)
    if not hasattr(creator, domain.individual_attr):
        creator.create(domain.individual_attr, list, fitness=fitness_cls)
    individual_cls = getattr(creator, domain.individual_attr)
    return fitness_cls, individual_cls


# ---------------------------------------------------------------------------
# NRGA selection variants
# ---------------------------------------------------------------------------


def sel_nrga(individuals: list, k: int) -> list:
    """Lightweight NRGA survivor selection (``--solver nrga``).

    Fills the next generation front-by-front (non-domination rank order). When a
    front would overflow the quota, the remaining spots are filled by uniform
    random sampling from that front — no crowding distance.
    """
    fronts = tools.sortNondominated(individuals, len(individuals))
    chosen: list = []
    for front in fronts:
        if len(chosen) + len(front) <= k:
            chosen.extend(front)
        else:
            needed = k - len(chosen)
            chosen.extend(_random.sample(front, needed))
            break
    return chosen


def sel_nrga_ranked(individuals: list, k: int) -> list:
    """Canonical NRGA survivor selection (``--solver nrga-ranked``).

    Implements the Non-dominated Ranked Genetic Algorithm of Al Jadaan, Rajamani
    & Rao (2008): individuals are sorted into non-dominated fronts, concatenated
    best-front-first into a global rank order, and assigned a linear rank weight
    (the best-ranked individual gets weight ``n``, the worst gets ``1``). The ``k``
    survivors are then drawn by **rank-biased roulette-wheel sampling without
    replacement** — better-ranked individuals are more likely to survive, but
    selection is stochastic rather than the strict crowding-distance elitism of
    NSGA-II. Seeded, hence reproducible.
    """
    fronts = tools.sortNondominated(individuals, len(individuals))
    ordered = [ind for front in fronts for ind in front]
    n = len(ordered)
    if k >= n:
        return ordered

    pool = list(ordered)
    weights = [float(n - idx) for idx in range(n)]  # linear ranking, best first
    chosen: list = []
    for _ in range(k):
        total = sum(weights)
        r = _random.random() * total
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                chosen.append(pool.pop(i))
                weights.pop(i)
                break
    return chosen


# Survivor-selection strategy per evolutionary solver type.
_SELECTORS = {
    SolverType.NSGA2: tools.selNSGA2,
    SolverType.NRGA: sel_nrga,
    SolverType.NRGA_RANKED: sel_nrga_ranked,
}


# ---------------------------------------------------------------------------
# Non-evolutionary baseline comparators (greedy + exact)
# ---------------------------------------------------------------------------

# Maps each non-evolutionary solver to the Domain hook that builds its
# candidate population. Anything not listed runs the evolutionary loop.
_CONSTRUCTIVE_HOOKS = {
    SolverType.GREEDY: "baseline_population",
    SolverType.EXACT: "exact_baseline_population",
}


def _build_constructive(
    solver_type: SolverType,
    problem: ProblemInstance,
    domain: Domain,
    cache: object,
    individual_cls: type,
) -> list:
    """Build and evaluate a deterministic constructive population for *domain*.

    ``solver_type`` selects the domain hook (greedy heuristic or exact
    weighted-sum). Returns the evaluated candidate individuals so the shared
    Pareto extractor can trim them to a front. Raises ``ValueError`` if the
    domain provides no comparator of that kind.
    """
    hook = getattr(domain, _CONSTRUCTIVE_HOOKS[solver_type])
    population = hook(problem, cache, individual_cls)
    if population is None:
        raise ValueError(
            f"domain {domain.name!r} provides no {solver_type.value!r} comparator "
            f"(solver {solver_type.value!r} is unavailable for this model)"
        )
    for ind in population:
        ind.fitness.values = domain.evaluate(ind, cache, problem)
    return population


# ---------------------------------------------------------------------------
# Evolutionary loop (shared by NSGA-II and NRGA)
# ---------------------------------------------------------------------------


def _evolve(
    problem: ProblemInstance,
    config: RunConfig,
    domain: Domain,
    cache: object,
    individual_cls: type,
    select,
) -> list:
    """Run (mu + lambda) evolution.  Returns the final population.

    ``select`` is the survivor-selection operator (one of the ``_SELECTORS``).
    """
    if config.seed is not None:
        _random.seed(config.seed)
        np.random.seed(config.seed)

    # Initial population
    population = [domain.init_individual(problem, individual_cls) for _ in range(config.pop_size)]

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = domain.evaluate(ind, cache, problem)

    for _ in range(config.generations):
        # Clone population to produce offspring
        offspring = [deepcopy(ind) for ind in population]

        # Crossover
        for i in range(1, len(offspring), 2):
            if _random.random() < _CXPB:
                offspring[i - 1], offspring[i] = domain.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # Mutation
        for ind in offspring:
            if _random.random() < _MUTPB:
                domain.mutate(ind)
                del ind.fitness.values

        # Re-evaluate modified offspring
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = domain.evaluate(ind, cache, problem)

        # Survivor selection: (mu + lambda)
        population = select(population + offspring, config.pop_size)

    return population


# ---------------------------------------------------------------------------
# Pareto-front extraction
# ---------------------------------------------------------------------------


def _extract_pareto_front(
    population: list,
    cache: object,
    problem: ProblemInstance,
    domain: Domain,
    solver_type: SolverType,
    cpu_time_sec: float,
) -> ParetoFront:
    first_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    # Sort by the first objective for a stable, readable front ordering.
    first_front.sort(key=lambda ind: ind.fitness.values[0])
    solutions = [domain.to_solution(ind, cache, problem) for ind in first_front]
    return ParetoFront(solver=solver_type, solutions=solutions, cpu_time_sec=cpu_time_sec)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _solvers_for(solver_val: str) -> list[SolverType]:
    """Expand a ``--solver`` value into the ordered list of solvers to run."""
    if solver_val == "both":
        return [SolverType.NSGA2, SolverType.NRGA]
    if solver_val == "all":
        return [SolverType.NSGA2, SolverType.NRGA, SolverType.NRGA_RANKED]
    return [SolverType(solver_val)]


def run(
    problem: ProblemInstance,
    config: RunConfig,
    domain: Domain,
    *,
    cache: object | None = None,
) -> list[ParetoFront]:
    """Run the configured solver(s) for *domain* and return one front per solver.

    ``config.solver`` may be ``"nsga2"``, ``"nrga"``, ``"nrga-ranked"``,
    ``"both"`` (nsga2 + nrga), or ``"all"`` (all three).

    If *cache* is provided it is used directly (e.g. a perturbed cache from the
    sensitivity analysis); otherwise it is computed via ``domain.precompute``.
    """
    _fitness_cls, individual_cls = ensure_creator_types(domain)
    if cache is None:
        cache = domain.precompute(problem)

    solver_val = config.solver if isinstance(config.solver, str) else config.solver.value
    solvers_to_run = _solvers_for(solver_val)

    results: list[ParetoFront] = []
    for solver_type in solvers_to_run:
        t0 = time.monotonic()
        if solver_type in _CONSTRUCTIVE_HOOKS:
            population = _build_constructive(solver_type, problem, domain, cache, individual_cls)
        else:
            population = _evolve(
                problem,
                config,
                domain,
                cache,
                individual_cls,
                select=_SELECTORS[solver_type],
            )
        elapsed = time.monotonic() - t0
        results.append(
            _extract_pareto_front(population, cache, problem, domain, solver_type, elapsed)
        )
    return results
