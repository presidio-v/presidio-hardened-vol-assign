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
# NRGA selection (replaces crowding distance with uniform random sampling)
# ---------------------------------------------------------------------------


def sel_nrga(individuals: list, k: int) -> list:
    """NRGA survivor selection.

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


# ---------------------------------------------------------------------------
# Evolutionary loop (shared by NSGA-II and NRGA)
# ---------------------------------------------------------------------------


def _evolve(
    problem: ProblemInstance,
    config: RunConfig,
    domain: Domain,
    cache: object,
    individual_cls: type,
    use_nrga: bool,
) -> list:
    """Run (mu + lambda) evolution.  Returns the final population."""
    if config.seed is not None:
        _random.seed(config.seed)
        np.random.seed(config.seed)

    # Initial population
    population = [domain.init_individual(problem, individual_cls) for _ in range(config.pop_size)]

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = domain.evaluate(ind, cache, problem)

    select = sel_nrga if use_nrga else tools.selNSGA2

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


def run(problem: ProblemInstance, config: RunConfig, domain: Domain) -> list[ParetoFront]:
    """Run the configured solver(s) for *domain* and return one front per solver.

    ``config.solver`` may be ``"nsga2"``, ``"nrga"``, or ``"both"``.
    """
    _fitness_cls, individual_cls = ensure_creator_types(domain)
    cache = domain.precompute(problem)

    solver_val = config.solver if isinstance(config.solver, str) else config.solver.value
    if solver_val == "both":
        solvers_to_run = [SolverType.NSGA2, SolverType.NRGA]
    elif solver_val == SolverType.NSGA2 or solver_val == "nsga2":
        solvers_to_run = [SolverType.NSGA2]
    else:
        solvers_to_run = [SolverType.NRGA]

    results: list[ParetoFront] = []
    for solver_type in solvers_to_run:
        t0 = time.monotonic()
        population = _evolve(
            problem,
            config,
            domain,
            cache,
            individual_cls,
            use_nrga=(solver_type == SolverType.NRGA),
        )
        elapsed = time.monotonic() - t0
        results.append(
            _extract_pareto_front(population, cache, problem, domain, solver_type, elapsed)
        )
    return results
