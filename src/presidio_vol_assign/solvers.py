"""NSGA-II and NRGA solvers for the volunteer assignment model.

Implements the metaheuristic layer described in Rabiei et al. (ESWA, 2023).

Chromosome encoding:
    A permutation of range(n_volunteers). For each vacancy (in order),
    the first unused volunteer in chromosome order whose skill_type matches
    the vacancy is assigned to it (greedy type-matching decoder). Any valid
    permutation produces a feasible assignment, so standard ordered crossover
    and shuffle mutation can be applied without a repair step.

Objective evaluation:
    FIS scores are pre-computed for all (volunteer, vacancy) pairs once
    before the evolutionary loop. Chromosome evaluation is then O(n_vacancies)
    dictionary lookups — avoiding repeated scikit-fuzzy calls per generation.

NSGA-II vs NRGA:
    Both use the same evolutionary loop. The only difference is the survivor
    selection operator:
        NSGA-II: selNSGA2 from DEAP (crowding distance breaks ties within fronts)
        NRGA:    sel_nrga (uniform random selection within the last partial front)

Public API:
    precompute_fis(problem) -> FISCache
    decode_chromosome(chromosome, volunteers, vacancies) -> list[tuple[int, int]]
    evaluate_chromosome(chromosome, fis_cache, volunteers, vacancies) -> tuple[float, float]
    solve(problem, config) -> list[ParetoFront]
"""

from __future__ import annotations

import random as _random
import time
from copy import deepcopy
from typing import Optional

import numpy as np
from deap import base, creator, tools

from presidio_vol_assign.fis import (
    compute_workload,
    evaluate_fis1,
    evaluate_fis2,
    evaluate_fis3,
)
from presidio_vol_assign.models import (
    Assignment,
    ParetoFront,
    ProblemInstance,
    RunConfig,
    SkillType,
    Solution,
    SolverType,
    Vacancy,
    Volunteer,
)

# ---------------------------------------------------------------------------
# DEAP creator setup — idempotent, uses "PVA" prefix to avoid collisions
# ---------------------------------------------------------------------------
if not hasattr(creator, "PVAFitness"):
    creator.create("PVAFitness", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "PVAIndividual"):
    creator.create("PVAIndividual", list, fitness=creator.PVAFitness)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# Maps (volunteer_idx, vacancy_idx) -> (importance, preference) | None (infeasible)
FISCache = dict[tuple[int, int], Optional[tuple[float, float]]]


# ---------------------------------------------------------------------------
# FIS pre-computation
# ---------------------------------------------------------------------------


def precompute_fis(problem: ProblemInstance) -> FISCache:
    """Compute FIS scores for all (volunteer, vacancy) pairs.

    Infeasible pairs (skill-type mismatch) are stored as None.
    This avoids repeated scikit-fuzzy calls inside the evolutionary loop.
    """
    cache: FISCache = {}
    for vi, vol in enumerate(problem.volunteers):
        for vj, vac in enumerate(problem.vacancies):
            if vol.skill_type != vac.vacancy_type:
                cache[(vi, vj)] = None
                continue
            workload = compute_workload(vac.num_patients, vac.emergency_level)
            distance = vol.distance_to(vac.ed_id)
            if vac.vacancy_type == SkillType.TRIAGE:
                importance = evaluate_fis1(vac.num_patients, vac.emergency_level, vol.skill_level)
            else:
                importance = evaluate_fis2(vac.num_patients, vac.emergency_level, vol.skill_level)
            preference = evaluate_fis3(distance, workload, vol.difficulty_tolerance)
            cache[(vi, vj)] = (importance, preference)
    return cache


# ---------------------------------------------------------------------------
# Chromosome encoding / decoding
# ---------------------------------------------------------------------------


def decode_chromosome(
    chromosome: list[int],
    volunteers: list[Volunteer],
    vacancies: list[Vacancy],
) -> list[tuple[int, int]]:
    """Map a permutation to (volunteer_idx, vacancy_idx) assignment pairs.

    For each vacancy in order, assign the first unused volunteer in chromosome
    order whose skill_type matches the vacancy type.

    Returns a list of length len(vacancies); guaranteed feasible if validation
    has confirmed enough volunteers of each type.
    """
    used: set[int] = set()
    result: list[tuple[int, int]] = []
    for vj, vac in enumerate(vacancies):
        for vi in chromosome:
            if vi not in used and volunteers[vi].skill_type == vac.vacancy_type:
                result.append((vi, vj))
                used.add(vi)
                break
    return result


def evaluate_chromosome(
    chromosome: list[int],
    fis_cache: FISCache,
    volunteers: list[Volunteer],
    vacancies: list[Vacancy],
) -> tuple[float, float]:
    """Compute (Z1, Z2) for a chromosome using the pre-cached FIS scores.

    Z1 = mean importance of unmet needs (minimise).
    Z2 = mean volunteer preference dissatisfaction (minimise).
    """
    pairs = decode_chromosome(chromosome, volunteers, vacancies)
    if not pairs:
        return (1.0, 1.0)

    importance_scores: list[float] = []
    preference_scores: list[float] = []
    for vi, vj in pairs:
        entry = fis_cache.get((vi, vj))
        if entry is None:
            importance_scores.append(1.0)
            preference_scores.append(1.0)
        else:
            importance_scores.append(entry[0])
            preference_scores.append(entry[1])

    return float(np.mean(importance_scores)), float(np.mean(preference_scores))


# ---------------------------------------------------------------------------
# Solution reconstruction
# ---------------------------------------------------------------------------


def _pairs_to_solution(
    pairs: list[tuple[int, int]],
    fis_cache: FISCache,
    volunteers: list[Volunteer],
    vacancies: list[Vacancy],
    z1: float,
    z2: float,
) -> Solution:
    assignments: list[Assignment] = []
    for vi, vj in pairs:
        vol = volunteers[vi]
        vac = vacancies[vj]
        entry = fis_cache.get((vi, vj)) or (0.0, 0.0)
        imp, pref = entry
        if vac.vacancy_type == SkillType.TRIAGE:
            a = Assignment(
                volunteer_id=vol.volunteer_id,
                ed_id=vac.ed_id,
                vacancy_type=vac.vacancy_type,
                fis1_score=imp,
                fis3_score=pref,
            )
        else:
            a = Assignment(
                volunteer_id=vol.volunteer_id,
                ed_id=vac.ed_id,
                vacancy_type=vac.vacancy_type,
                fis2_score=imp,
                fis3_score=pref,
            )
        assignments.append(a)
    return Solution(assignments=assignments, z1=z1, z2=z2)


# ---------------------------------------------------------------------------
# NRGA selection (replaces crowding distance with uniform random sampling)
# ---------------------------------------------------------------------------


def sel_nrga(individuals: list, k: int) -> list:
    """NRGA survivor selection.

    Fills the next generation front-by-front (non-domination rank order).
    When a front would overflow the quota, the remaining spots are filled
    by uniform random sampling from that front — no crowding distance.
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

_CXPB = 0.7  # crossover probability
_MUTPB = 0.2  # mutation probability


def _evolve(
    problem: ProblemInstance,
    config: RunConfig,
    fis_cache: FISCache,
    use_nrga: bool,
) -> list:
    """Run mu+lambda evolution.  Returns the final population."""
    if config.seed is not None:
        _random.seed(config.seed)
        np.random.seed(config.seed)

    n_vol = problem.n_volunteers

    # Initial population: random permutations
    population = [
        creator.PVAIndividual(_random.sample(range(n_vol), n_vol)) for _ in range(config.pop_size)
    ]

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = evaluate_chromosome(
            ind, fis_cache, problem.volunteers, problem.vacancies
        )

    select = sel_nrga if use_nrga else tools.selNSGA2

    for _ in range(config.generations):
        # Clone population to produce offspring
        offspring = [deepcopy(ind) for ind in population]

        # Ordered crossover
        for i in range(1, len(offspring), 2):
            if _random.random() < _CXPB:
                offspring[i - 1], offspring[i] = tools.cxOrdered(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # Shuffle mutation
        for ind in offspring:
            if _random.random() < _MUTPB:
                tools.mutShuffleIndexes(ind, indpb=0.05)
                del ind.fitness.values

        # Re-evaluate modified offspring
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate_chromosome(
                    ind, fis_cache, problem.volunteers, problem.vacancies
                )

        # Survivor selection: (mu + lambda)
        population = select(population + offspring, config.pop_size)

    return population


# ---------------------------------------------------------------------------
# Pareto-front extraction
# ---------------------------------------------------------------------------


def _extract_pareto_front(
    population: list,
    fis_cache: FISCache,
    problem: ProblemInstance,
    solver_type: SolverType,
    cpu_time_sec: float,
) -> ParetoFront:
    first_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    solutions: list[Solution] = []
    for ind in first_front:
        pairs = decode_chromosome(ind, problem.volunteers, problem.vacancies)
        z1, z2 = ind.fitness.values
        sol = _pairs_to_solution(pairs, fis_cache, problem.volunteers, problem.vacancies, z1, z2)
        solutions.append(sol)
    solutions.sort(key=lambda s: s.z1)
    return ParetoFront(solver=solver_type, solutions=solutions, cpu_time_sec=cpu_time_sec)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def solve(problem: ProblemInstance, config: RunConfig) -> list[ParetoFront]:
    """Run the specified solver(s) and return one ParetoFront per solver.

    config.solver can be "nsga2", "nrga", or "both".
    """
    fis_cache = precompute_fis(problem)

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
        population = _evolve(problem, config, fis_cache, use_nrga=(solver_type == SolverType.NRGA))
        elapsed = time.monotonic() - t0
        results.append(_extract_pareto_front(population, fis_cache, problem, solver_type, elapsed))
    return results
