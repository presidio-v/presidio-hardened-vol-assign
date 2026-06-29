"""ED-staffing domain — the original Rabiei et al. (2023) model.

Two objectives, permutation encoding with a greedy skill-type-matching decoder.
Behaviour is identical to v0.1.0; this module simply repackages the original
solver internals behind the :class:`Domain` interface so the shared engine can
drive it alongside other models.

Chromosome encoding:
    A permutation of range(n_volunteers). For each vacancy (in order), the first
    unused volunteer in chromosome order whose skill_type matches the vacancy is
    assigned to it (greedy type-matching decoder). Any valid permutation yields a
    feasible assignment, so ordered crossover and shuffle mutation apply without
    a repair step.

Objective evaluation:
    FIS scores are pre-computed for all (volunteer, vacancy) pairs once before
    the evolutionary loop; chromosome evaluation is then O(n_vacancies) lookups.

Public functions (re-exported by ``solvers.py`` for backward compatibility):
    precompute_fis, decode_chromosome, evaluate_chromosome
"""

from __future__ import annotations

import random as _random
from pathlib import Path
from typing import Any

import numpy as np
from deap import tools

from presidio_vol_assign.domains.base import Domain
from presidio_vol_assign.fis import (
    compute_workload,
    evaluate_fis1,
    evaluate_fis2,
    evaluate_fis3,
)
from presidio_vol_assign.models import (
    Assignment,
    ProblemInstance,
    SkillType,
    Solution,
    Vacancy,
    Volunteer,
)

# Maps (volunteer_idx, vacancy_idx) -> (importance, preference) | None (infeasible)
FISCache = dict[tuple[int, int], tuple[float, float] | None]


# ---------------------------------------------------------------------------
# FIS pre-computation
# ---------------------------------------------------------------------------


def precompute_fis(problem: ProblemInstance) -> FISCache:
    """Compute FIS scores for all (volunteer, vacancy) pairs.

    Infeasible pairs (skill-type mismatch) are stored as None. This avoids
    repeated scikit-fuzzy calls inside the evolutionary loop.
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
    order whose skill_type matches the vacancy type. Returns a list of length
    len(vacancies); guaranteed feasible if validation has confirmed enough
    volunteers of each type.
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
# Domain adapter
# ---------------------------------------------------------------------------


class EDStaffingDomain(Domain):
    """The original two-objective ED volunteer-staffing model."""

    name = "ed-staffing"
    objective_names = ("z1", "z2")
    reference_point = (1.0, 1.0)
    ideal_point = (0.0, 0.0)
    weights = (-1.0, -1.0)
    fitness_attr = "PVAFitness"
    individual_attr = "PVAIndividual"
    required_inputs = ("volunteers", "eds")
    assignment_fieldnames = (
        "volunteer_id",
        "ed_id",
        "vacancy_type",
        "fis1_score",
        "fis2_score",
        "fis3_score",
    )

    def load(self, primary: Path, secondary: Path) -> ProblemInstance:
        # Imported here to avoid a circular import (validation -> models only).
        from presidio_vol_assign.validation import load_problem

        return load_problem(primary, secondary)

    def assignment_row(self, assignment: Any) -> dict[str, Any]:
        return {
            "volunteer_id": assignment.volunteer_id,
            "ed_id": assignment.ed_id,
            "vacancy_type": assignment.vacancy_type.value,
            "fis1_score": round(assignment.fis1_score, 6),
            "fis2_score": round(assignment.fis2_score, 6),
            "fis3_score": round(assignment.fis3_score, 6),
        }

    def precompute(self, problem: ProblemInstance) -> FISCache:
        return precompute_fis(problem)

    def perturb(self, cache: FISCache, factor: float) -> FISCache:
        scale = 1.0 + factor
        out: FISCache = {}
        for key, entry in cache.items():
            if entry is None:
                out[key] = None
            else:
                imp, pref = entry
                out[key] = (min(max(imp * scale, 0.0), 1.0), min(max(pref * scale, 0.0), 1.0))
        return out

    def init_individual(self, problem: ProblemInstance, individual_cls: type) -> list:
        n = problem.n_volunteers
        return individual_cls(_random.sample(range(n), n))

    def baseline_population(
        self, problem: ProblemInstance, cache: FISCache, individual_cls: type
    ) -> list:
        """Weighted-sum greedy orderings across the objective simplex.

        For each weight vector ``(w1, w2)`` on the simplex, score every
        volunteer by the best (lowest) ``w1·importance + w2·preference`` it can
        achieve over its feasible vacancies, then order the chromosome
        best-score-first. The greedy type-matching decoder then fills each
        vacancy from these front-loaded high-quality volunteers — a constructive
        heuristic expressed as a permutation. Deterministic (no RNG); one
        candidate per weight.
        """
        from presidio_vol_assign.baselines import weight_simplex

        n = problem.n_volunteers
        # Per-volunteer feasible (importance, preference) pairs, precomputed once.
        feasible: dict[int, list[tuple[float, float]]] = {
            vi: [
                cache[(vi, vj)] for vj in range(problem.n_vacancies) if cache[(vi, vj)] is not None
            ]
            for vi in range(n)
        }

        population: list = []
        for w1, w2 in weight_simplex(2, steps=8):

            def key(vi: int, _w1: float = w1, _w2: float = w2) -> tuple[float, int]:
                pairs = feasible[vi]
                # Volunteers with no feasible vacancy sort last; ``vi`` breaks ties.
                best = min((_w1 * imp + _w2 * pref for imp, pref in pairs), default=float("inf"))
                return (best, vi)

            population.append(individual_cls(sorted(range(n), key=key)))
        return population

    def mate(self, ind1: list, ind2: list) -> tuple[list, list]:
        return tools.cxOrdered(ind1, ind2)

    def mutate(self, ind: list) -> tuple[list]:
        return tools.mutShuffleIndexes(ind, indpb=0.05)

    def evaluate(
        self, individual: list, cache: FISCache, problem: ProblemInstance
    ) -> tuple[float, ...]:
        return evaluate_chromosome(individual, cache, problem.volunteers, problem.vacancies)

    def to_solution(self, individual: list, cache: FISCache, problem: ProblemInstance) -> Solution:
        pairs = decode_chromosome(individual, problem.volunteers, problem.vacancies)
        z1, z2 = individual.fitness.values
        return _pairs_to_solution(pairs, cache, problem.volunteers, problem.vacancies, z1, z2)
