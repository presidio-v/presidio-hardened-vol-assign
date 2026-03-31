"""Tests for the NSGA-II / NRGA solver and supporting functions."""

from __future__ import annotations

import pytest

from presidio_vol_assign.models import (
    ProblemInstance,
    RunConfig,
    SkillType,
    SolverType,
    Vacancy,
    Volunteer,
)
from presidio_vol_assign.solvers import (
    decode_chromosome,
    evaluate_chromosome,
    precompute_fis,
    sel_nrga,
    solve,
)

# ---------------------------------------------------------------------------
# Minimal test problem: 4 volunteers, 2 vacancies (1 triage + 1 ER nurse)
# ---------------------------------------------------------------------------


def _make_problem() -> ProblemInstance:
    volunteers = [
        Volunteer("V1", SkillType.TRIAGE, 7.0, {"ED1": 5.0, "ED2": 20.0}, 6.0),
        Volunteer("V2", SkillType.TRIAGE, 5.0, {"ED1": 12.0, "ED2": 8.0}, 4.0),
        Volunteer("V3", SkillType.ER_NURSE, 8.0, {"ED1": 3.0, "ED2": 15.0}, 7.0),
        Volunteer("V4", SkillType.ER_NURSE, 4.0, {"ED1": 25.0, "ED2": 6.0}, 3.0),
    ]
    vacancies = [
        Vacancy("ED1", SkillType.TRIAGE, 40, 7.0),
        Vacancy("ED2", SkillType.ER_NURSE, 25, 5.0),
    ]
    return ProblemInstance(volunteers=volunteers, vacancies=vacancies)


# ---------------------------------------------------------------------------
# precompute_fis
# ---------------------------------------------------------------------------


def test_precompute_fis_keys() -> None:
    prob = _make_problem()
    cache = precompute_fis(prob)
    # 4 volunteers × 2 vacancies = 8 entries
    assert len(cache) == 8


def test_precompute_fis_infeasible_pairs_are_none() -> None:
    prob = _make_problem()
    cache = precompute_fis(prob)
    # V1 (triage) vs vacancy 1 (er_nurse) → infeasible
    assert cache[(0, 1)] is None  # V1 → ED2 vacancy (er_nurse) — mismatch
    assert cache[(2, 0)] is None  # V3 (er_nurse) → ED1 vacancy (triage) — mismatch


def test_precompute_fis_feasible_pairs_are_tuples() -> None:
    prob = _make_problem()
    cache = precompute_fis(prob)
    entry = cache[(0, 0)]  # V1 (triage) → ED1 (triage) — feasible
    assert entry is not None
    importance, preference = entry
    assert 0.0 <= importance <= 1.0
    assert 0.0 <= preference <= 1.0


# ---------------------------------------------------------------------------
# decode_chromosome
# ---------------------------------------------------------------------------


def test_decode_returns_correct_count() -> None:
    prob = _make_problem()
    chromosome = [0, 1, 2, 3]
    pairs = decode_chromosome(chromosome, prob.volunteers, prob.vacancies)
    assert len(pairs) == prob.n_vacancies


def test_decode_respects_skill_type() -> None:
    prob = _make_problem()
    for perm in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
        pairs = decode_chromosome(perm, prob.volunteers, prob.vacancies)
        for vi, vj in pairs:
            assert prob.volunteers[vi].skill_type == prob.vacancies[vj].vacancy_type


def test_decode_no_duplicate_volunteers() -> None:
    prob = _make_problem()
    chromosome = [0, 1, 2, 3]
    pairs = decode_chromosome(chromosome, prob.volunteers, prob.vacancies)
    assigned_vols = [vi for vi, _ in pairs]
    assert len(assigned_vols) == len(set(assigned_vols))


# ---------------------------------------------------------------------------
# evaluate_chromosome
# ---------------------------------------------------------------------------


def test_evaluate_returns_tuple_in_range() -> None:
    prob = _make_problem()
    cache = precompute_fis(prob)
    z1, z2 = evaluate_chromosome([0, 1, 2, 3], cache, prob.volunteers, prob.vacancies)
    assert 0.0 <= z1 <= 1.0
    assert 0.0 <= z2 <= 1.0


def test_evaluate_consistent() -> None:
    """Same chromosome, same cache → same result."""
    prob = _make_problem()
    cache = precompute_fis(prob)
    r1 = evaluate_chromosome([0, 2, 1, 3], cache, prob.volunteers, prob.vacancies)
    r2 = evaluate_chromosome([0, 2, 1, 3], cache, prob.volunteers, prob.vacancies)
    assert r1 == pytest.approx(r2)


# ---------------------------------------------------------------------------
# sel_nrga
# ---------------------------------------------------------------------------


def test_sel_nrga_returns_k_individuals() -> None:
    from deap import creator

    # Ensure DEAP creator types exist (importing solvers sets them up)
    inds = [creator.PVAIndividual([i]) for i in range(6)]
    for i, ind in enumerate(inds):
        ind.fitness.values = (i * 0.1, 1.0 - i * 0.1)
    selected = sel_nrga(inds, 4)
    assert len(selected) == 4


# ---------------------------------------------------------------------------
# solve (integration — tiny problem, few generations for speed)
# ---------------------------------------------------------------------------


def _tiny_config(solver: str) -> RunConfig:
    return RunConfig(solver=solver, pop_size=10, generations=5, seed=42)


def test_solve_nsga2_returns_pareto_front() -> None:
    prob = _make_problem()
    fronts = solve(prob, _tiny_config("nsga2"))
    assert len(fronts) == 1
    front = fronts[0]
    assert front.solver == SolverType.NSGA2
    assert front.nns >= 1


def test_solve_nrga_returns_pareto_front() -> None:
    prob = _make_problem()
    fronts = solve(prob, _tiny_config("nrga"))
    assert len(fronts) == 1
    assert fronts[0].solver == SolverType.NRGA
    assert fronts[0].nns >= 1


def test_solve_both_returns_two_fronts() -> None:
    prob = _make_problem()
    fronts = solve(prob, _tiny_config("both"))
    assert len(fronts) == 2
    solvers = {f.solver for f in fronts}
    assert solvers == {SolverType.NSGA2, SolverType.NRGA}


def test_solve_solutions_have_correct_assignment_count() -> None:
    prob = _make_problem()
    fronts = solve(prob, _tiny_config("nsga2"))
    for sol in fronts[0].solutions:
        assert sol.n_assignments == prob.n_vacancies


def test_solve_reproducible_with_seed() -> None:
    prob = _make_problem()
    cfg = RunConfig(solver="nsga2", pop_size=10, generations=5, seed=7)
    f1 = solve(prob, cfg)
    f2 = solve(prob, cfg)
    zs1 = [(s.z1, s.z2) for s in f1[0].solutions]
    zs2 = [(s.z1, s.z2) for s in f2[0].solutions]
    assert zs1 == pytest.approx(zs2)


def test_solve_cpu_time_recorded() -> None:
    prob = _make_problem()
    fronts = solve(prob, _tiny_config("nsga2"))
    assert fronts[0].cpu_time_sec > 0
