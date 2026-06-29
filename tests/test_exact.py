"""Tests for the exact weighted-sum baseline solver (SolverType.EXACT)."""

from __future__ import annotations

import pytest

from presidio_vol_assign.domains import get_domain
from presidio_vol_assign.engine import run
from presidio_vol_assign.metrics import compute_metrics, fronts_signature
from presidio_vol_assign.models import (
    Center,
    HumanitarianProblem,
    Person,
    ProblemInstance,
    RunConfig,
    SkillType,
    SolverType,
    Vacancy,
    Volunteer,
)


def _hum_problem() -> HumanitarianProblem:
    people = [
        Person("P1", 9.0, 2.0, 3, {"C1": 5.0, "C2": 30.0}),
        Person("P2", 4.0, 7.0, 1, {"C1": 12.0, "C2": 8.0}),
        Person("P3", 7.5, 4.0, 2, {"C1": 25.0, "C2": 4.0}),
        Person("P4", 2.0, 9.0, 1, {"C1": 8.0, "C2": 15.0}),
    ]
    centers = [
        Center("C1", capacity=6, service_level=8.0, road_accessibility=7.0),
        Center("C2", capacity=6, service_level=5.5, road_accessibility=4.0),
    ]
    return HumanitarianProblem(people=people, centers=centers)


def _ed_problem() -> ProblemInstance:
    vols = [
        Volunteer("V1", SkillType.TRIAGE, 8.0, {"ED1": 5.0, "ED2": 12.0}, 7.0),
        Volunteer("V2", SkillType.TRIAGE, 6.5, {"ED1": 3.0, "ED2": 8.5}, 5.0),
        Volunteer("V3", SkillType.ER_NURSE, 9.0, {"ED1": 15.0, "ED2": 4.0}, 8.0),
        Volunteer("V4", SkillType.ER_NURSE, 7.0, {"ED1": 9.0, "ED2": 6.0}, 6.0),
    ]
    vacs = [
        Vacancy("ED1", SkillType.TRIAGE, 40, 8.0),
        Vacancy("ED2", SkillType.ER_NURSE, 25, 6.5),
    ]
    return ProblemInstance(volunteers=vols, vacancies=vacs)


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


def test_base_domain_has_no_exact_baseline_by_default() -> None:
    from presidio_vol_assign.domains.base import Domain

    assert Domain.exact_baseline_population(None, None, None, list) is None


def test_engine_runs_exact_humanitarian() -> None:
    d = get_domain("humanitarian")
    fronts = run(_hum_problem(), RunConfig(solver="exact"), d)
    assert fronts[0].solver == SolverType.EXACT
    m = compute_metrics(fronts[0])
    assert m.nns >= 1
    for s in fronts[0].solutions:
        assert len(s.objectives) == 3 and all(0.0 <= v <= 1.0 for v in s.objectives)


def test_engine_runs_exact_ed() -> None:
    d = get_domain("ed-staffing")
    fronts = run(_ed_problem(), RunConfig(solver="exact"), d)
    assert fronts[0].solver == SolverType.EXACT
    assert compute_metrics(fronts[0]).nns >= 1


def test_exact_is_deterministic() -> None:
    d = get_domain("humanitarian")
    prob = _hum_problem()
    f1 = run(prob, RunConfig(solver="exact"), d)
    f2 = run(prob, RunConfig(solver="exact"), d)
    assert fronts_signature(f1) == fronts_signature(f2)


# ---------------------------------------------------------------------------
# Optimality — the exact solver must hit the true scalarised optimum
# ---------------------------------------------------------------------------


def test_humanitarian_exact_attains_z1_ideal() -> None:
    # The (w1=1, w2=0, w3=0) vertex minimises total fairness; with slack capacity
    # each person independently takes its min-fairness centre, so the front's best
    # z1 must equal the mean of per-person minimum fairness.
    d = get_domain("humanitarian")
    prob = _hum_problem()
    cache = d.precompute(prob)
    ideal_z1 = (
        sum(
            min(cache.pairs[(pi, cj)][0] for cj in range(prob.n_centers))
            for pi in range(prob.n_people)
        )
        / prob.n_people
    )

    fronts = run(prob, RunConfig(solver="exact"), d)
    best_z1 = min(s.objectives[0] for s in fronts[0].solutions)
    assert best_z1 == pytest.approx(ideal_z1, abs=1e-6)


def test_ed_exact_attains_min_importance() -> None:
    # Brute-force the optimal type-feasible matching for pure importance (w=(1,0)):
    # ED1 takes the lowest-importance triage volunteer, ED2 the lowest ER nurse.
    d = get_domain("ed-staffing")
    prob = _ed_problem()
    cache = d.precompute(prob)
    triage = [vi for vi, v in enumerate(prob.volunteers) if v.skill_type == SkillType.TRIAGE]
    ernurse = [vi for vi, v in enumerate(prob.volunteers) if v.skill_type == SkillType.ER_NURSE]
    best_ed1 = min(cache[(vi, 0)][0] for vi in triage)
    best_ed2 = min(cache[(vi, 1)][0] for vi in ernurse)
    min_mean_importance = (best_ed1 + best_ed2) / 2

    fronts = run(prob, RunConfig(solver="exact"), d)
    best_z1 = min(s.objectives[0] for s in fronts[0].solutions)
    assert best_z1 == pytest.approx(min_mean_importance, abs=1e-6)
