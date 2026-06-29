"""Tests for the greedy baseline comparator (SolverType.GREEDY)."""

from __future__ import annotations

import pytest

from presidio_vol_assign.baselines import weight_simplex
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
        Person(
            "P1", vulnerability=9.0, mobility=2.0, group_size=3, distances={"C1": 5.0, "C2": 30.0}
        ),
        Person(
            "P2", vulnerability=4.0, mobility=7.0, group_size=1, distances={"C1": 12.0, "C2": 8.0}
        ),
        Person(
            "P3", vulnerability=7.5, mobility=4.0, group_size=2, distances={"C1": 25.0, "C2": 4.0}
        ),
        Person(
            "P4", vulnerability=2.0, mobility=9.0, group_size=1, distances={"C1": 8.0, "C2": 15.0}
        ),
    ]
    centers = [
        Center("C1", capacity=20, service_level=8.0, road_accessibility=7.0),
        Center("C2", capacity=15, service_level=5.5, road_accessibility=4.0),
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
        Vacancy("ED1", SkillType.TRIAGE, num_patients=40, emergency_level=8.0),
        Vacancy("ED2", SkillType.ER_NURSE, num_patients=25, emergency_level=6.5),
    ]
    return ProblemInstance(volunteers=vols, vacancies=vacs)


# ---------------------------------------------------------------------------
# weight_simplex
# ---------------------------------------------------------------------------


def test_weight_simplex_counts() -> None:
    assert len(weight_simplex(2, steps=8)) == 9  # C(9, 1)
    assert len(weight_simplex(3, steps=6)) == 28  # C(8, 2)


def test_weight_simplex_sums_to_one() -> None:
    for w in weight_simplex(3, steps=6):
        assert len(w) == 3
        assert sum(w) == pytest.approx(1.0)
        assert all(c >= 0.0 for c in w)


def test_weight_simplex_includes_vertices() -> None:
    w2 = weight_simplex(2, steps=8)
    assert (1.0, 0.0) in w2 and (0.0, 1.0) in w2


def test_weight_simplex_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        weight_simplex(0, steps=4)
    with pytest.raises(ValueError):
        weight_simplex(2, steps=0)


# ---------------------------------------------------------------------------
# baseline_population per domain
# ---------------------------------------------------------------------------


def test_base_domain_has_no_baseline_by_default() -> None:
    from presidio_vol_assign.domains.base import Domain

    # The abstract base's default hook returns None (no baseline available);
    # it ignores self, so calling it unbound with a dummy self is sufficient.
    assert Domain.baseline_population(None, None, None, list) is None


def test_humanitarian_baseline_population_feasible() -> None:
    d = get_domain("humanitarian")
    prob = _hum_problem()
    cache = d.precompute(prob)
    pop = d.baseline_population(prob, cache, list)
    assert len(pop) == len(weight_simplex(3, steps=6))
    for genome in pop:
        assert len(genome) == prob.n_people
        assert all(0 <= g < prob.n_centers for g in genome)


def test_ed_baseline_population_is_permutation() -> None:
    d = get_domain("ed-staffing")
    prob = _ed_problem()
    cache = d.precompute(prob)
    pop = d.baseline_population(prob, cache, list)
    assert len(pop) == len(weight_simplex(2, steps=8))
    for genome in pop:
        assert sorted(genome) == list(range(prob.n_volunteers))


# ---------------------------------------------------------------------------
# engine integration: --solver greedy
# ---------------------------------------------------------------------------


def test_engine_runs_greedy_humanitarian() -> None:
    d = get_domain("humanitarian")
    fronts = run(_hum_problem(), RunConfig(solver="greedy"), d)
    assert len(fronts) == 1
    assert fronts[0].solver == SolverType.GREEDY
    m = compute_metrics(fronts[0])
    assert m.nns >= 1
    assert 0.0 <= m.hv <= 1.0


def test_engine_runs_greedy_ed() -> None:
    d = get_domain("ed-staffing")
    fronts = run(_ed_problem(), RunConfig(solver="greedy"), d)
    assert fronts[0].solver == SolverType.GREEDY
    assert compute_metrics(fronts[0]).nns >= 1


def test_greedy_is_deterministic_without_seed() -> None:
    # No randomness in the baseline → identical fronts even with seed=None.
    d = get_domain("humanitarian")
    prob = _hum_problem()
    f1 = run(prob, RunConfig(solver="greedy", seed=None), d)
    f2 = run(prob, RunConfig(solver="greedy", seed=None), d)
    assert fronts_signature(f1) == fronts_signature(f2)
