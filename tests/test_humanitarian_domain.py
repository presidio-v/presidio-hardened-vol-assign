"""Tests for the humanitarian allocation domain and its engine integration."""

from __future__ import annotations

from presidio_vol_assign.domains import HumanitarianDomain, get_domain
from presidio_vol_assign.engine import run
from presidio_vol_assign.models import Center, HumanitarianProblem, Person, RunConfig


def _make_problem() -> HumanitarianProblem:
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


def _cfg(solver: str = "nsga2") -> RunConfig:
    return RunConfig(solver=solver, pop_size=12, generations=6, seed=42)


# ---------------------------------------------------------------------------
# Registry + metadata
# ---------------------------------------------------------------------------


def test_get_domain_humanitarian() -> None:
    d = get_domain("humanitarian")
    assert isinstance(d, HumanitarianDomain)
    assert d.n_objectives == 3
    assert d.objective_names == ("z1", "z2", "z3")
    assert d.reference_point == (1.0, 1.0, 1.0)
    assert d.required_inputs == ("people", "centers")


# ---------------------------------------------------------------------------
# Encoding / evaluation hooks
# ---------------------------------------------------------------------------


def test_precompute_pairs_cover_all_person_center() -> None:
    d = HumanitarianDomain()
    prob = _make_problem()
    cache = d.precompute(prob)
    assert len(cache.pairs) == prob.n_people * prob.n_centers
    assert d._n_centers == prob.n_centers


def test_init_individual_in_range() -> None:
    d = HumanitarianDomain()
    prob = _make_problem()
    d.precompute(prob)
    ind = d.init_individual(prob, list)
    assert len(ind) == prob.n_people
    assert all(0 <= g < prob.n_centers for g in ind)


def test_evaluate_returns_three_objectives_in_range() -> None:
    d = HumanitarianDomain()
    prob = _make_problem()
    cache = d.precompute(prob)
    obj = d.evaluate([0, 1, 0, 1], cache, prob)
    assert len(obj) == 3
    assert all(0.0 <= z <= 1.0 for z in obj)


def test_overcrowding_rises_when_everyone_piles_into_one_centre() -> None:
    # Tight capacities so concentration actually exceeds a centre's capacity.
    people = [
        Person("P1", 9.0, 2.0, 3, {"C1": 5.0, "C2": 30.0}),
        Person("P2", 4.0, 7.0, 3, {"C1": 12.0, "C2": 8.0}),
        Person("P3", 7.5, 4.0, 3, {"C1": 25.0, "C2": 4.0}),
    ]
    centers = [
        Center("C1", capacity=6, service_level=8.0, road_accessibility=7.0),
        Center("C2", capacity=6, service_level=5.5, road_accessibility=4.0),
    ]
    prob = HumanitarianProblem(people=people, centers=centers)
    d = HumanitarianDomain()
    cache = d.precompute(prob)
    z3_concentrated = d.evaluate([0, 0, 0], cache, prob)[2]  # all in C1 (9/6 = 1.5)
    z3_spread = d.evaluate([0, 0, 1], cache, prob)[2]  # 6/6 and 3/6
    assert z3_concentrated > z3_spread


# ---------------------------------------------------------------------------
# Full solve via the engine
# ---------------------------------------------------------------------------


def test_solve_produces_three_objective_front() -> None:
    d = HumanitarianDomain()
    prob = _make_problem()
    fronts = run(prob, _cfg("both"), d)
    assert len(fronts) == 2
    for front in fronts:
        assert front.nns >= 1
        for sol in front.solutions:
            assert sol.n_objectives == 3
            assert sol.n_assignments == prob.n_people


def test_solve_reproducible_with_seed() -> None:
    d = HumanitarianDomain()
    prob = _make_problem()
    f1 = run(prob, _cfg("nsga2"), d)
    f2 = run(prob, _cfg("nsga2"), d)
    pts1 = sorted(s.objectives for s in f1[0].solutions)
    pts2 = sorted(s.objectives for s in f2[0].solutions)
    assert pts1 == pts2


def test_assignment_row_schema() -> None:
    d = HumanitarianDomain()
    prob = _make_problem()
    fronts = run(prob, _cfg("nsga2"), d)
    sol = fronts[0].solutions[0]
    row = d.assignment_row(sol.assignments[0])
    assert set(row) == set(d.assignment_fieldnames)
