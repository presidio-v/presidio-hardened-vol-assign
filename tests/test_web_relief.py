"""Tests for the relief-allocation demo scenarios (published and repaired model).

These scenarios put the audited relief-allocation model on the demo page: the
evolutionary fronts next to an exact reference. The tests pin the instance
generator (deterministic, inside the model's valid ranges), the payload the
page consumes, and the audit claims the page makes (the exact row really is at
least as good; the repaired exact row never overfills a centre).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from presidio_vol_assign.allocation.load_coupling import free_capacity
from presidio_vol_assign.allocation.models import (
    DisabilityStatus,
    HazardLevel,
    InjuryLevel,
    LivingStatus,
    RoadCondition,
)
from presidio_vol_assign.web import relief, runner
from presidio_vol_assign.web.runner import build_request
from presidio_vol_assign.web.scenarios import AREA_KM, generate_instance, get_scenario

RELIEF_IDS = ["relief-published", "relief-repaired"]


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", RELIEF_IDS)
def test_generation_is_deterministic(scenario_id: str) -> None:
    scenario = get_scenario(scenario_id)
    knobs = scenario.resolve_knobs({})
    first = generate_instance(scenario, knobs, seed=11)
    second = generate_instance(scenario, knobs, seed=11)
    assert first.unit_points == second.unit_points
    assert first.site_points == second.site_points
    assert first.summary == second.summary
    assert first.problem.travel == second.problem.travel
    assert first.problem.people == second.problem.people

    other = generate_instance(scenario, knobs, seed=12)
    assert other.unit_points != first.unit_points


@pytest.mark.parametrize("scenario_id", RELIEF_IDS)
@pytest.mark.parametrize("n_people,n_centers", [(30, 2), (150, 5), (300, 10)])
def test_generated_instances_are_valid(scenario_id: str, n_people: int, n_centers: int) -> None:
    """Every field inside the ranges the allocation CSV validator enforces."""
    scenario = get_scenario(scenario_id)
    knobs = scenario.resolve_knobs({"n_people": n_people, "n_centers": n_centers})
    problem = generate_instance(scenario, knobs, seed=5).problem

    assert problem.n_people == n_people
    assert problem.n_centers == n_centers
    assert problem.n_dir == n_people // 3
    assert 0 < problem.n_dir < problem.n_people  # ATRes Eq. 15
    assert len(problem.travel) == n_people * n_centers
    assert len({p.person_id for p in problem.people}) == n_people

    for person in problem.people:
        assert 0.0 <= person.age <= 120.0
        assert 0.0 <= person.infrastructure_damage_level <= 100.0
        assert 0.0 <= person.resource_time_remaining <= 48.0
        assert isinstance(person.disability_status, DisabilityStatus)
        assert isinstance(person.injury_level, InjuryLevel)
        assert isinstance(person.living_status, LivingStatus)
    for centre in problem.centers:
        assert 0.0 <= centre.center_occupancy_rate <= 100.0
        assert 0.0 <= centre.resource_depletion_rate <= 100.0
    for travel in problem.travel.values():
        assert 0.0 <= travel.travel_duration <= 180.0
        assert isinstance(travel.road_condition, RoadCondition)
        assert isinstance(travel.possible_hazard, HazardLevel)


def test_travel_time_follows_map_distance() -> None:
    """The map and the TIL objective must tell the same story."""
    scenario = get_scenario("relief-published")
    instance = generate_instance(scenario, scenario.resolve_knobs({}), seed=3)
    unit = instance.unit_points[0]
    sites = instance.site_points
    durations = [instance.problem.travel[(unit["id"], s["id"])].travel_duration for s in sites]
    distances = [np.hypot(unit["x"] - s["x"], unit["y"] - s["y"]) for s in sites]
    assert int(np.argmin(durations)) == int(np.argmin(distances))
    for point in instance.unit_points + sites:
        assert 0.0 <= point["x"] <= AREA_KM and 0.0 <= point["y"] <= AREA_KM


def test_published_model_has_no_capacity() -> None:
    scenario = get_scenario("relief-published")
    instance = generate_instance(scenario, scenario.resolve_knobs({}), seed=3)
    assert all(site["capacity"] is None for site in instance.site_points)
    assert instance.extras == {}


@pytest.mark.parametrize("kappa_f", [1.1, 1.5, 2.0])
@pytest.mark.parametrize("n_people,n_centers", [(30, 10), (150, 5), (300, 10)])
def test_repaired_hard_limits_are_always_feasible(
    kappa_f: float, n_people: int, n_centers: int
) -> None:
    scenario = get_scenario("relief-repaired")
    knobs = scenario.resolve_knobs(
        {"n_people": n_people, "n_centers": n_centers, "kappa_f": kappa_f}
    )
    instance = generate_instance(scenario, knobs, seed=9)
    limits = instance.extras["limits"]
    assert int(limits.sum()) >= instance.problem.n_dir
    assert instance.extras["kappa_f"] >= kappa_f
    assert [s["capacity"] for s in instance.site_points] == [int(v) for v in limits]


def test_kappa_f_knob_is_clamped() -> None:
    scenario = get_scenario("relief-repaired")
    knobs = scenario.resolve_knobs({"kappa_f": 50.0, "n_people": 10_000, "n_centers": 99})
    assert knobs["kappa_f"] == scenario.knob("kappa_f").maximum
    assert knobs["n_people"] == 300
    assert knobs["n_centers"] == 10


def test_simplex_lattice_covers_the_simplex() -> None:
    weights = relief.simplex_lattice(10)
    assert len(weights) == 66 + 1  # lattice plus the equal-weight point
    assert all(abs(sum(w) - 1.0) < 1e-9 and min(w) >= 0 for w in weights)
    assert (1 / 3, 1 / 3, 1 / 3) in weights


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def _run(scenario_id: str, **overrides) -> dict:
    body = {
        "scenario": scenario_id,
        "solver": "both",
        "generations": 15,
        "pop_size": 40,
        "seed": 4,
        "knobs": {"n_people": 60, "n_centers": 4},
    }
    body.update(overrides)
    return runner._solve(build_request(body).as_dict())


def _check_common_shape(payload: dict) -> None:
    assert payload["model"] == "allocation"
    assert [r["solver"] for r in payload["results"]] == ["nsga2", "nrga", "exact"]
    assert len(payload["objectives"]) == 3
    n_units, n_sites = len(payload["units"]), len(payload["sites"])
    n_dir = payload["summary"]["directed"]
    for result in payload["results"]:
        assert set(result["metrics"]) >= {"nns", "hv", "sm", "mid", "cpuTimeSec"}
        assert 0.0 <= result["metrics"]["hv"] <= 1.0
        assert result["solutions"]
        for solution in result["solutions"]:
            assert len(solution["objectives"]) == 3
            assert len(solution["alloc"]) == n_units
            assert all(-1 <= s < n_sites for s in solution["alloc"])
            assert sum(1 for s in solution["alloc"] if s >= 0) == n_dir
    json.dumps(payload)  # must be serialisable as-is


def test_published_payload_shape() -> None:
    payload = _run("relief-published")
    _check_common_shape(payload)
    exact = payload["results"][-1]
    assert exact["exact"]["method"] == "separable-weighted-sum"
    assert exact["exact"]["provenOptimal"] == exact["exact"]["weights"]
    assert all(s["optimal"] for s in exact["solutions"])
    assert "EXACT" in exact["note"]
    # No capacity in the published model, so no overload anywhere.
    assert all("overload" not in s for r in payload["results"] for s in r["solutions"])
    assert payload["cliHint"] == ""


def test_published_exact_front_is_mutually_non_dominated() -> None:
    payload = _run("relief-published")
    points = [tuple(s["objectives"]) for s in payload["results"][-1]["solutions"]]
    for a in points:
        for b in points:
            if a != b:
                assert not all(x <= y for x, y in zip(b, a)), (b, a)


def test_exact_equal_weight_point_beats_every_moea_point() -> None:
    """The audit's core claim: the exact rule is optimal on the weighted sum.

    Payload objectives are rounded to 6 decimals, hence the tolerance.
    """
    payload = _run("relief-published", generations=30)
    exact_best = min(sum(s["objectives"]) for s in payload["results"][-1]["solutions"])
    for result in payload["results"][:-1]:
        moea_best = min(sum(s["objectives"]) for s in result["solutions"])
        assert exact_best <= moea_best + 1e-5, result["solver"]


def test_repaired_payload_shape_reports_optimality_and_overload() -> None:
    payload = _run("relief-repaired", knobs={"n_people": 60, "n_centers": 4, "kappa_f": 1.2})
    _check_common_shape(payload)
    summary = payload["summary"]
    assert summary["kappaF"] == pytest.approx(1.2)
    assert summary["freePlaces"] >= summary["directed"]

    exact = payload["results"][-1]
    info = exact["exact"]
    assert info["method"] == "mip-weighted-sum"
    assert info["weights"] == len(relief.MIP_WEIGHTS)
    assert info["solved"] + info["skipped"] + info["failed"] == info["weights"]
    assert 0 <= info["provenOptimal"] <= info["solved"]
    assert info["timeLimitSec"] <= 5.0
    assert all(isinstance(s["optimal"], bool) for s in exact["solutions"])

    for result in payload["results"]:
        assert "overloaded" in result["metrics"]
        for solution in result["solutions"]:
            assert solution["overload"] >= 0.0
    # The hard-capacity reference never overfills a centre.
    assert exact["metrics"]["overloaded"] == 0
    assert all(s["overload"] == 0.0 for s in exact["solutions"])


def test_repaired_overload_matches_the_site_free_places() -> None:
    """The page's load table uses site capacity; it must agree with overload."""
    payload = _run("relief-repaired", knobs={"n_people": 60, "n_centers": 4, "kappa_f": 1.2})
    scenario = get_scenario("relief-repaired")
    request = build_request({"scenario": "relief-repaired", "seed": 4, "knobs": payload["knobs"]})
    instance = generate_instance(scenario, request.knobs, request.seed)
    free = free_capacity(instance.problem, instance.extras["capacity"])
    limits = [s["capacity"] for s in payload["sites"]]
    for result in payload["results"]:
        for solution in result["solutions"]:
            loads = np.bincount([s for s in solution["alloc"] if s >= 0], minlength=len(limits))
            expected = float(np.maximum(0.0, loads - free).sum())
            assert solution["overload"] == pytest.approx(expected, abs=1e-3)
            if result["solver"] == "exact":
                assert all(load <= cap for load, cap in zip(loads, limits))


def test_exact_row_is_memoised_across_solver_settings() -> None:
    runner._INSTANCE_CACHE.clear()
    first = _run("relief-repaired", solver="nsga2")
    second = _run("relief-repaired", solver="nrga", generations=20)
    assert [r["solver"] for r in first["results"]] == ["nsga2", "exact"]
    assert [r["solver"] for r in second["results"]] == ["nrga", "exact"]
    assert first["results"][-1] == second["results"][-1]


def test_same_request_gives_the_same_payload() -> None:
    runner._INSTANCE_CACHE.clear()
    first = _run("relief-published")
    runner._INSTANCE_CACHE.clear()
    second = _run("relief-published")
    for a, b in zip(first["results"], second["results"]):
        assert [s["objectives"] for s in a["solutions"]] == [
            s["objectives"] for s in b["solutions"]
        ]
        assert [s["alloc"] for s in a["solutions"]] == [s["alloc"] for s in b["solutions"]]


def test_evidence_is_declined_for_relief_scenarios(monkeypatch) -> None:
    monkeypatch.setenv("PVA_EVIDENCE_KEY", "x" * 64)
    payload = _run("relief-published", evidence=True)
    assert payload["evidence"]["available"] is False


def test_exact_is_not_a_requestable_solver() -> None:
    from presidio_vol_assign.web.runner import RunRejected

    with pytest.raises(RunRejected):
        build_request({"scenario": "relief-published", "solver": "exact"})


# ---------------------------------------------------------------------------
# Static build
# ---------------------------------------------------------------------------


def test_relief_grid_is_small() -> None:
    from presidio_vol_assign.web import static_build as sb

    points, values = sb.plan("compact")
    relief_points = [p for p in points if p.scenario_id in RELIEF_IDS]
    assert 0 < len(relief_points) <= 60
    assert values["relief-repaired"][2] == [1.2, 1.5, 2.0]
    for point in relief_points:
        assert point.knobs["n_people"] <= 300 and point.knobs["n_centers"] <= 10


@pytest.mark.slow
def test_static_build_of_relief_scenarios_resolves(tmp_path, monkeypatch) -> None:
    """A tiny build of only the relief scenarios writes every advertised file."""
    import itertools

    from presidio_vol_assign.web import static_build as sb
    from presidio_vol_assign.web.scenarios import SCENARIOS

    monkeypatch.setattr(sb, "SCENARIOS", tuple(s for s in SCENARIOS if s.id in RELIEF_IDS))
    monkeypatch.setattr(
        sb,
        "_SCENARIO_GRID",
        {
            "relief-published": {"n_people": [30, 60], "n_centers": [3]},
            "relief-repaired": {"n_people": [30], "n_centers": [3], "kappa_f": [1.2, 2.0]},
        },
    )
    monkeypatch.setattr(sb, "SEEDS", [42])
    monkeypatch.setattr(sb, "STATIC_GENERATIONS", 5)

    summary = sb.build(tmp_path / "site", workers=1)
    site = tmp_path / "site"
    assert summary["runs"] == 4

    config = json.loads((site / "config.json").read_text())
    assert [s["id"] for s in config["scenarios"]] == RELIEF_IDS
    for scenario in config["scenarios"]:
        lists = [k["values"] for k in scenario["knobs"]]
        for combo in itertools.product(*(range(len(v)) for v in lists)):
            key = "-".join(map(str, combo)) + "__s0"
            payload = json.loads((site / "runs" / scenario["id"] / f"{key}.json").read_text())
            assert payload["gridKey"] == key
            assert payload["scenario"] == scenario["id"]
            assert payload["results"][-1]["solver"] == "exact"


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("scenario_id", RELIEF_IDS)
def test_relief_run_over_http(scenario_id: str) -> None:
    pytest.importorskip("fastapi", reason="requires the 'web' extra")
    from fastapi.testclient import TestClient

    from presidio_vol_assign.web.app import create_app

    with TestClient(create_app()) as client:
        listed = [s["id"] for s in client.get("/api/scenarios").json()["scenarios"]]
        assert scenario_id in listed
        response = client.post(
            "/api/run",
            json={
                "scenario": scenario_id,
                "solver": "nsga2",
                "seed": 5,
                "generations": 10,
                "knobs": {"n_people": 60, "n_centers": 3, "kappa_f": 1.5},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert [r["solver"] for r in body["results"]] == ["nsga2", "exact"]
    assert len(body["units"]) == 60 and len(body["sites"]) == 3


def test_static_build_worker_solves_mips_to_optimality(monkeypatch) -> None:
    """Pre-built pages must not depend on machine speed: the worker lifts MIP limits."""
    from presidio_vol_assign.web import relief, static_build

    monkeypatch.delenv(relief.EXACT_TO_OPTIMALITY_ENV, raising=False)
    assert relief._mip_limits() == (relief.MIP_TIME_LIMIT_SEC, relief.MIP_BUDGET_SEC)
    static_build._init_static_worker()
    try:
        per_mip, budget = relief._mip_limits()
        assert per_mip > relief.MIP_TIME_LIMIT_SEC and budget > relief.MIP_BUDGET_SEC
    finally:
        monkeypatch.delenv(relief.EXACT_TO_OPTIMALITY_ENV, raising=False)
