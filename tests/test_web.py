"""Tests for the optional demo web GUI (``pva serve``).

Skipped in full when the ``web`` extra is not installed, so the core test suite
still passes on a minimal environment.
"""

from __future__ import annotations

import pytest

from presidio_vol_assign.validation import (
    _check_assignment_feasibility,
    _check_capacity_feasibility,
)
from presidio_vol_assign.web import runner
from presidio_vol_assign.web.runner import LIMITS, RunRejected, build_request
from presidio_vol_assign.web.scenarios import (
    SCENARIOS,
    SCENARIOS_BY_ID,
    generate_instance,
    get_scenario,
)

pytest.importorskip("fastapi", reason="requires the 'web' extra")

from fastapi.testclient import TestClient  # noqa: E402

from presidio_vol_assign.web.app import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Scenarios and knobs
# ---------------------------------------------------------------------------


def test_three_presets_are_exposed() -> None:
    assert [s.id for s in SCENARIOS] == ["volunteers", "relief-centres", "last-mile"]


def test_last_mile_is_hard_capacity_humanitarian() -> None:
    last_mile = get_scenario("last-mile")
    soft = get_scenario("relief-centres")
    assert last_mile.model == soft.model == "humanitarian"
    assert last_mile.hard_capacity is True
    assert soft.hard_capacity is False


def test_every_scenario_names_one_objective_per_solver_objective() -> None:
    expected = {"ed-staffing": 2, "humanitarian": 3}
    for scenario in SCENARIOS:
        assert len(scenario.objectives) == expected[scenario.model]


def test_knobs_clamp_out_of_range_values() -> None:
    scenario = get_scenario("relief-centres")
    knobs = scenario.resolve_knobs({"n_people": 10_000, "n_centers": -5})
    assert knobs["n_people"] == scenario.knob("n_people").maximum
    assert knobs["n_centers"] == scenario.knob("n_centers").minimum


def test_knobs_ignore_unknown_keys_and_fill_defaults() -> None:
    scenario = get_scenario("relief-centres")
    knobs = scenario.resolve_knobs({"rm -rf": 1, "n_people": 40})
    assert "rm -rf" not in knobs
    assert knobs["n_people"] == 40
    assert knobs["n_centers"] == scenario.knob("n_centers").default


# ---------------------------------------------------------------------------
# Request validation and caps
# ---------------------------------------------------------------------------


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(RunRejected):
        build_request({"scenario": "../../etc/passwd"})


def test_disallowed_solver_is_rejected() -> None:
    # 'exact' is a valid library solver but deliberately not exposed by the demo.
    with pytest.raises(RunRejected):
        build_request({"scenario": "relief-centres", "solver": "exact"})


@pytest.mark.parametrize(
    "field,value",
    [("generations", 10_000), ("pop_size", 5), ("generations", 0), ("seed", -1)],
)
def test_out_of_range_solver_settings_are_rejected(field: str, value: int) -> None:
    with pytest.raises(RunRejected):
        build_request({"scenario": "relief-centres", field: value})


def test_size_caps_cannot_be_exceeded_by_a_handcrafted_request() -> None:
    request = build_request(
        {"scenario": "relief-centres", "knobs": {"n_people": 5_000, "n_centers": 500}}
    )
    assert request.knobs["n_people"] <= LIMITS.max_units
    assert request.knobs["n_centers"] <= LIMITS.max_sites


def test_defaults_are_within_caps_for_every_scenario() -> None:
    for scenario in SCENARIOS:
        request = build_request({"scenario": scenario.id})
        units = request.knobs.get("n_people", request.knobs.get("n_volunteers"))
        assert units <= LIMITS.max_units


# ---------------------------------------------------------------------------
# Generated instances are always feasible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_people", [10, 70, 150, 300])
@pytest.mark.parametrize("slack", [1.0, 1.5])
def test_humanitarian_instances_always_have_enough_capacity(n_people: int, slack: float) -> None:
    scenario = get_scenario("relief-centres")
    knobs = scenario.resolve_knobs({"n_people": n_people, "capacity_slack": slack})
    instance = generate_instance(scenario, knobs, seed=3)
    # Raises if total capacity cannot hold total demand.
    _check_capacity_feasibility(instance.problem.people, instance.problem.centers)


@pytest.mark.parametrize("n_volunteers,n_vacancies", [(6, 2), (6, 40), (60, 12), (200, 40)])
def test_ed_staffing_instances_always_satisfy_per_type_feasibility(
    n_volunteers: int, n_vacancies: int
) -> None:
    scenario = get_scenario("volunteers")
    knobs = scenario.resolve_knobs({"n_volunteers": n_volunteers, "n_vacancies": n_vacancies})
    instance = generate_instance(scenario, knobs, seed=5)
    # Raises if any skill type has fewer volunteers than vacancies of that type.
    _check_assignment_feasibility(instance.problem.volunteers, instance.problem.vacancies)


def test_generation_is_deterministic_for_a_given_seed() -> None:
    scenario = get_scenario("last-mile")
    knobs = scenario.resolve_knobs({})
    first = generate_instance(scenario, knobs, seed=17)
    second = generate_instance(scenario, knobs, seed=17)
    assert first.unit_points == second.unit_points
    assert first.site_points == second.site_points


def test_map_coordinates_cover_every_unit_and_site() -> None:
    for scenario in SCENARIOS:
        instance = generate_instance(scenario, scenario.resolve_knobs({}), seed=2)
        problem = instance.problem
        n_units = getattr(problem, "n_people", None) or problem.n_volunteers
        assert len(instance.unit_points) == n_units
        assert len(instance.site_points) >= 1
        for point in instance.unit_points + instance.site_points:
            assert 0.0 <= point["x"] <= 70.0
            assert 0.0 <= point["y"] <= 70.0


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("scenario_id", list(SCENARIOS_BY_ID))
def test_solve_returns_a_usable_front_for_every_scenario(scenario_id: str) -> None:
    request = build_request({"scenario": scenario_id, "generations": 10, "seed": 4})
    payload = runner._solve(request.as_dict())

    result = payload["results"][0]
    assert result["metrics"]["nns"] >= 1
    assert len(payload["objectives"]) == len(result["solutions"][0]["objectives"])

    n_sites = len(payload["sites"])
    for solution in result["solutions"]:
        assert len(solution["alloc"]) == len(payload["units"])
        assert all(-1 <= site_idx < n_sites for site_idx in solution["alloc"])


@pytest.mark.slow
def test_hard_capacity_never_overfills_a_centre() -> None:
    request = build_request({"scenario": "last-mile", "generations": 10, "seed": 8})
    payload = runner._solve(request.as_dict())
    capacities = [site["capacity"] for site in payload["sites"]]
    weights = [unit["weight"] for unit in payload["units"]]

    for solution in payload["results"][0]["solutions"]:
        loads = [0] * len(capacities)
        for unit_idx, site_idx in enumerate(solution["alloc"]):
            loads[site_idx] += weights[unit_idx]
        assert all(load <= cap for load, cap in zip(loads, capacities))


@pytest.mark.slow
def test_soft_capacity_allows_overfilling() -> None:
    """The contrast that the two humanitarian presets exist to show."""
    request = build_request(
        {
            "scenario": "relief-centres",
            "generations": 10,
            "seed": 8,
            "knobs": {"capacity_slack": 1.0},
        }
    )
    payload = runner._solve(request.as_dict())
    capacities = [site["capacity"] for site in payload["sites"]]
    weights = [unit["weight"] for unit in payload["units"]]

    overfilled = False
    for solution in payload["results"][0]["solutions"]:
        loads = [0] * len(capacities)
        for unit_idx, site_idx in enumerate(solution["alloc"]):
            loads[site_idx] += weights[unit_idx]
        if any(load > cap for load, cap in zip(loads, capacities)):
            overfilled = True
            break
    assert overfilled, "soft-capacity mode should be able to exceed a centre's capacity"


@pytest.mark.slow
def test_same_request_gives_the_same_objectives() -> None:
    request = build_request({"scenario": "relief-centres", "generations": 10, "seed": 21})
    first = runner._solve(request.as_dict())
    runner._INSTANCE_CACHE.clear()
    second = runner._solve(request.as_dict())
    assert [s["objectives"] for s in first["results"][0]["solutions"]] == [
        s["objectives"] for s in second["results"][0]["solutions"]
    ]


@pytest.mark.slow
def test_evidence_is_unavailable_without_a_signing_key(monkeypatch) -> None:
    monkeypatch.delenv("PVA_EVIDENCE_KEY", raising=False)
    monkeypatch.delenv("PVA_EVIDENCE_ED25519_KEY", raising=False)
    assert runner.evidence_available() is False

    request = build_request(
        {"scenario": "relief-centres", "generations": 10, "seed": 6, "evidence": True}
    )
    payload = runner._solve(request.as_dict())
    assert payload["evidence"]["available"] is False
    assert "record" not in payload["evidence"]


@pytest.mark.slow
def test_evidence_record_is_emitted_and_verifies(monkeypatch, tmp_path) -> None:
    import json

    from presidio_vol_assign.evidence import ALG_HMAC, load_trust_store, verify_record

    secret = "a" * 64
    monkeypatch.setenv("PVA_EVIDENCE_KEY", secret)
    monkeypatch.setenv("PVA_EVIDENCE_SIGNER", "demo-server")
    monkeypatch.delenv("PVA_EVIDENCE_ED25519_KEY", raising=False)
    assert runner.evidence_available() is True

    request = build_request(
        {
            "scenario": "relief-centres",
            "generations": 10,
            "seed": 6,
            "evidence": True,
            "knobs": {"n_people": 20, "n_centers": 3},
        }
    )
    payload = runner._solve(request.as_dict())
    assert payload["evidence"]["available"] is True

    trust_path = tmp_path / "trust.json"
    trust_path.write_text(json.dumps({"demo-server": {"alg": ALG_HMAC, "secret": secret}}))
    # Raises on any of: schema mismatch, float leak, hash mismatch, bad signature.
    verify_record(payload["evidence"]["record"], load_trust_store(trust_path))


@pytest.mark.slow
def test_evidence_leaves_no_temporary_files_behind(monkeypatch) -> None:
    """The generated CSVs are hashed, then removed — a public instance keeps nothing.

    ``tempfile`` caches its directory on first use, so this checks the real
    temporary directory rather than trying to redirect it via ``TMPDIR``.
    """
    import tempfile
    from pathlib import Path

    monkeypatch.setenv("PVA_EVIDENCE_KEY", "b" * 64)
    tmp_root = Path(tempfile.gettempdir())
    before = set(tmp_root.glob("pva-web-evidence-*"))

    request = build_request(
        {
            "scenario": "relief-centres",
            "generations": 10,
            "seed": 6,
            "evidence": True,
            "knobs": {"n_people": 20, "n_centers": 3},
        }
    )
    payload = runner._solve(request.as_dict())

    assert payload["evidence"]["available"] is True, "test would be vacuous without a key"
    assert set(tmp_root.glob("pva-web-evidence-*")) == before


class _StubProcess:
    """Stands in for a worker process so the kill path can be asserted."""

    def __init__(self) -> None:
        self.killed = False

    def kill(self) -> None:
        self.killed = True


class _StubExecutor:
    """Executor whose futures always time out.

    Racing a real solver against a short deadline is nondeterministic and, on a
    two-core CI runner, expensive enough to stall the suite. The behaviour worth
    pinning is ours — raise RunTimeout, kill the workers, drop the executor —
    not concurrent.futures' internals.
    """

    def __init__(self) -> None:
        self.processes = {1: _StubProcess(), 2: _StubProcess()}
        self._processes = self.processes
        self.shutdown_calls: list[bool] = []

    def submit(self, *_args, **_kwargs):
        from concurrent.futures import TimeoutError as FutureTimeoutError

        class _Future:
            def result(self, timeout=None):
                raise FutureTimeoutError()

        return _Future()

    def shutdown(self, wait=True, **_kwargs):
        self.shutdown_calls.append(wait)


def test_a_run_that_overruns_its_timeout_kills_its_workers(monkeypatch) -> None:
    """The wall-clock backstop must terminate the workers, not just stop waiting."""
    pool = runner.SolverPool()
    stub = _StubExecutor()
    # Install the stub exactly as _ensure_executor would, so _kill() finds it.
    pool._executor = stub

    request = build_request({"scenario": "relief-centres", "generations": 10})
    with pytest.raises(runner.RunTimeout):
        pool.run(request)

    assert all(proc.killed for proc in stub.processes.values()), "workers were not killed"
    assert stub.shutdown_calls == [False], "timed-out pool should not be waited on"
    # The executor is dropped so the next request rebuilds a clean pool.
    assert pool._executor is None


def test_a_kill_that_raises_still_drops_the_pool(monkeypatch) -> None:
    """Recovery must not depend on the workers dying cleanly."""
    pool = runner.SolverPool()
    stub = _StubExecutor()
    for proc in stub.processes.values():
        proc.kill = lambda: (_ for _ in ()).throw(OSError("no such process"))
    # Install the stub exactly as _ensure_executor would, so _kill() finds it.
    pool._executor = stub

    with pytest.raises(runner.RunTimeout):
        pool.run(build_request({"scenario": "relief-centres", "generations": 10}))
    assert pool._executor is None


@pytest.mark.slow
def test_a_normal_run_goes_through_the_worker_pool() -> None:
    """The real pool path, exercised once at the smallest size the demo allows."""
    pool = runner.SolverPool()
    try:
        payload = pool.run(
            build_request(
                {
                    "scenario": "relief-centres",
                    "generations": 5,
                    "seed": 3,
                    "knobs": {"n_people": 10, "n_centers": 2},
                }
            )
        )
        assert payload["results"][0]["metrics"]["nns"] >= 1
    finally:
        pool.shutdown()


def test_instance_cache_reuses_the_precompute_across_solver_settings() -> None:
    runner._INSTANCE_CACHE.clear()
    base = {"scenario": "volunteers", "knobs": {"n_volunteers": 8, "n_vacancies": 2, "n_eds": 1}}
    first = runner._prepare(build_request({**base, "generations": 10}))
    second = runner._prepare(build_request({**base, "generations": 20, "solver": "nrga"}))
    assert first[2] is second[2]

    other_seed = runner._prepare(build_request({**base, "seed": 999}))
    assert other_seed[2] is not first[2]


# ---------------------------------------------------------------------------
# Static build (`pva build-demo`)
# ---------------------------------------------------------------------------


def test_grid_covers_every_scenario_and_seed() -> None:
    from presidio_vol_assign.web import static_build as sb

    points, values = sb.plan("compact")
    covered = {p.scenario_id for p in points}
    assert covered == set(SCENARIOS_BY_ID)
    for scenario in SCENARIOS:
        assert len(values[scenario.id]) == len(scenario.knobs)
    seeds = {p.seed for p in points}
    assert seeds == set(sb.SEEDS)


def test_grid_keys_are_unique() -> None:
    from presidio_vol_assign.web import static_build as sb

    points, _ = sb.plan("compact")
    keys = [(p.scenario_id, p.key) for p in points]
    assert len(keys) == len(set(keys))


def test_grid_key_format_matches_the_frontend_contract() -> None:
    """app.js builds `indices.join("-") + "__s" + seedIndex`; keep them in step.

    The key is addressed by slider *position* precisely so that no float
    formatting has to agree between Python and JavaScript.
    """
    from presidio_vol_assign.web.static_build import GridPoint

    point = GridPoint(
        scenario_id="relief-centres",
        knob_indices=(0, 2, 1, 3),
        seed_index=1,
        knobs={},
        seed=7,
    )
    assert point.key == "0-2-1-3__s1"
    assert point.rel_path == "runs/relief-centres/0-2-1-3__s1.json"


def test_grid_values_are_all_accepted_by_their_knob() -> None:
    """A grid value the knob would clamp would produce an unreachable file."""
    from presidio_vol_assign.web.static_build import plan

    _points, values = plan("full")
    for scenario in SCENARIOS:
        for knob, options in zip(scenario.knobs, values[scenario.id]):
            for value in options:
                assert knob.clamp(value) == value, (scenario.id, knob.key, value)


def test_every_planned_point_has_knobs_for_every_knob() -> None:
    from presidio_vol_assign.web.static_build import plan

    points, _ = plan("compact")
    for point in points[:200]:
        scenario = get_scenario(point.scenario_id)
        assert set(point.knobs) == {k.key for k in scenario.knobs}
        assert len(point.knob_indices) == len(scenario.knobs)


def test_htaccess_caches_runs_but_not_the_grid_definition() -> None:
    """A cached config.json would leave visitors on a stale grid after a rebuild."""
    from presidio_vol_assign.web.static_build import _HTACCESS

    assert 'Header set Cache-Control "public, max-age=86400"' in _HTACCESS
    config_block = _HTACCESS.split('<Files "config.json">')[1]
    assert "no-cache" in config_block.split("</Files>")[0]
    assert "DEFLATE" in _HTACCESS and "application/json" in _HTACCESS


@pytest.mark.slow
def test_static_build_produces_a_servable_tree(tmp_path, monkeypatch) -> None:
    """One grid point per scenario, end to end, checking what the browser needs."""
    import json

    from presidio_vol_assign.web import static_build as sb

    tiny = {key: [values[0]] for key, values in sb._COMPACT_GRID.items()}
    monkeypatch.setattr(sb, "_COMPACT_GRID", tiny)
    monkeypatch.setattr(sb, "SEEDS", [42])
    # Keep the solve trivial: this test is about the emitted tree, not front quality.
    monkeypatch.setattr(sb, "STATIC_GENERATIONS", 5)

    summary = sb.build(tmp_path / "site", workers=1)
    site = tmp_path / "site"

    assert summary["runs"] == len(SCENARIOS)
    for name in ("index.html", "config.json", "app.js", "style.css", ".htaccess"):
        assert (site / name).is_file(), name

    # The marker must be flipped, or the hosted page would call a missing API.
    assert 'content="static"' in (site / "index.html").read_text()

    config = json.loads((site / "config.json").read_text())
    assert config["mode"] == "static"

    for scenario in config["scenarios"]:
        for knob in scenario["knobs"]:
            assert knob["values"], knob["key"]
        # The key the browser builds for the all-zero position.
        key = "-".join("0" for _ in scenario["knobs"]) + "__s0"
        payload_path = site / "runs" / scenario["id"] / f"{key}.json"
        assert payload_path.is_file(), payload_path

        payload = json.loads(payload_path.read_text())
        assert payload["gridKey"] == key
        assert [r["solver"] for r in payload["results"]] == ["nsga2", "nrga"]
        solution = payload["results"][0]["solutions"][0]
        assert len(solution["alloc"]) == len(payload["units"])
        assert len(solution["objectives"]) == len(payload["objectives"])


def test_live_index_html_carries_the_mode_marker() -> None:
    """`build` flips this marker; without it the static build would be silently live."""
    from presidio_vol_assign.web.static_build import STATIC_SRC

    index = (STATIC_SRC / "index.html").read_text(encoding="utf-8")
    assert 'name="pva-mode" content="live"' in index


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_scenarios_endpoint_describes_the_presets(client) -> None:
    body = client.get("/api/scenarios").json()
    assert [s["id"] for s in body["scenarios"]] == ["volunteers", "relief-centres", "last-mile"]
    assert body["limits"]["maxUnits"] == LIMITS.max_units
    for scenario in body["scenarios"]:
        assert scenario["knobs"] and scenario["objectives"]


def test_security_headers_are_applied(client) -> None:
    headers = client.get("/api/health").headers
    assert headers["content-security-policy"] == "default-src 'self'; frame-ancestors 'none'"
    assert headers["x-frame-options"] == "DENY"


def test_index_page_is_served(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "trade-off" in response.text


def test_static_assets_are_served(client) -> None:
    for path in ("/static/app.js", "/static/style.css"):
        assert client.get(path).status_code == 200


def test_run_rejects_an_unknown_scenario(client) -> None:
    response = client.post("/api/run", json={"scenario": "nope"})
    assert response.status_code == 422


def test_run_rejects_unknown_body_fields(client) -> None:
    response = client.post(
        "/api/run", json={"scenario": "relief-centres", "output_dir": "/etc/passwd"}
    )
    assert response.status_code == 422


@pytest.mark.slow
def test_run_end_to_end_over_http(client) -> None:
    response = client.post(
        "/api/run",
        json={
            "scenario": "relief-centres",
            "solver": "nsga2",
            "seed": 5,
            "generations": 10,
            "knobs": {"n_people": 30, "n_centers": 3},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scenario"] == "relief-centres"
    assert len(body["units"]) == 30
    assert len(body["sites"]) == 3
    assert body["results"][0]["metrics"]["nns"] >= 1
    assert body["cliHint"].startswith("pva ")
