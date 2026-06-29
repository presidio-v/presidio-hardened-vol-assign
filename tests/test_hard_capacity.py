"""Tests for the humanitarian hard-capacity / transport-limit constraint mode."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from presidio_vol_assign.cli import app
from presidio_vol_assign.domains import HumanitarianDomain, get_domain
from presidio_vol_assign.engine import run
from presidio_vol_assign.metrics import fronts_signature
from presidio_vol_assign.models import Center, HumanitarianProblem, Person, RunConfig

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def _contended_problem() -> HumanitarianProblem:
    """6 people who all prefer C1 (nearest); C1+C2 are small, C3 absorbs overflow.

    P0/P1 are immobile (mobility 0) and far from everything but C1.
    """
    people = [
        Person(
            f"P{i}",
            vulnerability=float(i),
            mobility=(0.0 if i < 2 else 9.0),
            group_size=1,
            distances={"C1": 1.0, "C2": 50.0, "C3": 80.0},
        )
        for i in range(6)
    ]
    centers = [
        Center("C1", capacity=2, service_level=8.0, road_accessibility=7.0),
        Center("C2", capacity=2, service_level=6.0, road_accessibility=6.0),
        Center("C3", capacity=4, service_level=5.0, road_accessibility=5.0),
    ]
    return HumanitarianProblem(people=people, centers=centers)


_CAPS = {"C1": 2, "C2": 2, "C3": 4}


def _cfg() -> RunConfig:
    return RunConfig(solver="nsga2", pop_size=20, generations=15, seed=1)


# ---------------------------------------------------------------------------
# Defaults / construction
# ---------------------------------------------------------------------------


def test_default_domain_is_soft() -> None:
    d = get_domain("humanitarian")
    assert isinstance(d, HumanitarianDomain)
    assert d._hard_capacity is False


def test_hard_domain_flags() -> None:
    d = HumanitarianDomain(hard_capacity=True, max_distance=10.0, mobility_threshold=3.0)
    assert d._hard_capacity and d._max_distance == 10.0 and d._mobility_threshold == 3.0


# ---------------------------------------------------------------------------
# Repair guarantees
# ---------------------------------------------------------------------------


def test_repair_respects_capacity_directly() -> None:
    d = HumanitarianDomain(hard_capacity=True, max_distance=10.0)
    prob = _contended_problem()
    cache = d.precompute(prob)
    # Worst-case genome: everyone wants C1.
    repaired = d._repair([0, 0, 0, 0, 0, 0], cache)
    loads = [0, 0, 0]
    for cj in repaired:
        loads[cj] += 1
    assert all(loads[i] <= cache.capacities[i] for i in range(3))
    # Immobile P0, P1 (allowed only C1 within 10km) are placed at C1.
    assert repaired[0] == 0 and repaired[1] == 0


def test_hard_capacity_enforced_across_front() -> None:
    d = HumanitarianDomain(hard_capacity=True, max_distance=10.0)
    fronts = run(_contended_problem(), _cfg(), d)
    for sol in fronts[0].solutions:
        load = {"C1": 0, "C2": 0, "C3": 0}
        for a in sol.assignments:
            load[a.center_id] += 1
        assert all(load[c] <= _CAPS[c] for c in _CAPS)


def test_transport_limit_enforced_across_front() -> None:
    d = HumanitarianDomain(hard_capacity=True, max_distance=10.0, mobility_threshold=3.0)
    fronts = run(_contended_problem(), _cfg(), d)
    for sol in fronts[0].solutions:
        for a in sol.assignments:
            if a.person_id in ("P0", "P1"):  # immobile → only C1 is within 10 km
                assert a.center_id == "C1"


def test_hard_mode_is_deterministic() -> None:
    d = HumanitarianDomain(hard_capacity=True, max_distance=10.0)
    prob = _contended_problem()
    assert fronts_signature(run(prob, _cfg(), d)) == fronts_signature(run(prob, _cfg(), d))


def test_soft_default_can_overcrowd() -> None:
    # Without hard mode the same contended instance is free to overcrowd C1.
    fronts = run(_contended_problem(), _cfg(), get_domain("humanitarian"))
    overcrowded = False
    for sol in fronts[0].solutions:
        load = {"C1": 0, "C2": 0, "C3": 0}
        for a in sol.assignments:
            load[a.center_id] += 1
        if any(load[c] > _CAPS[c] for c in _CAPS):
            overcrowded = True
    assert overcrowded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_hard_capacity_runs(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "assign",
            "--model",
            "humanitarian",
            "--people",
            str(FIXTURES / "people_valid.csv"),
            "--centers",
            str(FIXTURES / "centers_valid.csv"),
            "--hard-capacity",
            "--max-distance",
            "60",
            "--solver",
            "nsga2",
            "--pop-size",
            "10",
            "--generations",
            "4",
            "--seed",
            "1",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert "hard-capacity" in result.output
    assert list(tmp_path.glob("pareto_*.csv"))


def test_cli_hard_capacity_rejects_ed_staffing(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "assign",
            "--model",
            "ed-staffing",
            "--volunteers",
            str(FIXTURES / "volunteers_valid.csv"),
            "--eds",
            str(FIXTURES / "eds_valid.csv"),
            "--hard-capacity",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
