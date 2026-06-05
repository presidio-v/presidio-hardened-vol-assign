"""Tests for the FIS-perturbation sensitivity analysis."""

from __future__ import annotations

import pytest

from presidio_vol_assign.domains import EDStaffingDomain, HumanitarianDomain
from presidio_vol_assign.engine import run
from presidio_vol_assign.metrics import compute_metrics
from presidio_vol_assign.models import (
    Center,
    HumanitarianProblem,
    Person,
    ProblemInstance,
    RunConfig,
    SkillType,
    Vacancy,
    Volunteer,
)
from presidio_vol_assign.sensitivity import parse_factors, run_sensitivity


def _ed_problem() -> ProblemInstance:
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


def _hum_problem() -> HumanitarianProblem:
    people = [
        Person("P1", 9.0, 2.0, 2, {"C1": 5.0, "C2": 30.0}),
        Person("P2", 4.0, 7.0, 1, {"C1": 12.0, "C2": 8.0}),
        Person("P3", 7.5, 4.0, 2, {"C1": 25.0, "C2": 4.0}),
    ]
    centers = [
        Center("C1", 10, 8.0, 7.0),
        Center("C2", 10, 5.5, 4.0),
    ]
    return HumanitarianProblem(people=people, centers=centers)


# ---------------------------------------------------------------------------
# perturb()
# ---------------------------------------------------------------------------


def test_ed_perturb_zero_is_identity() -> None:
    d = EDStaffingDomain()
    cache = d.precompute(_ed_problem())
    assert d.perturb(cache, 0.0) == cache


def test_ed_perturb_scales_and_clips() -> None:
    d = EDStaffingDomain()
    cache = d.precompute(_ed_problem())
    up = d.perturb(cache, 0.5)
    for key, entry in cache.items():
        if entry is None:
            assert up[key] is None
        else:
            imp, pref = entry
            assert up[key][0] == pytest.approx(min(imp * 1.5, 1.0))
            assert all(0.0 <= v <= 1.0 for v in up[key])


def test_hum_perturb_zero_is_identity() -> None:
    d = HumanitarianDomain()
    cache = d.precompute(_hum_problem())
    z = d.perturb(cache, 0.0)
    assert z.pairs == cache.pairs
    assert (z.util_lut == cache.util_lut).all()


# ---------------------------------------------------------------------------
# engine.run with an explicit (perturbed) cache
# ---------------------------------------------------------------------------


def test_run_with_explicit_cache_matches_factor_zero() -> None:
    d = HumanitarianDomain()
    prob = _hum_problem()
    cfg = RunConfig(solver="nsga2", pop_size=10, generations=5, seed=42)
    normal = run(prob, cfg, d)
    via_cache = run(prob, cfg, d, cache=d.perturb(d.precompute(prob), 0.0))
    assert [s.objectives for s in normal[0].solutions] == [
        s.objectives for s in via_cache[0].solutions
    ]


# ---------------------------------------------------------------------------
# run_sensitivity
# ---------------------------------------------------------------------------


def test_run_sensitivity_rows_and_factor_zero_matches_normal() -> None:
    d = HumanitarianDomain()
    prob = _hum_problem()
    cfg = RunConfig(solver="both", pop_size=10, generations=5, seed=42)
    rows = run_sensitivity(d, prob, cfg, factors=(-0.1, 0.0, 0.1))
    # 3 factors x 2 solvers
    assert len(rows) == 6
    assert {r.solver for r in rows} == {"nsga2", "nrga"}
    assert {round(r.factor, 3) for r in rows} == {-0.1, 0.0, 0.1}

    # factor 0.0 nsga2 row equals a plain run's metrics
    plain = compute_metrics(
        run(prob, RunConfig(solver="nsga2", pop_size=10, generations=5, seed=42), d)[0]
    )
    zero = next(r for r in rows if r.factor == 0.0 and r.solver == "nsga2")
    assert zero.hv == pytest.approx(plain.hv)
    assert zero.mid == pytest.approx(plain.mid)
    assert zero.nns == plain.nns


def test_sensitivity_changes_metrics() -> None:
    d = HumanitarianDomain()
    prob = _hum_problem()
    cfg = RunConfig(solver="nsga2", pop_size=12, generations=6, seed=42)
    rows = run_sensitivity(d, prob, cfg, factors=(-0.2, 0.2))
    mids = {r.factor: r.mid for r in rows}
    # Perturbing FIS outputs up vs down shifts the mean ideal distance.
    assert mids[-0.2] != pytest.approx(mids[0.2])


# ---------------------------------------------------------------------------
# parse_factors
# ---------------------------------------------------------------------------


def test_parse_factors_valid() -> None:
    assert parse_factors("-0.2,-0.1,0,0.1,0.2") == (-0.2, -0.1, 0.0, 0.1, 0.2)
    assert parse_factors("0.05") == (0.05,)


def test_parse_factors_invalid() -> None:
    with pytest.raises(ValueError):
        parse_factors("a,b")
    with pytest.raises(ValueError):
        parse_factors("")
