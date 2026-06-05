"""Tests for the domain-adapter layer introduced in v0.2.0.

These verify the abstraction the humanitarian model will plug into, and confirm
the ED-staffing domain drives the generic engine identically to the legacy
``solve`` facade.
"""

from __future__ import annotations

import pytest

from presidio_vol_assign.domains import EDStaffingDomain, get_domain
from presidio_vol_assign.engine import ensure_creator_types, run
from presidio_vol_assign.models import (
    ProblemInstance,
    RunConfig,
    SkillType,
    Vacancy,
    Volunteer,
)
from presidio_vol_assign.solvers import solve


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
# Registry
# ---------------------------------------------------------------------------


def test_get_domain_returns_ed_staffing() -> None:
    domain = get_domain("ed-staffing")
    assert isinstance(domain, EDStaffingDomain)
    assert domain.name == "ed-staffing"


def test_get_domain_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        get_domain("does-not-exist")


# ---------------------------------------------------------------------------
# Objective-space metadata
# ---------------------------------------------------------------------------


def test_ed_domain_metadata() -> None:
    d = EDStaffingDomain()
    assert d.n_objectives == 2
    assert d.objective_names == ("z1", "z2")
    assert d.reference_point == (1.0, 1.0)
    assert d.ideal_point == (0.0, 0.0)
    assert d.weights == (-1.0, -1.0)


def test_ensure_creator_types_idempotent() -> None:
    d = EDStaffingDomain()
    fit1, ind1 = ensure_creator_types(d)
    fit2, ind2 = ensure_creator_types(d)
    # Second call must reuse the same registered types, not recreate them.
    assert fit1 is fit2
    assert ind1 is ind2


# ---------------------------------------------------------------------------
# Engine path == facade path
# ---------------------------------------------------------------------------


def test_engine_run_matches_solve_facade() -> None:
    """run(..., EDStaffingDomain()) must equal the legacy solve() output."""
    prob = _make_problem()
    cfg = RunConfig(solver="nsga2", pop_size=10, generations=5, seed=123)

    via_engine = run(prob, cfg, EDStaffingDomain())
    via_facade = solve(prob, cfg)

    assert len(via_engine) == len(via_facade) == 1
    front_e, front_f = via_engine[0], via_facade[0]
    pts_e = [(s.z1, s.z2) for s in front_e.solutions]
    pts_f = [(s.z1, s.z2) for s in front_f.solutions]
    assert pts_e == pts_f


def test_engine_run_both_solvers() -> None:
    prob = _make_problem()
    cfg = RunConfig(solver="both", pop_size=10, generations=5, seed=1)
    fronts = run(prob, cfg, EDStaffingDomain())
    assert len(fronts) == 2
