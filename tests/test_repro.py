"""Tests for the bit-for-bit reproducibility check."""

from __future__ import annotations

import pytest

from presidio_vol_assign.domains import EDStaffingDomain
from presidio_vol_assign.models import (
    ProblemInstance,
    RunConfig,
    SkillType,
    Vacancy,
    Volunteer,
)
from presidio_vol_assign.repro import verify_reproducibility


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


def test_seeded_run_is_reproducible() -> None:
    prob = _make_problem()
    cfg = RunConfig(solver="both", pop_size=10, generations=5, seed=42)
    report = verify_reproducibility(prob, cfg, EDStaffingDomain(), n_runs=3)
    assert report.identical is True
    assert report.rep == 1.0
    assert report.n_runs == 3
    assert len(report.signature) == 64  # SHA-256 hex digest


def test_unseeded_run_rejected() -> None:
    prob = _make_problem()
    cfg = RunConfig(solver="nsga2", pop_size=10, generations=5, seed=None)
    with pytest.raises(ValueError, match="seed"):
        verify_reproducibility(prob, cfg, EDStaffingDomain())


def test_n_runs_must_be_at_least_two() -> None:
    prob = _make_problem()
    cfg = RunConfig(solver="nsga2", pop_size=10, generations=5, seed=42)
    with pytest.raises(ValueError, match="n_runs"):
        verify_reproducibility(prob, cfg, EDStaffingDomain(), n_runs=1)
