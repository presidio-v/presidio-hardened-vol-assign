"""Tests for the canonical rank-biased NRGA variant (--solver nrga-ranked)."""

from __future__ import annotations

import pytest
from deap import creator

from presidio_vol_assign.domains import EDStaffingDomain
from presidio_vol_assign.engine import _solvers_for, run, sel_nrga_ranked
from presidio_vol_assign.models import (
    ProblemInstance,
    RunConfig,
    SkillType,
    SolverType,
    Vacancy,
    Volunteer,
)
from presidio_vol_assign.validation import validate_run_config


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
# Selector
# ---------------------------------------------------------------------------


def test_sel_nrga_ranked_returns_k_distinct() -> None:
    import random

    random.seed(0)
    inds = [creator.PVAIndividual([i]) for i in range(8)]
    for i, ind in enumerate(inds):
        ind.fitness.values = (i * 0.1, 1.0 - i * 0.1)
    chosen = sel_nrga_ranked(inds, 4)
    assert len(chosen) == 4
    assert len({id(c) for c in chosen}) == 4  # without replacement


def test_sel_nrga_ranked_k_ge_n_returns_all() -> None:
    inds = [creator.PVAIndividual([i]) for i in range(3)]
    for i, ind in enumerate(inds):
        ind.fitness.values = (float(i), float(-i))
    assert len(sel_nrga_ranked(inds, 5)) == 3


# ---------------------------------------------------------------------------
# Solver expansion + validation
# ---------------------------------------------------------------------------


def test_solvers_for_expansion() -> None:
    assert _solvers_for("both") == [SolverType.NSGA2, SolverType.NRGA]
    assert _solvers_for("all") == [SolverType.NSGA2, SolverType.NRGA, SolverType.NRGA_RANKED]
    assert _solvers_for("nrga-ranked") == [SolverType.NRGA_RANKED]


def test_validate_accepts_new_solvers() -> None:
    for s in ("nrga-ranked", "all"):
        validate_run_config(RunConfig(solver=s, pop_size=10, generations=3, seed=1))


def test_validate_rejects_unknown_solver() -> None:
    with pytest.raises(ValueError, match="solver"):
        validate_run_config(RunConfig(solver="nrga-roulette", pop_size=10, generations=3))


# ---------------------------------------------------------------------------
# End-to-end through the engine
# ---------------------------------------------------------------------------


def _cfg(solver: str) -> RunConfig:
    return RunConfig(solver=solver, pop_size=12, generations=6, seed=42)


def test_nrga_ranked_front_and_reproducible() -> None:
    prob = _make_problem()
    f1 = run(prob, _cfg("nrga-ranked"), EDStaffingDomain())
    f2 = run(prob, _cfg("nrga-ranked"), EDStaffingDomain())
    assert len(f1) == 1
    assert f1[0].solver == SolverType.NRGA_RANKED
    assert f1[0].nns >= 1
    assert [(s.z1, s.z2) for s in f1[0].solutions] == [(s.z1, s.z2) for s in f2[0].solutions]


def test_all_runs_three_solvers() -> None:
    fronts = run(_make_problem(), _cfg("all"), EDStaffingDomain())
    assert [f.solver for f in fronts] == [
        SolverType.NSGA2,
        SolverType.NRGA,
        SolverType.NRGA_RANKED,
    ]
