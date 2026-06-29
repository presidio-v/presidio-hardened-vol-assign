"""Tests for the objective-ablation analysis (indicator-validation, R2.2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from presidio_vol_assign.ablation import (
    AblationRow,
    _AblatedDomain,
    run_ablation,
    write_ablation_csv,
)
from presidio_vol_assign.domains import get_domain
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


def _hum_problem() -> HumanitarianProblem:
    people = [
        Person("P1", 9.0, 2.0, 3, {"C1": 5.0, "C2": 30.0}),
        Person("P2", 4.0, 7.0, 1, {"C1": 12.0, "C2": 8.0}),
        Person("P3", 7.5, 4.0, 2, {"C1": 25.0, "C2": 4.0}),
        Person("P4", 2.0, 9.0, 1, {"C1": 8.0, "C2": 15.0}),
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


def _cfg(solver: str = "nsga2") -> RunConfig:
    return RunConfig(solver=solver, pop_size=16, generations=8, seed=42)


# ---------------------------------------------------------------------------
# _AblatedDomain
# ---------------------------------------------------------------------------


def test_ablated_domain_projects_out_objective() -> None:
    base = get_domain("humanitarian")
    prob = _hum_problem()
    cache = base.precompute(prob)
    ablated = _AblatedDomain(base, drop_idx=1)

    assert ablated.n_objectives == 2
    assert ablated.objective_names == ("z1", "z3")
    assert ablated.weights == (-1.0, -1.0)
    # Reduced fitness/individual creator names are distinct from the base ones.
    assert ablated.fitness_attr != base.fitness_attr
    assert ablated.individual_attr != base.individual_attr

    ind = base.init_individual(prob, list)
    full = base.evaluate(ind, cache, prob)
    reduced = ablated.evaluate(ind, cache, prob)
    assert len(full) == 3 and len(reduced) == 2
    assert reduced == (full[0], full[2])


# ---------------------------------------------------------------------------
# run_ablation
# ---------------------------------------------------------------------------


def test_run_ablation_humanitarian_one_row_per_objective() -> None:
    domain = get_domain("humanitarian")
    rows = run_ablation(domain, _hum_problem(), _cfg())
    assert [r.dropped for r in rows] == ["z1", "z2", "z3"]
    for r in rows:
        assert r.nns_full >= 1 and r.nns_ablated >= 1
        assert r.solver == "nsga2"
        # delta_dropped and delta_hv are finite numbers.
        assert r.delta_dropped == pytest.approx(r.mean_dropped_ablated - r.mean_dropped_full)
        assert r.delta_hv == pytest.approx(r.hv_full - r.hv_ablated)


def test_run_ablation_ed_two_rows() -> None:
    domain = get_domain("ed-staffing")
    rows = run_ablation(domain, _ed_problem(), _cfg())
    assert [r.dropped for r in rows] == ["z1", "z2"]


def test_run_ablation_is_deterministic() -> None:
    domain = get_domain("humanitarian")
    prob = _hum_problem()
    r1 = run_ablation(domain, prob, _cfg())
    r2 = run_ablation(domain, prob, _cfg())
    assert [r.delta_hv for r in r1] == [r.delta_hv for r in r2]
    assert [r.delta_dropped for r in r1] == [r.delta_dropped for r in r2]


def test_run_ablation_resolves_single_solver_from_both() -> None:
    # 'both' expands to [nsga2, nrga]; ablation uses the first deterministically.
    domain = get_domain("humanitarian")
    rows = run_ablation(domain, _hum_problem(), _cfg("both"))
    assert all(r.solver == "nsga2" for r in rows)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_write_ablation_csv(tmp_path: Path) -> None:
    rows = [
        AblationRow(
            dropped="z1",
            solver="nsga2",
            nns_full=5,
            nns_ablated=3,
            mean_dropped_full=0.30,
            mean_dropped_ablated=0.55,
            delta_dropped=0.25,
            hv_full=0.40,
            hv_ablated=0.28,
            delta_hv=0.12,
        )
    ]
    path = write_ablation_csv(rows, tmp_path)
    assert path.exists() and path.name.startswith("ablation_")
    df = pd.read_csv(path)
    assert {"dropped", "delta_dropped", "delta_hv", "hv_full", "hv_ablated"} <= set(df.columns)
