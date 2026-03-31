"""Tests for core data models."""

from __future__ import annotations

from presidio_vol_assign.models import (
    Assignment,
    Metrics,
    ParetoFront,
    ProblemInstance,
    RunConfig,
    SkillType,
    Solution,
    SolverType,
    Vacancy,
    Volunteer,
)


def _volunteer(vid: str = "V1", ed_ids: list[str] | None = None) -> Volunteer:
    ed_ids = ed_ids or ["ED1", "ED2"]
    return Volunteer(
        volunteer_id=vid,
        skill_type=SkillType.TRIAGE,
        skill_level=8.0,
        distances={eid: float(i + 5) for i, eid in enumerate(ed_ids)},
        difficulty_tolerance=7.0,
    )


def _vacancy(vid: str = "ED1") -> Vacancy:
    return Vacancy(
        ed_id=vid,
        vacancy_type=SkillType.TRIAGE,
        num_patients=40,
        emergency_level=8.0,
    )


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


def test_skill_type_values() -> None:
    assert SkillType.TRIAGE == "triage"
    assert SkillType.ER_NURSE == "er_nurse"


def test_solver_type_values() -> None:
    assert SolverType.NSGA2 == "nsga2"
    assert SolverType.NRGA == "nrga"


# ---------------------------------------------------------------------------
# Volunteer
# ---------------------------------------------------------------------------


def test_volunteer_distance_to() -> None:
    v = _volunteer(ed_ids=["ED1", "ED2"])
    assert v.distance_to("ED1") == 5.0
    assert v.distance_to("ED2") == 6.0


def test_volunteer_missing_ed_raises() -> None:
    v = _volunteer(ed_ids=["ED1"])
    try:
        v.distance_to("ED99")
        assert False, "expected KeyError"
    except KeyError:
        pass


# ---------------------------------------------------------------------------
# ProblemInstance
# ---------------------------------------------------------------------------


def test_problem_instance_counts() -> None:
    prob = ProblemInstance(
        volunteers=[_volunteer("V1"), _volunteer("V2")],
        vacancies=[_vacancy("ED1")],
    )
    assert prob.n_volunteers == 2
    assert prob.n_vacancies == 1


# ---------------------------------------------------------------------------
# RunConfig defaults
# ---------------------------------------------------------------------------


def test_run_config_defaults() -> None:
    cfg = RunConfig(solver=SolverType.NSGA2)
    assert cfg.pop_size == 100
    assert cfg.generations == 200
    assert cfg.seed is None
    assert cfg.output_dir == "./results"


# ---------------------------------------------------------------------------
# Solution and ParetoFront
# ---------------------------------------------------------------------------


def test_solution_n_assignments() -> None:
    a = Assignment(volunteer_id="V1", ed_id="ED1", vacancy_type=SkillType.TRIAGE)
    sol = Solution(assignments=[a], z1=0.6, z2=0.4)
    assert sol.n_assignments == 1


def test_pareto_front_nns() -> None:
    a = Assignment(volunteer_id="V1", ed_id="ED1", vacancy_type=SkillType.TRIAGE)
    sol = Solution(assignments=[a], z1=0.6, z2=0.4)
    front = ParetoFront(solver=SolverType.NRGA, solutions=[sol], cpu_time_sec=2.5)
    assert front.nns == 1
    assert front.cpu_time_sec == 2.5


def test_pareto_front_empty_by_default() -> None:
    front = ParetoFront(solver=SolverType.NSGA2)
    assert front.nns == 0
    assert front.solutions == []


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_fields() -> None:
    m = Metrics(
        solver=SolverType.NSGA2,
        nns=12,
        mid=0.34,
        sm=0.05,
        hv=0.78,
        cpu_time_sec=3.14,
    )
    assert m.nns == 12
    assert m.solver == SolverType.NSGA2
