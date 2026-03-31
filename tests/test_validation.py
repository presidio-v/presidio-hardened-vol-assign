"""Tests for the validation module."""

from __future__ import annotations

from pathlib import Path

import pytest

from presidio_vol_assign.models import RunConfig, SkillType, SolverType
from presidio_vol_assign.validation import guard_output_path, load_problem, validate_run_config

FIXTURES = Path(__file__).parent / "fixtures"
VALID_VOL = FIXTURES / "volunteers_valid.csv"
VALID_EDS = FIXTURES / "eds_valid.csv"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_problem_happy_path() -> None:
    prob = load_problem(VALID_VOL, VALID_EDS)
    assert prob.n_volunteers == 3
    assert prob.n_vacancies == 2
    assert prob.volunteers[0].volunteer_id == "V1"
    assert prob.volunteers[0].skill_type == SkillType.TRIAGE
    assert prob.volunteers[0].skill_level == 8.0
    assert prob.volunteers[0].distance_to("ED1") == 5.0
    assert prob.vacancies[0].ed_id == "ED1"
    assert prob.vacancies[1].vacancy_type == SkillType.ER_NURSE


# ---------------------------------------------------------------------------
# File-not-found
# ---------------------------------------------------------------------------


def test_missing_volunteers_file() -> None:
    with pytest.raises(FileNotFoundError, match="volunteers"):
        load_problem(FIXTURES / "nonexistent.csv", VALID_EDS)


def test_missing_eds_file() -> None:
    with pytest.raises(FileNotFoundError, match="eds"):
        load_problem(VALID_VOL, FIXTURES / "nonexistent.csv")


# ---------------------------------------------------------------------------
# Schema errors — EDs
# ---------------------------------------------------------------------------


def test_eds_missing_column(tmp_path: Path) -> None:
    bad = tmp_path / "eds.csv"
    bad.write_text("ed_id,vacancy_type,num_patients\nED1,triage,40\n")
    with pytest.raises(ValueError, match="emergency_level"):
        load_problem(VALID_VOL, bad)


def test_eds_duplicate_id(tmp_path: Path) -> None:
    bad = tmp_path / "eds.csv"
    bad.write_text(
        "ed_id,vacancy_type,num_patients,emergency_level\nED1,triage,40,8\nED1,er_nurse,20,6\n"
    )
    with pytest.raises(ValueError, match="duplicate ed_id"):
        load_problem(VALID_VOL, bad)


def test_eds_bad_vacancy_type(tmp_path: Path) -> None:
    bad = tmp_path / "eds.csv"
    bad.write_text("ed_id,vacancy_type,num_patients,emergency_level\nED1,surgeon,40,8\n")
    with pytest.raises(ValueError, match="vacancy_type"):
        load_problem(VALID_VOL, bad)


def test_eds_num_patients_out_of_range(tmp_path: Path) -> None:
    bad = tmp_path / "eds.csv"
    bad.write_text("ed_id,vacancy_type,num_patients,emergency_level\nED1,triage,150,8\n")
    with pytest.raises(ValueError, match="num_patients"):
        load_problem(VALID_VOL, bad)


def test_eds_emergency_level_out_of_range(tmp_path: Path) -> None:
    bad = tmp_path / "eds.csv"
    bad.write_text("ed_id,vacancy_type,num_patients,emergency_level\nED1,triage,40,11\n")
    with pytest.raises(ValueError, match="emergency_level"):
        load_problem(VALID_VOL, bad)


def test_eds_num_patients_non_integer(tmp_path: Path) -> None:
    bad = tmp_path / "eds.csv"
    bad.write_text("ed_id,vacancy_type,num_patients,emergency_level\nED1,triage,40.5,8\n")
    with pytest.raises(ValueError, match="num_patients"):
        load_problem(VALID_VOL, bad)


def test_eds_empty_file(tmp_path: Path) -> None:
    bad = tmp_path / "eds.csv"
    bad.write_text("ed_id,vacancy_type,num_patients,emergency_level\n")
    with pytest.raises(ValueError, match="no rows"):
        load_problem(VALID_VOL, bad)


# ---------------------------------------------------------------------------
# Schema errors — Volunteers
# ---------------------------------------------------------------------------


def test_volunteers_missing_distance_column(tmp_path: Path) -> None:
    # eds has ED1 + ED2, but volunteers CSV omits distance_ed_ED2
    bad = tmp_path / "vol.csv"
    bad.write_text(
        "volunteer_id,skill_type,skill_level,distance_ed_ED1,difficulty_tolerance\n"
        "V1,triage,8,5.0,7\n"
    )
    with pytest.raises(ValueError, match="distance_ed_ED2"):
        load_problem(bad, VALID_EDS)


def test_volunteers_duplicate_id(tmp_path: Path) -> None:
    bad = tmp_path / "vol.csv"
    bad.write_text(
        "volunteer_id,skill_type,skill_level,distance_ed_ED1,distance_ed_ED2,difficulty_tolerance\n"
        "V1,triage,8,5.0,12.0,7\n"
        "V1,triage,6,3.0,8.0,5\n"
    )
    with pytest.raises(ValueError, match="duplicate volunteer_id"):
        load_problem(bad, VALID_EDS)


def test_volunteers_bad_skill_type(tmp_path: Path) -> None:
    bad = tmp_path / "vol.csv"
    bad.write_text(
        "volunteer_id,skill_type,skill_level,distance_ed_ED1,distance_ed_ED2,difficulty_tolerance\n"
        "V1,doctor,8,5.0,12.0,7\n"
    )
    with pytest.raises(ValueError, match="skill_type"):
        load_problem(bad, VALID_EDS)


def test_volunteers_skill_level_out_of_range(tmp_path: Path) -> None:
    bad = tmp_path / "vol.csv"
    bad.write_text(
        "volunteer_id,skill_type,skill_level,distance_ed_ED1,distance_ed_ED2,difficulty_tolerance\n"
        "V1,triage,11,5.0,12.0,7\n"
    )
    with pytest.raises(ValueError, match="skill_level"):
        load_problem(bad, VALID_EDS)


def test_volunteers_distance_out_of_range(tmp_path: Path) -> None:
    bad = tmp_path / "vol.csv"
    bad.write_text(
        "volunteer_id,skill_type,skill_level,distance_ed_ED1,distance_ed_ED2,difficulty_tolerance\n"
        "V1,triage,8,200.0,12.0,7\n"
    )
    with pytest.raises(ValueError, match="distance_ed_ED1"):
        load_problem(bad, VALID_EDS)


def test_volunteers_id_injection_chars(tmp_path: Path) -> None:
    bad = tmp_path / "vol.csv"
    bad.write_text(
        "volunteer_id,skill_type,skill_level,distance_ed_ED1,distance_ed_ED2,difficulty_tolerance\n"
        "V1;rm -rf,triage,8,5.0,12.0,7\n"
    )
    with pytest.raises(ValueError, match="disallowed characters"):
        load_problem(bad, VALID_EDS)


# ---------------------------------------------------------------------------
# Feasibility constraints
# ---------------------------------------------------------------------------


def test_fewer_volunteers_than_vacancies(tmp_path: Path) -> None:
    # Only 1 volunteer, 2 vacancies
    vol = tmp_path / "vol.csv"
    vol.write_text(
        "volunteer_id,skill_type,skill_level,distance_ed_ED1,distance_ed_ED2,difficulty_tolerance\n"
        "V1,triage,8,5.0,12.0,7\n"
    )
    with pytest.raises(ValueError, match="Infeasible"):
        load_problem(vol, VALID_EDS)


def test_insufficient_skill_type_coverage(tmp_path: Path) -> None:
    # 3 volunteers but all triage; ED2 needs er_nurse
    vol = tmp_path / "vol.csv"
    vol.write_text(
        "volunteer_id,skill_type,skill_level,distance_ed_ED1,distance_ed_ED2,difficulty_tolerance\n"
        "V1,triage,8,5.0,12.0,7\n"
        "V2,triage,6,3.0,8.0,5\n"
        "V3,triage,7,2.0,9.0,6\n"
    )
    with pytest.raises(ValueError, match="er_nurse"):
        load_problem(vol, VALID_EDS)


# ---------------------------------------------------------------------------
# validate_run_config
# ---------------------------------------------------------------------------


def test_valid_run_config() -> None:
    validate_run_config(RunConfig(solver=SolverType.NSGA2))
    validate_run_config(RunConfig(solver="nrga"))
    validate_run_config(RunConfig(solver="both"))


def test_invalid_solver() -> None:
    with pytest.raises(ValueError, match="solver"):
        validate_run_config(RunConfig(solver="genetic"))


def test_pop_size_too_small() -> None:
    with pytest.raises(ValueError, match="pop_size"):
        validate_run_config(RunConfig(solver="nsga2", pop_size=1))


def test_generations_zero() -> None:
    with pytest.raises(ValueError, match="generations"):
        validate_run_config(RunConfig(solver="nsga2", generations=0))


def test_seed_non_integer() -> None:
    with pytest.raises(ValueError, match="seed"):
        validate_run_config(RunConfig(solver="nsga2", seed="not_an_int"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# guard_output_path
# ---------------------------------------------------------------------------


def test_guard_output_path_absolute(tmp_path: Path) -> None:
    result = guard_output_path(tmp_path / "results")
    assert result.is_absolute()


def test_guard_output_path_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = guard_output_path(Path("results"))
    assert result.is_absolute()
    assert "results" in str(result)


def test_guard_output_path_traversal_in_symlink(tmp_path: Path) -> None:
    # Construct a path object whose .parts contain ".." after resolve is bypassed.
    # The simplest way: craft a PurePosixPath-like string, then use the real guard.
    # On most OSes, Path("a/../../b").resolve() will NOT contain ".." because
    # resolve() normalises the path. We instead monkeypatch Path.resolve to
    # return a path that still has ".." in its parts.
    from unittest.mock import patch as _patch

    fake_resolved = Path("/some/../../etc/passwd")
    with _patch.object(Path, "resolve", return_value=fake_resolved):
        with pytest.raises(ValueError, match="path traversal"):
            guard_output_path(Path("/some/../../etc/passwd"))


def test_volunteers_empty_no_data_rows(tmp_path: Path) -> None:
    vol = tmp_path / "vol.csv"
    vol.write_text(
        "volunteer_id,skill_type,skill_level,distance_ed_ED1,distance_ed_ED2,difficulty_tolerance\n"
    )
    with pytest.raises(ValueError, match="no rows"):
        load_problem(vol, VALID_EDS)
