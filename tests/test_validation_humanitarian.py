"""Tests for humanitarian CSV validation (load_humanitarian_problem)."""

from __future__ import annotations

from pathlib import Path

import pytest

from presidio_vol_assign.validation import load_humanitarian_problem

FIXTURES = Path(__file__).parent / "fixtures"
VALID_PEOPLE = FIXTURES / "people_valid.csv"
VALID_CENTERS = FIXTURES / "centers_valid.csv"

_CENTERS = "center_id,capacity,service_level,road_accessibility\nC1,50,8.0,7.0\nC2,40,5.0,6.0\n"
_PEOPLE_HEADER = "person_id,vulnerability,mobility,group_size,distance_center_C1,distance_center_C2"


def _write(tmp: Path, centers: str, people: str) -> tuple[Path, Path]:
    cpath = tmp / "centers.csv"
    ppath = tmp / "people.csv"
    cpath.write_text(centers)
    ppath.write_text(people)
    return ppath, cpath


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_valid_fixture() -> None:
    prob = load_humanitarian_problem(VALID_PEOPLE, VALID_CENTERS)
    assert prob.n_people == 6
    assert prob.n_centers == 3
    assert prob.people[0].distance_to("C1") == 5.0


def test_group_size_optional_defaults_to_one(tmp_path: Path) -> None:
    people = (
        "person_id,vulnerability,mobility,distance_center_C1,distance_center_C2\n"
        "P1,9.0,2.0,5.0,18.0\n"
    )
    ppath, cpath = _write(tmp_path, _CENTERS, people)
    prob = load_humanitarian_problem(ppath, cpath)
    assert prob.people[0].group_size == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_center_column_raises(tmp_path: Path) -> None:
    bad_centers = "center_id,capacity,service_level\nC1,50,8.0\n"
    people = f"{_PEOPLE_HEADER}\nP1,9.0,2.0,3,5.0,18.0\n"
    ppath, cpath = _write(tmp_path, bad_centers, people)
    with pytest.raises(ValueError, match="road_accessibility"):
        load_humanitarian_problem(ppath, cpath)


def test_missing_distance_column_raises(tmp_path: Path) -> None:
    people = "person_id,vulnerability,mobility,group_size,distance_center_C1\nP1,9.0,2.0,3,5.0\n"
    ppath, cpath = _write(tmp_path, _CENTERS, people)
    with pytest.raises(ValueError, match="distance_center_C2"):
        load_humanitarian_problem(ppath, cpath)


def test_capacity_infeasible_raises(tmp_path: Path) -> None:
    centers = "center_id,capacity,service_level,road_accessibility\nC1,2,8.0,7.0\nC2,1,5.0,6.0\n"
    people = f"{_PEOPLE_HEADER}\nP1,9.0,2.0,5,5.0,18.0\nP2,4.0,7.0,5,12.0,8.0\n"
    ppath, cpath = _write(tmp_path, centers, people)
    with pytest.raises(ValueError, match="capacity"):
        load_humanitarian_problem(ppath, cpath)


def test_out_of_range_vulnerability_raises(tmp_path: Path) -> None:
    people = f"{_PEOPLE_HEADER}\nP1,99.0,2.0,3,5.0,18.0\n"
    ppath, cpath = _write(tmp_path, _CENTERS, people)
    with pytest.raises(ValueError, match="vulnerability"):
        load_humanitarian_problem(ppath, cpath)


def test_duplicate_person_id_raises(tmp_path: Path) -> None:
    people = f"{_PEOPLE_HEADER}\nP1,9.0,2.0,3,5.0,18.0\nP1,4.0,7.0,1,12.0,8.0\n"
    ppath, cpath = _write(tmp_path, _CENTERS, people)
    with pytest.raises(ValueError, match="duplicate"):
        load_humanitarian_problem(ppath, cpath)
