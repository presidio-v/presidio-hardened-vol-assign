"""Tests for output writers, focused on CSV formula-injection neutralisation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from presidio_vol_assign.domains import EDStaffingDomain, HumanitarianDomain
from presidio_vol_assign.models import (
    Assignment,
    CenterAssignment,
    ParetoFront,
    SkillType,
    Solution,
    SolverType,
)
from presidio_vol_assign.writers import _csv_safe, write_assignments_csv

# ---------------------------------------------------------------------------
# _csv_safe unit behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("=SUM(1+1)", "'=SUM(1+1)"),
        ("+1", "'+1"),
        ("-cmd", "'-cmd"),
        ("@x", "'@x"),
        ("\tTAB", "'\tTAB"),
        ("\rCR", "'\rCR"),
        ("V1", "V1"),  # benign string untouched
        ("normal_id", "normal_id"),
    ],
)
def test_csv_safe_escapes_formula_prefixes(value: str, expected: str) -> None:
    assert _csv_safe(value) == expected


def test_csv_safe_passes_non_strings() -> None:
    assert _csv_safe(3) == 3
    assert _csv_safe(0.5) == 0.5


# ---------------------------------------------------------------------------
# End-to-end: malicious IDs are neutralised in the assignments CSV
# ---------------------------------------------------------------------------


def test_ed_assignment_id_is_escaped(tmp_path: Path) -> None:
    malicious = "=cmd|'/C calc'!A1"
    asgn = Assignment(
        volunteer_id=malicious,
        ed_id="ED1",
        vacancy_type=SkillType.TRIAGE,
        fis1_score=0.1,
        fis3_score=0.2,
    )
    front = ParetoFront(
        solver=SolverType.NSGA2,
        solutions=[Solution(assignments=[asgn], z1=0.1, z2=0.2)],
    )
    path = write_assignments_csv(front, tmp_path, EDStaffingDomain())

    raw = path.read_text()
    # The raw cell must not begin a formula; it is quote-prefixed.
    assert "'" + malicious in raw
    df = pd.read_csv(path, dtype=str)
    assert df.iloc[0]["volunteer_id"] == "'" + malicious


def test_humanitarian_ids_are_escaped(tmp_path: Path) -> None:
    asgn = CenterAssignment(
        person_id="@evil()",
        center_id="-1+2",
        fairness=0.1,
        transport=0.2,
        overcrowding=0.3,
    )
    front = ParetoFront(
        solver=SolverType.NRGA,
        solutions=[Solution(assignments=[asgn], objectives=(0.1, 0.2, 0.3))],
    )
    path = write_assignments_csv(front, tmp_path, HumanitarianDomain())

    df = pd.read_csv(path, dtype=str)
    assert df.iloc[0]["person_id"] == "'@evil()"
    assert df.iloc[0]["center_id"] == "'-1+2"


def test_benign_ids_unchanged(tmp_path: Path) -> None:
    asgn = Assignment(volunteer_id="V1", ed_id="ED1", vacancy_type=SkillType.TRIAGE)
    front = ParetoFront(
        solver=SolverType.NSGA2,
        solutions=[Solution(assignments=[asgn], z1=0.1, z2=0.2)],
    )
    path = write_assignments_csv(front, tmp_path, EDStaffingDomain())
    df = pd.read_csv(path, dtype=str)
    assert df.iloc[0]["volunteer_id"] == "V1"
    assert df.iloc[0]["ed_id"] == "ED1"
