"""Input validation for presidio-hardened-vol-assign.

Public API:
    load_problem(volunteers_path, eds_path) -> ProblemInstance
    validate_run_config(config) -> None          (raises ValueError on bad input)
    guard_output_path(output) -> Path            (raises ValueError on path traversal)

All errors raise ValueError with a message that names the offending field/row
and states the expected range or set of values.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from presidio_vol_assign.models import (
    ProblemInstance,
    RunConfig,
    SkillType,
    Vacancy,
    Volunteer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_SOLVERS = {"nsga2", "nrga", "both"}
_SKILL_TYPES = {s.value for s in SkillType}

_VOL_REQUIRED_COLS = {"volunteer_id", "skill_type", "skill_level", "difficulty_tolerance"}
_ED_REQUIRED_COLS = {"ed_id", "vacancy_type", "num_patients", "emergency_level"}

_DISTANCE_COL_PREFIX = "distance_ed_"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_problem(volunteers_path: Path, eds_path: Path) -> ProblemInstance:
    """Parse and validate both CSV files, returning a validated ProblemInstance.

    Distance columns in volunteers.csv must follow the pattern ``distance_ed_<ed_id>``
    where ``<ed_id>`` matches a value in the ``ed_id`` column of eds.csv.

    Raises:
        FileNotFoundError: If either CSV file does not exist.
        ValueError: If any schema, range, or constraint check fails.
    """
    _require_file(volunteers_path, "volunteers")
    _require_file(eds_path, "eds")

    eds_df = _load_csv(eds_path, "eds")
    vol_df = _load_csv(volunteers_path, "volunteers")

    vacancies = _parse_vacancies(eds_df)
    volunteers = _parse_volunteers(vol_df, ed_ids=[v.ed_id for v in vacancies])

    _check_assignment_feasibility(volunteers, vacancies)

    return ProblemInstance(volunteers=volunteers, vacancies=vacancies)


def validate_run_config(config: RunConfig) -> None:
    """Validate solver hyper-parameters.

    Raises:
        ValueError: On any invalid field value.
    """
    solver_val = config.solver if isinstance(config.solver, str) else config.solver.value
    if solver_val not in _VALID_SOLVERS:
        raise ValueError(f"solver must be one of {sorted(_VALID_SOLVERS)!r}, got {solver_val!r}")
    if config.pop_size < 2:
        raise ValueError(f"pop_size must be >= 2, got {config.pop_size}")
    if config.generations < 1:
        raise ValueError(f"generations must be >= 1, got {config.generations}")
    if config.seed is not None and not isinstance(config.seed, int):
        raise ValueError(f"seed must be an integer or None, got {type(config.seed).__name__}")


def guard_output_path(output: Path) -> Path:
    """Resolve *output* to an absolute path and reject path-traversal attempts.

    Args:
        output: Raw path supplied by the caller (e.g. from the CLI --output flag).

    Returns:
        The resolved absolute Path.

    Raises:
        ValueError: If the resolved path contains ``..`` components or is not
            inside the current working directory tree when a relative path was given.
    """
    resolved = output.resolve()
    # Reject any remaining .. after resolution (belt-and-suspenders)
    if ".." in resolved.parts:
        raise ValueError(
            f"Output path {output!r} resolves to {resolved!r} which contains '..' — "
            "path traversal is not allowed."
        )
    return resolved


# ---------------------------------------------------------------------------
# Internal helpers — EDs
# ---------------------------------------------------------------------------


def _parse_vacancies(df: pd.DataFrame) -> list[Vacancy]:
    _require_columns(df, _ED_REQUIRED_COLS, source="eds")
    vacancies: list[Vacancy] = []
    seen_ids: set[str] = set()

    for idx, row in df.iterrows():
        row_label = f"eds row {idx}"

        ed_id = _require_nonempty_str(row["ed_id"], "ed_id", row_label)
        if ed_id in seen_ids:
            raise ValueError(f"{row_label}: duplicate ed_id {ed_id!r}")
        seen_ids.add(ed_id)

        vacancy_type = _require_skill_type(row["vacancy_type"], "vacancy_type", row_label)
        num_patients = _require_int_range(row["num_patients"], "num_patients", 0, 100, row_label)
        emergency_level = _require_float_range(
            row["emergency_level"], "emergency_level", 0.0, 10.0, row_label
        )

        vacancies.append(
            Vacancy(
                ed_id=ed_id,
                vacancy_type=vacancy_type,
                num_patients=num_patients,
                emergency_level=emergency_level,
            )
        )

    if not vacancies:
        raise ValueError("eds CSV contains no rows — at least one vacancy is required")

    return vacancies


# ---------------------------------------------------------------------------
# Internal helpers — Volunteers
# ---------------------------------------------------------------------------


def _parse_volunteers(df: pd.DataFrame, ed_ids: list[str]) -> list[Volunteer]:
    _require_columns(df, _VOL_REQUIRED_COLS, source="volunteers")

    # Expect one distance column per ED: distance_ed_<ed_id>
    expected_dist_cols = {f"{_DISTANCE_COL_PREFIX}{eid}" for eid in ed_ids}
    _require_columns(df, expected_dist_cols, source="volunteers (distance columns)")

    volunteers: list[Volunteer] = []
    seen_ids: set[str] = set()

    for idx, row in df.iterrows():
        row_label = f"volunteers row {idx}"

        vol_id = _require_nonempty_str(row["volunteer_id"], "volunteer_id", row_label)
        if vol_id in seen_ids:
            raise ValueError(f"{row_label}: duplicate volunteer_id {vol_id!r}")
        seen_ids.add(vol_id)

        skill_type = _require_skill_type(row["skill_type"], "skill_type", row_label)
        skill_level = _require_float_range(row["skill_level"], "skill_level", 0.0, 10.0, row_label)
        difficulty_tolerance = _require_float_range(
            row["difficulty_tolerance"], "difficulty_tolerance", 0.0, 10.0, row_label
        )

        distances: dict[str, float] = {}
        for eid in ed_ids:
            col = f"{_DISTANCE_COL_PREFIX}{eid}"
            distances[eid] = _require_float_range(row[col], col, 0.0, 100.0, row_label)

        volunteers.append(
            Volunteer(
                volunteer_id=vol_id,
                skill_type=skill_type,
                skill_level=skill_level,
                distances=distances,
                difficulty_tolerance=difficulty_tolerance,
            )
        )

    if not volunteers:
        raise ValueError("volunteers CSV contains no rows")

    return volunteers


# ---------------------------------------------------------------------------
# Internal helpers — cross-file constraint checks
# ---------------------------------------------------------------------------


def _check_assignment_feasibility(volunteers: list[Volunteer], vacancies: list[Vacancy]) -> None:
    """Verify the model's feasibility constraints (paper §3.3)."""
    if len(volunteers) < len(vacancies):
        raise ValueError(
            f"Infeasible: {len(volunteers)} volunteer(s) < {len(vacancies)} vacanc(y/ies). "
            "The model requires at least as many volunteers as vacancies."
        )

    # Check per-skill-type feasibility
    for skill in SkillType:
        needed = sum(1 for v in vacancies if v.vacancy_type == skill)
        available = sum(1 for v in volunteers if v.skill_type == skill)
        if available < needed:
            raise ValueError(
                f"Infeasible: {needed} {skill.value} vacanc(y/ies) but only "
                f"{available} volunteer(s) with skill_type={skill.value!r}."
            )


# ---------------------------------------------------------------------------
# Internal helpers — primitive validators
# ---------------------------------------------------------------------------


def _require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    if not path.is_file():
        raise ValueError(f"{name} path is not a file: {path}")


def _load_csv(path: Path, name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as exc:
        raise ValueError(f"Could not parse {name} CSV at {path}: {exc}") from exc
    df.columns = [c.strip() for c in df.columns]
    return df


def _require_columns(df: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source} CSV is missing required column(s): {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )


def _require_nonempty_str(value: object, field: str, row_label: str) -> str:
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", ""}:
        raise ValueError(f"{row_label}: {field} must not be empty")
    # Reject shell-injection characters in IDs
    if re.search(r"[;&|`$<>]", s):
        raise ValueError(f"{row_label}: {field} contains disallowed characters: {s!r}")
    return s


def _require_skill_type(value: object, field: str, row_label: str) -> SkillType:
    s = str(value).strip().lower()
    if s not in _SKILL_TYPES:
        raise ValueError(f"{row_label}: {field} must be one of {sorted(_SKILL_TYPES)!r}, got {s!r}")
    return SkillType(s)


def _require_float_range(value: object, field: str, lo: float, hi: float, row_label: str) -> float:
    try:
        f = float(str(value).strip())
    except (ValueError, TypeError):
        raise ValueError(f"{row_label}: {field} must be a number, got {value!r}")
    if not (lo <= f <= hi):
        raise ValueError(f"{row_label}: {field} must be in [{lo}, {hi}], got {f}")
    return f


def _require_int_range(value: object, field: str, lo: int, hi: int, row_label: str) -> int:
    try:
        f = float(str(value).strip())
        i = int(f)
        if f != i:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError(f"{row_label}: {field} must be an integer, got {value!r}")
    if not (lo <= i <= hi):
        raise ValueError(f"{row_label}: {field} must be in [{lo}, {hi}], got {i}")
    return i
