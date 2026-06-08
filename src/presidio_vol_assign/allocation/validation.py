"""Input validation for the allocation module.

Public API:
    load_allocation_problem(people, centers, travel, n_dir) -> AllocationProblem
    validate_allocation_config(config) -> None
    guard_output_path(output) -> Path     (delegates to volunteer-assign module)

Three CSV files form the allocation input:

    people.csv      person_id, age, disability_status, injury_level,
                    living_status, idl, rtr
    centers.csv     center_id, cor, rdr
    travel.csv      person_id, center_id, td, rcs, phs

Plus the scalar `n_dir` from the CLI: how many people we can direct to
relief centers (Eq. 15: 0 < n_dir < n_people).

All errors raise ValueError with a message that names the offending field
or row and states the expected range or set of values, matching the style
of the existing volunteer-assignment validator.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from presidio_vol_assign.allocation.models import (
    AllocationConfig,
    AllocationProblem,
    AllocationSolverType,
    DisabilityStatus,
    HazardLevel,
    InjuryLevel,
    LivingStatus,
    Person,
    ReliefCenter,
    RoadCondition,
    TravelInfo,
)

# Reuse the primitives proved out for the volunteer-assignment validator;
# they are general CSV / range / path utilities.
from presidio_vol_assign.validation import (
    _load_csv,
    _require_columns,
    _require_file,
    _require_float_range,
    _require_nonempty_str,
)
from presidio_vol_assign.validation import guard_output_path as guard_output_path  # re-export

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_PEOPLE_REQUIRED_COLS = {
    "person_id",
    "age",
    "disability_status",
    "injury_level",
    "living_status",
    "idl",
    "rtr",
}
_CENTERS_REQUIRED_COLS = {"center_id", "cor", "rdr"}
_TRAVEL_REQUIRED_COLS = {"person_id", "center_id", "td", "rcs", "phs"}

_DISABILITY_VALUES = {s.value for s in DisabilityStatus}
_INJURY_VALUES = {s.value for s in InjuryLevel}
_LIVING_VALUES = {s.value for s in LivingStatus}
_RCS_VALUES = {s.value for s in RoadCondition}
_PHS_VALUES = {s.value for s in HazardLevel}

_VALID_SOLVERS = {s.value for s in AllocationSolverType}
_VALID_OBJECTIVES = {3, 4}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_allocation_problem(
    people_path: Path,
    centers_path: Path,
    travel_path: Path,
    n_dir: int,
) -> AllocationProblem:
    """Parse and validate the three allocation CSVs and the n_dir scalar.

    Cross-file checks enforced:
        - travel.csv covers exactly the (person, center) Cartesian product
        - 0 < n_dir < n_people  (ATRes Eq. 15)
        - n_centers > 0
        - no duplicate IDs in any source

    Raises:
        FileNotFoundError: If any input file is missing.
        ValueError: On any schema, range, set-membership, or constraint
            violation. The message names the offending CSV, row, and field.
    """
    _require_file(people_path, "people")
    _require_file(centers_path, "centers")
    _require_file(travel_path, "travel")

    people_df = _load_csv(people_path, "people")
    centers_df = _load_csv(centers_path, "centers")
    travel_df = _load_csv(travel_path, "travel")

    centers = _parse_centers(centers_df)
    people = _parse_people(people_df)
    travel = _parse_travel(travel_df, people, centers)

    if n_dir <= 0:
        raise ValueError(f"n_dir must be > 0, got {n_dir}")
    if n_dir >= len(people):
        raise ValueError(
            f"n_dir must be < n_people for the model to be meaningful "
            f"(ATRes Eq. 15): got n_dir={n_dir}, n_people={len(people)}"
        )

    return AllocationProblem(people=people, centers=centers, travel=travel, n_dir=n_dir)


def validate_allocation_config(config: AllocationConfig) -> None:
    """Validate solver hyper-parameters for the allocation run.

    Raises ValueError on any invalid field value.
    """
    solver_val = config.solver if isinstance(config.solver, str) else config.solver.value
    if solver_val not in _VALID_SOLVERS:
        raise ValueError(f"solver must be one of {sorted(_VALID_SOLVERS)!r}, got {solver_val!r}")
    if config.objectives not in _VALID_OBJECTIVES:
        raise ValueError(f"objectives must be 3 or 4, got {config.objectives}")
    if config.pop_size < 2:
        raise ValueError(f"pop_size must be >= 2, got {config.pop_size}")
    if config.generations < 1:
        raise ValueError(f"generations must be >= 1, got {config.generations}")
    if config.seed is not None and not isinstance(config.seed, int):
        raise ValueError(f"seed must be an integer or None, got {type(config.seed).__name__}")
    if config.nsga3_divisions < 1:
        raise ValueError(f"nsga3_divisions must be >= 1, got {config.nsga3_divisions}")
    for name in ("was", "wds", "wil", "wls", "wrc", "wph"):
        w = getattr(config.weights, name)
        if not (0.0 <= float(w) <= 1.0):
            raise ValueError(f"weights.{name} must be in [0, 1], got {w}")


# ---------------------------------------------------------------------------
# Centers
# ---------------------------------------------------------------------------


def _parse_centers(df: pd.DataFrame) -> list[ReliefCenter]:
    _require_columns(df, _CENTERS_REQUIRED_COLS, source="centers")
    centers: list[ReliefCenter] = []
    seen_ids: set[str] = set()

    for idx, row in df.iterrows():
        row_label = f"centers row {idx}"
        center_id = _require_nonempty_str(row["center_id"], "center_id", row_label)
        if center_id in seen_ids:
            raise ValueError(f"{row_label}: duplicate center_id {center_id!r}")
        seen_ids.add(center_id)

        cor = _require_float_range(row["cor"], "cor", 0.0, 100.0, row_label)
        rdr = _require_float_range(row["rdr"], "rdr", 0.0, 100.0, row_label)

        centers.append(
            ReliefCenter(
                center_id=center_id, center_occupancy_rate=cor, resource_depletion_rate=rdr
            )
        )

    if not centers:
        raise ValueError("centers CSV contains no rows — at least one center is required")
    return centers


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------


def _parse_people(df: pd.DataFrame) -> list[Person]:
    _require_columns(df, _PEOPLE_REQUIRED_COLS, source="people")
    people: list[Person] = []
    seen_ids: set[str] = set()

    for idx, row in df.iterrows():
        row_label = f"people row {idx}"
        person_id = _require_nonempty_str(row["person_id"], "person_id", row_label)
        if person_id in seen_ids:
            raise ValueError(f"{row_label}: duplicate person_id {person_id!r}")
        seen_ids.add(person_id)

        age = _require_float_range(row["age"], "age", 0.0, 120.0, row_label)
        disability = _require_enum_value(
            row["disability_status"], "disability_status", _DISABILITY_VALUES, row_label
        )
        injury = _require_enum_value(row["injury_level"], "injury_level", _INJURY_VALUES, row_label)
        living = _require_enum_value(
            row["living_status"], "living_status", _LIVING_VALUES, row_label
        )
        idl = _require_float_range(row["idl"], "idl", 0.0, 100.0, row_label)
        rtr = _require_float_range(row["rtr"], "rtr", 0.0, 48.0, row_label)

        people.append(
            Person(
                person_id=person_id,
                age=age,
                disability_status=DisabilityStatus(disability),
                injury_level=InjuryLevel(injury),
                living_status=LivingStatus(living),
                infrastructure_damage_level=idl,
                resource_time_remaining=rtr,
            )
        )

    if not people:
        raise ValueError("people CSV contains no rows")
    return people


# ---------------------------------------------------------------------------
# Travel
# ---------------------------------------------------------------------------


def _parse_travel(
    df: pd.DataFrame,
    people: list[Person],
    centers: list[ReliefCenter],
) -> dict[tuple[str, str], TravelInfo]:
    _require_columns(df, _TRAVEL_REQUIRED_COLS, source="travel")
    person_ids = {p.person_id for p in people}
    center_ids = {c.center_id for c in centers}

    travel: dict[tuple[str, str], TravelInfo] = {}
    seen_pairs: set[tuple[str, str]] = set()

    for idx, row in df.iterrows():
        row_label = f"travel row {idx}"
        person_id = _require_nonempty_str(row["person_id"], "person_id", row_label)
        center_id = _require_nonempty_str(row["center_id"], "center_id", row_label)

        if person_id not in person_ids:
            raise ValueError(f"{row_label}: person_id {person_id!r} not present in people.csv")
        if center_id not in center_ids:
            raise ValueError(f"{row_label}: center_id {center_id!r} not present in centers.csv")

        pair = (person_id, center_id)
        if pair in seen_pairs:
            raise ValueError(f"{row_label}: duplicate (person_id, center_id) pair {pair!r}")
        seen_pairs.add(pair)

        td = _require_float_range(row["td"], "td", 0.0, 180.0, row_label)
        rcs = _require_enum_value(row["rcs"], "rcs", _RCS_VALUES, row_label)
        phs = _require_enum_value(row["phs"], "phs", _PHS_VALUES, row_label)

        travel[pair] = TravelInfo(
            person_id=person_id,
            center_id=center_id,
            travel_duration=td,
            road_condition=RoadCondition(rcs),
            possible_hazard=HazardLevel(phs),
        )

    expected_pairs = {(p, c) for p in person_ids for c in center_ids}
    missing = expected_pairs - seen_pairs
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"travel CSV is missing {len(missing)} (person, center) pair(s); "
            f"first missing: {sample}"
        )

    return travel


# ---------------------------------------------------------------------------
# Local primitive — enum value validator (paper-style discrete strings)
# ---------------------------------------------------------------------------


def _require_enum_value(
    value: object,
    field: str,
    allowed: set[str],
    row_label: str,
) -> str:
    s = str(value).strip().lower()
    if s not in allowed:
        raise ValueError(f"{row_label}: {field} must be one of {sorted(allowed)!r}, got {s!r}")
    return s
