"""Shared fixtures for allocation tests.

Builds a minimal but solvable problem instance directly in memory and as
on-disk CSVs. Eight people, three centers, n_dir=4 — enough for a
non-trivial Pareto front but tiny enough for fast tests.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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
    Weights,
)


@pytest.fixture
def people() -> list[Person]:
    return [
        Person("P0", 75, DisabilityStatus.MINOR, InjuryLevel.MODERATE, LivingStatus.ALONE, 60, 12),
        Person(
            "P1", 30, DisabilityStatus.NONE, InjuryLevel.NONE, LivingStatus.WITH_SUPPORT, 10, 36
        ),
        Person("P2", 50, DisabilityStatus.SEVERE, InjuryLevel.SERIOUS, LivingStatus.ALONE, 75, 6),
        Person(
            "P3", 8, DisabilityStatus.NONE, InjuryLevel.MINOR, LivingStatus.WITH_SUPPORT, 30, 24
        ),
        Person(
            "P4",
            90,
            DisabilityStatus.SEVERE,
            InjuryLevel.LIFE_THREATENING,
            LivingStatus.ALONE,
            95,
            2,
        ),
        Person("P5", 45, DisabilityStatus.MINOR, InjuryLevel.MODERATE, LivingStatus.ALONE, 50, 18),
        Person(
            "P6", 60, DisabilityStatus.NONE, InjuryLevel.MINOR, LivingStatus.WITH_SUPPORT, 20, 30
        ),
        Person("P7", 25, DisabilityStatus.SEVERE, InjuryLevel.NONE, LivingStatus.ALONE, 70, 8),
    ]


@pytest.fixture
def centers() -> list[ReliefCenter]:
    return [
        ReliefCenter("C0", 60, 40),
        ReliefCenter("C1", 30, 70),
        ReliefCenter("C2", 80, 20),
    ]


@pytest.fixture
def travel(people, centers) -> dict[tuple[str, str], TravelInfo]:
    out: dict[tuple[str, str], TravelInfo] = {}
    td_by_center = {"C0": 30.0, "C1": 60.0, "C2": 110.0}
    for p in people:
        for c in centers:
            out[(p.person_id, c.center_id)] = TravelInfo(
                p.person_id,
                c.center_id,
                td_by_center[c.center_id],
                RoadCondition.PARTIALLY_BLOCKED,
                HazardLevel.MODERATE,
            )
    return out


@pytest.fixture
def problem(people, centers, travel) -> AllocationProblem:
    return AllocationProblem(people=people, centers=centers, travel=travel, n_dir=4)


@pytest.fixture
def base_config() -> AllocationConfig:
    return AllocationConfig(
        solver=AllocationSolverType.NSGA2,
        objectives=4,
        weights=Weights(),
        pop_size=20,
        generations=10,
        seed=42,
    )


@pytest.fixture
def csv_dir(tmp_path: Path, people, centers, travel) -> Path:
    """Write fixture data as the three required CSVs in a tmp_path."""
    pd.DataFrame(
        [
            {
                "person_id": p.person_id,
                "age": p.age,
                "disability_status": p.disability_status.value,
                "injury_level": p.injury_level.value,
                "living_status": p.living_status.value,
                "idl": p.infrastructure_damage_level,
                "rtr": p.resource_time_remaining,
            }
            for p in people
        ]
    ).to_csv(tmp_path / "people.csv", index=False)

    pd.DataFrame(
        [
            {
                "center_id": c.center_id,
                "cor": c.center_occupancy_rate,
                "rdr": c.resource_depletion_rate,
            }
            for c in centers
        ]
    ).to_csv(tmp_path / "centers.csv", index=False)

    pd.DataFrame(
        [
            {
                "person_id": t.person_id,
                "center_id": t.center_id,
                "td": t.travel_duration,
                "rcs": t.road_condition.value,
                "phs": t.possible_hazard.value,
            }
            for t in travel.values()
        ]
    ).to_csv(tmp_path / "travel.csv", index=False)

    return tmp_path
