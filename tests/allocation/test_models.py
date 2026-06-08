"""Unit tests for allocation/models.py — enum scoring + Person.age_score."""

from __future__ import annotations

import pytest

from presidio_vol_assign.allocation.models import (
    AllocationSolution,
    DisabilityStatus,
    HazardLevel,
    InjuryLevel,
    LivingStatus,
    Person,
    RoadCondition,
)


class TestEnumScoring:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (DisabilityStatus.NONE, 0.0),
            (DisabilityStatus.MINOR, 0.5),
            (DisabilityStatus.SEVERE, 1.0),
        ],
    )
    def test_disability_score(self, status, expected):
        assert status.score == expected

    @pytest.mark.parametrize(
        "status,expected",
        [
            (InjuryLevel.NONE, 0.0),
            (InjuryLevel.MINOR, 0.25),
            (InjuryLevel.MODERATE, 0.5),
            (InjuryLevel.SERIOUS, 0.75),
            (InjuryLevel.LIFE_THREATENING, 1.0),
        ],
    )
    def test_injury_score(self, status, expected):
        assert status.score == expected

    def test_living_status(self):
        assert LivingStatus.WITH_SUPPORT.score == 0.0
        assert LivingStatus.ALONE.score == 1.0

    @pytest.mark.parametrize(
        "rcs,expected",
        [
            (RoadCondition.CLEAR, 0.1),
            (RoadCondition.PARTIALLY_BLOCKED, 0.5),
            (RoadCondition.BLOCKED, 1.0),
        ],
    )
    def test_road_condition(self, rcs, expected):
        assert rcs.score == expected

    def test_hazard_levels_monotone(self):
        scores = [
            HazardLevel.NONE.score,
            HazardLevel.MINOR.score,
            HazardLevel.MODERATE.score,
            HazardLevel.SIGNIFICANT.score,
            HazardLevel.EXTREME.score,
        ]
        assert scores == sorted(scores)
        assert scores[0] == 0.05
        assert scores[-1] == 1.0


class TestAgeScore:
    """Piecewise AS(age) per ATRes §3.2.1."""

    @pytest.mark.parametrize(
        "age,expected",
        [
            (0, 1.0),
            (10, 0.6),
            (24, 0.04),
            (25, 0.0),
            (40, 0.0),
            (60, 0.0),
            (75, 0.5),
            (90, 1.0),
            (95, 1.0),  # plateau above 90
        ],
    )
    def test_age_score_piecewise(self, age, expected):
        p = Person(
            "p",
            age,
            DisabilityStatus.NONE,
            InjuryLevel.NONE,
            LivingStatus.WITH_SUPPORT,
            0,
            48,
        )
        assert p.age_score == pytest.approx(expected, abs=1e-9)


class TestAllocationSolutionFitness:
    def test_4obj_fitness_order(self):
        sol = AllocationSolution(
            allocations=[],
            objectives_count=4,
            mn_ulpp=10.0,
            mn_trd=20.0,
            mn_rpd=30.0,
            mn_cail=40.0,
        )
        assert sol.fitness == (10.0, 20.0, 30.0, 40.0)

    def test_3obj_fitness_order(self):
        sol = AllocationSolution(
            allocations=[],
            objectives_count=3,
            mn_ulpp=10.0,
            mn_til=25.0,
            mn_cail=40.0,
        )
        assert sol.fitness == (10.0, 25.0, 40.0)
