"""Unit tests for allocation/fis.py — derived helpers + sign correctness."""

from __future__ import annotations

import pytest

from presidio_vol_assign.allocation.fis import (
    compute_rws,
    compute_vs,
    evaluate_fis1_ulpp,
    evaluate_fis2_til,
    evaluate_fis2a_trd,
    evaluate_fis2b_rpd,
    evaluate_fis3_cail,
)
from presidio_vol_assign.allocation.models import (
    DisabilityStatus,
    HazardLevel,
    InjuryLevel,
    LivingStatus,
    Person,
    RoadCondition,
    TravelInfo,
    Weights,
)


def _person(**overrides):
    base = dict(
        person_id="p",
        age=40,
        disability_status=DisabilityStatus.NONE,
        injury_level=InjuryLevel.NONE,
        living_status=LivingStatus.WITH_SUPPORT,
        infrastructure_damage_level=0.0,
        resource_time_remaining=48.0,
    )
    base.update(overrides)
    return Person(**base)


class TestComputeVS:
    def test_healthy_baseline_is_zero(self):
        assert compute_vs(_person(), Weights()) == 0.0

    def test_max_vulnerability(self):
        p = _person(
            age=95,
            disability_status=DisabilityStatus.SEVERE,
            injury_level=InjuryLevel.LIFE_THREATENING,
            living_status=LivingStatus.ALONE,
        )
        assert compute_vs(p, Weights()) == pytest.approx(1.0, abs=1e-9)

    def test_zero_weights_is_zero(self):
        p = _person(
            age=95,
            disability_status=DisabilityStatus.SEVERE,
            injury_level=InjuryLevel.LIFE_THREATENING,
            living_status=LivingStatus.ALONE,
        )
        w = Weights(was=0, wds=0, wil=0, wls=0)
        assert compute_vs(p, w) == 0.0


class TestComputeRWS:
    """ATRes Eq. (5) sign-corrected: high RWS = safe route."""

    def test_clear_safe_route_is_high(self):
        t = TravelInfo("p", "c", 30, RoadCondition.CLEAR, HazardLevel.NONE)
        rws = compute_rws(t, Weights())
        assert rws > 0.85

    def test_blocked_extreme_route_is_zero(self):
        t = TravelInfo("p", "c", 30, RoadCondition.BLOCKED, HazardLevel.EXTREME)
        assert compute_rws(t, Weights()) == 0.0

    def test_zero_weights_is_one(self):
        # No weights = no danger contribution = perfectly safe by convention
        t = TravelInfo("p", "c", 30, RoadCondition.BLOCKED, HazardLevel.EXTREME)
        assert compute_rws(t, Weights(wrc=0, wph=0)) == 1.0


class TestFISMonotonicity:
    """Sanity: FIS outputs respect intuitive monotonicity."""

    def test_ulpp_higher_for_more_vulnerable(self):
        # High VS, severe IDL, short RTR → very high ULPP
        high = evaluate_fis1_ulpp(0.9, 90.0, 4.0)
        # Low VS, minor IDL, long RTR → very low ULPP
        low = evaluate_fis1_ulpp(0.1, 10.0, 40.0)
        assert high > low + 30  # broad sanity gap

    def test_til_higher_for_worse_route(self):
        worse = evaluate_fis2_til(120.0, 0.05)  # long + unsafe (low RWS)
        better = evaluate_fis2_til(15.0, 0.95)  # short + safe (high RWS)
        assert worse > better + 30

    def test_trd_higher_for_worse_road(self):
        worse = evaluate_fis2a_trd(0.95, 0.95)
        better = evaluate_fis2a_trd(0.05, 0.05)
        assert worse > better + 30

    def test_rpd_monotone_in_td(self):
        short = evaluate_fis2b_rpd(15.0)
        long = evaluate_fis2b_rpd(150.0)
        assert long > short + 20

    def test_cail_higher_for_overloaded_center(self):
        worse = evaluate_fis3_cail(85.0, 75.0, 120.0)  # crowded, fast depletion, long trip
        better = evaluate_fis3_cail(10.0, 10.0, 15.0)  # empty, slow depletion, short trip
        assert worse > better + 30
