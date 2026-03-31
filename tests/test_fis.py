"""Tests for the fuzzy inference systems."""

from __future__ import annotations

import pytest

from presidio_vol_assign.fis import (
    compute_workload,
    evaluate_fis1,
    evaluate_fis2,
    evaluate_fis3,
)

# ---------------------------------------------------------------------------
# Output range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "patients,emergency,skill",
    [
        (80, 9.0, 2.0),  # critical ED, low skill → high importance
        (10, 1.0, 9.0),  # quiet ED, high skill → low importance
        (50, 5.0, 5.0),  # mid-range
        (0, 0, 0),  # boundary
        (100, 10, 10),  # boundary
    ],
)
def test_fis1_output_in_range(patients, emergency, skill) -> None:
    result = evaluate_fis1(patients, emergency, skill)
    assert 0.0 <= result <= 1.0, f"fis1({patients},{emergency},{skill}) = {result}"


@pytest.mark.parametrize(
    "patients,emergency,skill",
    [
        (80, 9.0, 2.0),
        (10, 1.0, 9.0),
        (50, 5.0, 5.0),
        (0, 0, 0),
        (100, 10, 10),
    ],
)
def test_fis2_output_in_range(patients, emergency, skill) -> None:
    result = evaluate_fis2(patients, emergency, skill)
    assert 0.0 <= result <= 1.0, f"fis2({patients},{emergency},{skill}) = {result}"


@pytest.mark.parametrize(
    "distance,workload,tolerance",
    [
        (90, 9.0, 1.0),  # far, high workload, low tolerance → high dissatisfaction
        (5, 1.0, 9.0),  # near, low workload, high tolerance → low dissatisfaction
        (50, 5.0, 5.0),  # mid-range
        (0, 0, 0),  # boundary
        (100, 10, 10),  # boundary
    ],
)
def test_fis3_output_in_range(distance, workload, tolerance) -> None:
    result = evaluate_fis3(distance, workload, tolerance)
    assert 0.0 <= result <= 1.0, f"fis3({distance},{workload},{tolerance}) = {result}"


# ---------------------------------------------------------------------------
# Monotonicity / ordering
# ---------------------------------------------------------------------------


def test_fis1_critical_worse_than_quiet() -> None:
    """Critical ED with low-skill volunteer should have higher importance than quiet ED."""
    critical = evaluate_fis1(90, 9.0, 2.0)
    quiet = evaluate_fis1(10, 1.0, 9.0)
    assert critical > quiet, f"expected critical({critical:.3f}) > quiet({quiet:.3f})"


def test_fis2_critical_worse_than_quiet() -> None:
    critical = evaluate_fis2(90, 9.0, 2.0)
    quiet = evaluate_fis2(10, 1.0, 9.0)
    assert critical > quiet


def test_fis3_far_worse_than_near() -> None:
    """Far volunteer with high workload and low tolerance is more dissatisfied."""
    dissatisfied = evaluate_fis3(90, 9.0, 1.0)
    satisfied = evaluate_fis3(5, 1.0, 9.0)
    assert dissatisfied > satisfied, (
        f"expected dissatisfied({dissatisfied:.3f}) > satisfied({satisfied:.3f})"
    )


def test_fis1_high_skill_reduces_importance() -> None:
    """At a critical ED, higher volunteer skill should reduce importance (better match)."""
    low_skill = evaluate_fis1(80, 8.0, 2.0)
    high_skill = evaluate_fis1(80, 8.0, 9.0)
    assert low_skill > high_skill


def test_fis3_high_tolerance_reduces_dissatisfaction() -> None:
    """Higher difficulty tolerance should reduce dissatisfaction at a far, busy ED."""
    low_tol = evaluate_fis3(80, 8.0, 1.0)
    high_tol = evaluate_fis3(80, 8.0, 9.0)
    assert low_tol > high_tol


# ---------------------------------------------------------------------------
# Statelessness: repeated calls return the same value
# ---------------------------------------------------------------------------


def test_fis1_stateless() -> None:
    r1 = evaluate_fis1(60, 7.0, 4.0)
    r2 = evaluate_fis1(60, 7.0, 4.0)
    assert r1 == pytest.approx(r2)


def test_fis3_stateless() -> None:
    r1 = evaluate_fis3(40, 6.0, 5.0)
    r2 = evaluate_fis3(40, 6.0, 5.0)
    assert r1 == pytest.approx(r2)


# ---------------------------------------------------------------------------
# compute_workload
# ---------------------------------------------------------------------------


def test_compute_workload_range() -> None:
    wl = compute_workload(50, 5.0)
    assert 0.0 <= wl <= 10.0


def test_compute_workload_zero() -> None:
    assert compute_workload(0, 0.0) == pytest.approx(0.0)


def test_compute_workload_max() -> None:
    assert compute_workload(100, 10.0) == pytest.approx(10.0)


def test_compute_workload_midpoint() -> None:
    # patients=50 → normalised=5; emergency=5 → average=5
    assert compute_workload(50, 5.0) == pytest.approx(5.0)
