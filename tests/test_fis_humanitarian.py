"""Tests for the humanitarian Fuzzy Inference Systems.

These assert the FIS are well-formed and directionally correct (monotone in the
intended sense). Exact values are intentionally not pinned: the membership
functions and rule bases are placeholders pending the paper's final tables.
"""

from __future__ import annotations

from presidio_vol_assign.fis_humanitarian import (
    evaluate_fairness,
    evaluate_overcrowding,
    evaluate_transport,
)

# ---------------------------------------------------------------------------
# Output ranges
# ---------------------------------------------------------------------------


def test_outputs_in_unit_range() -> None:
    assert 0.0 <= evaluate_fairness(5.0, 5.0, 50.0) <= 1.0
    assert 0.0 <= evaluate_transport(50.0, 5.0, 5.0) <= 1.0
    assert 0.0 <= evaluate_overcrowding(1.0) <= 1.0


# ---------------------------------------------------------------------------
# FIS-A — fairness
# ---------------------------------------------------------------------------


def test_fairness_worse_for_vulnerable_poorly_served() -> None:
    # High vulnerability + low service + far  >>  low vulnerability + high service + near
    unfair = evaluate_fairness(vulnerability=9.0, service_level=1.0, distance=90.0)
    fair = evaluate_fairness(vulnerability=1.0, service_level=9.0, distance=5.0)
    assert unfair > fair


def test_fairness_low_when_vulnerable_well_served() -> None:
    # A vulnerable person sent to a high-service, near centre is treated fairly.
    assert evaluate_fairness(vulnerability=9.0, service_level=9.0, distance=5.0) < 0.5


# ---------------------------------------------------------------------------
# FIS-B — transportation feasibility
# ---------------------------------------------------------------------------


def test_transport_worse_when_far_immobile_poor_roads() -> None:
    hard = evaluate_transport(distance=95.0, mobility=1.0, road_accessibility=1.0)
    easy = evaluate_transport(distance=3.0, mobility=9.0, road_accessibility=9.0)
    assert hard > easy


# ---------------------------------------------------------------------------
# FIS-C — centre overcrowding
# ---------------------------------------------------------------------------


def test_overcrowding_monotonic_in_utilisation() -> None:
    low = evaluate_overcrowding(0.3)
    mid = evaluate_overcrowding(1.0)
    high = evaluate_overcrowding(1.8)
    assert low < high
    assert low <= mid <= high
