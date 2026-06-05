"""Tests for the humanitarian Fuzzy Inference Systems.

These assert the FIS are well-formed and directionally correct (monotone in the
intended sense), and pin the definitive rule tables and the worked example in
docs/fis-worked-example.md.
"""

from __future__ import annotations

import pytest

from presidio_vol_assign.fis_humanitarian import (
    BALANCE_RULES,
    FAIRNESS_RULES,
    TRANSPORT_RULES,
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


# ---------------------------------------------------------------------------
# Definitive rule tables (Table 1) — completeness + pinned worked-example values
# ---------------------------------------------------------------------------

_LEVELS = {"low", "medium", "high"}


def _leaf_outputs(table: dict) -> list[str]:
    out: list[str] = []
    for v in table.values():
        if isinstance(v, dict):
            out.extend(_leaf_outputs(v))
        else:
            out.append(v)
    return out


def test_rule_tables_are_complete() -> None:
    fa = _leaf_outputs(FAIRNESS_RULES)
    tb = _leaf_outputs(TRANSPORT_RULES)
    assert len(fa) == 27  # 3 vulnerability x 3 service x 3 distance
    assert len(tb) == 27  # 3 distance x 3 mobility x 3 road
    assert len(BALANCE_RULES) == 3
    assert set(fa) | set(tb) | set(BALANCE_RULES.values()) <= _LEVELS


def test_worked_example_values_match_docs() -> None:
    # Pins the values in docs/fis-worked-example.md (P1 -> C1, examples/small).
    assert evaluate_fairness(7.2, 5.6, 26.2) == pytest.approx(0.347, abs=0.01)
    assert evaluate_transport(26.2, 8.2, 4.2) == pytest.approx(0.333, abs=0.01)
    assert evaluate_overcrowding(1.30) == pytest.approx(0.814, abs=0.01)
