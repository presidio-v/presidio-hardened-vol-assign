"""Fuzzy Inference Systems for the humanitarian allocation model.

Three Mamdani FIS with centroid defuzzification, mirroring the structure of the
ED-staffing FIS in ``fis.py``:

    FIS-A — Fairness in People Prioritization
            inputs:  vulnerability (0-10), centre service_level (0-10),
                     distance person->centre (0-100 km)
            output:  unfairness (0-1) — high when a highly vulnerable person is
                     sent to a low-service and/or distant centre.

    FIS-B — Transportation Feasibility
            inputs:  distance (0-100 km), person mobility (0-10),
                     centre road_accessibility (0-10)
            output:  transport_infeasibility (0-1) — high when far, low mobility,
                     poor road access.

    FIS-C — Center Allocation Balance
            input:   centre utilisation = assigned load / capacity (0-2, i.e.
                     0-200 %)
            output:  overcrowding (0-1) — low when under-utilised, high when
                     over capacity.

The membership functions (Table 2 below) are triangular/trapezoidal over three
linguistic levels; the rule bases (Table 1 below, ``FAIRNESS_RULES`` /
``TRANSPORT_RULES`` / ``BALANCE_RULES``) are explicit Mamdani rule tables. The
values here are a self-consistent synthetic specification for the humanitarian
model; they can be regenerated or replaced wholesale by editing the tables
without touching any solver code.

Table 2 — Membership functions
    0-10 variables (vulnerability, service_level, mobility, road_accessibility):
        low    = trap[0, 0, 2, 5]      medium = tri[2, 5, 8]    high = trap[5, 8, 10, 10]
    distance (0-100 km):
        near   = trap[0, 0, 15, 35]    medium = tri[15, 50, 85] far  = trap[65, 85, 100, 100]
    utilisation (0-2):
        under  = trap[0, 0, 0.6, 0.9]  balanced = tri[0.7, 1.0, 1.3]  over = trap[1.1, 1.4, 2, 2]
    outputs (0-1):
        low    = trap[0, 0, 0.2, 0.45] medium = tri[0.2, 0.5, 0.8]    high = trap[0.55, 0.8, 1, 1]

Public API:
    evaluate_fairness(vulnerability, service_level, distance)       -> float [0,1]
    evaluate_transport(distance, mobility, road_accessibility)      -> float [0,1]
    evaluate_overcrowding(utilisation)                             -> float [0,1]
"""

from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Linguistic-level labels
_LOW, _MED, _HIGH = "low", "medium", "high"

# ---------------------------------------------------------------------------
# Table 1 — Rule bases (explicit Mamdani tables)
#
# Each table maps antecedent linguistic levels to the output level. They are
# the single source of truth for the rule bases; the control systems below are
# assembled directly from them.
# ---------------------------------------------------------------------------

# FIS-A: unfairness[vulnerability][service_level][distance]
# Rationale: unfairness bites only when a *vulnerable* person is placed poorly
# (low service and/or far). A low-priority person is treated fairly almost
# regardless of placement.
FAIRNESS_RULES: dict[str, dict[str, dict[str, str]]] = {
    _LOW: {
        _LOW: {"near": _LOW, "medium": _LOW, "far": _MED},
        _MED: {"near": _LOW, "medium": _LOW, "far": _LOW},
        _HIGH: {"near": _LOW, "medium": _LOW, "far": _LOW},
    },
    _MED: {
        _LOW: {"near": _MED, "medium": _MED, "far": _HIGH},
        _MED: {"near": _LOW, "medium": _MED, "far": _MED},
        _HIGH: {"near": _LOW, "medium": _LOW, "far": _MED},
    },
    _HIGH: {
        _LOW: {"near": _MED, "medium": _HIGH, "far": _HIGH},
        _MED: {"near": _LOW, "medium": _MED, "far": _HIGH},
        _HIGH: {"near": _LOW, "medium": _LOW, "far": _MED},
    },
}

# FIS-B: transport_infeasibility[distance][mobility][road_accessibility]
# Rationale: infeasibility grows with distance and falls with personal mobility
# and road access; distance dominates.
TRANSPORT_RULES: dict[str, dict[str, dict[str, str]]] = {
    "near": {
        _LOW: {_LOW: _MED, _MED: _LOW, _HIGH: _LOW},
        _MED: {_LOW: _LOW, _MED: _LOW, _HIGH: _LOW},
        _HIGH: {_LOW: _LOW, _MED: _LOW, _HIGH: _LOW},
    },
    "medium": {
        _LOW: {_LOW: _HIGH, _MED: _MED, _HIGH: _MED},
        _MED: {_LOW: _MED, _MED: _MED, _HIGH: _LOW},
        _HIGH: {_LOW: _MED, _MED: _LOW, _HIGH: _LOW},
    },
    "far": {
        _LOW: {_LOW: _HIGH, _MED: _HIGH, _HIGH: _MED},
        _MED: {_LOW: _HIGH, _MED: _MED, _HIGH: _MED},
        _HIGH: {_LOW: _MED, _MED: _MED, _HIGH: _LOW},
    },
}

# FIS-C: overcrowding[utilisation]
BALANCE_RULES: dict[str, str] = {
    "under": _LOW,
    "balanced": _MED,
    "over": _HIGH,
}

# ---------------------------------------------------------------------------
# Universe arrays
# ---------------------------------------------------------------------------
_U_SCORE10 = np.linspace(0, 10, 101)  # any 0-10 variable
_U_DIST = np.linspace(0, 100, 101)  # distance in km
_U_UTIL = np.linspace(0, 2, 101)  # centre utilisation ratio (0-200 %)
_U_OUT = np.linspace(0, 1, 101)  # output 0-1

_EPS = 1e-4


# ---------------------------------------------------------------------------
# Table 2 — Membership-function builders
# ---------------------------------------------------------------------------


def _lmh_10(var: ctrl.Antecedent) -> None:
    """Low / Medium / High for a 0-10 variable."""
    var["low"] = fuzz.trapmf(var.universe, [0, 0, 2, 5])
    var["medium"] = fuzz.trimf(var.universe, [2, 5, 8])
    var["high"] = fuzz.trapmf(var.universe, [5, 8, 10, 10])


def _nmf_distance(var: ctrl.Antecedent) -> None:
    """Near / Medium / Far for distance (0-100 km)."""
    var["near"] = fuzz.trapmf(var.universe, [0, 0, 15, 35])
    var["medium"] = fuzz.trimf(var.universe, [15, 50, 85])
    var["far"] = fuzz.trapmf(var.universe, [65, 85, 100, 100])


def _util_levels(var: ctrl.Antecedent) -> None:
    """Under / Balanced / Over for centre utilisation (0-2)."""
    var["under"] = fuzz.trapmf(var.universe, [0, 0, 0.6, 0.9])
    var["balanced"] = fuzz.trimf(var.universe, [0.7, 1.0, 1.3])
    var["over"] = fuzz.trapmf(var.universe, [1.1, 1.4, 2, 2])


def _lmh_out(var: ctrl.Consequent) -> None:
    """Low / Medium / High for a 0-1 output."""
    var["low"] = fuzz.trapmf(var.universe, [0, 0, 0.2, 0.45])
    var["medium"] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
    var["high"] = fuzz.trapmf(var.universe, [0.55, 0.8, 1, 1])


# ---------------------------------------------------------------------------
# Control-system assembly from the rule tables
# ---------------------------------------------------------------------------


def _build_fairness_system() -> ctrl.ControlSystem:
    vulnerability = ctrl.Antecedent(_U_SCORE10, "vulnerability")
    service = ctrl.Antecedent(_U_SCORE10, "service_level")
    distance = ctrl.Antecedent(_U_DIST, "distance")
    unfairness = ctrl.Consequent(_U_OUT, "unfairness")

    _lmh_10(vulnerability)
    _lmh_10(service)
    _nmf_distance(distance)
    _lmh_out(unfairness)

    rules = [
        ctrl.Rule(
            vulnerability[v_lvl] & service[s_lvl] & distance[d_lvl],
            unfairness[out],
        )
        for v_lvl, s_map in FAIRNESS_RULES.items()
        for s_lvl, d_map in s_map.items()
        for d_lvl, out in d_map.items()
    ]
    return ctrl.ControlSystem(rules)


def _build_transport_system() -> ctrl.ControlSystem:
    distance = ctrl.Antecedent(_U_DIST, "distance")
    mobility = ctrl.Antecedent(_U_SCORE10, "mobility")
    road = ctrl.Antecedent(_U_SCORE10, "road_accessibility")
    infeasibility = ctrl.Consequent(_U_OUT, "transport_infeasibility")

    _nmf_distance(distance)
    _lmh_10(mobility)
    _lmh_10(road)
    _lmh_out(infeasibility)

    rules = [
        ctrl.Rule(
            distance[d_lvl] & mobility[m_lvl] & road[r_lvl],
            infeasibility[out],
        )
        for d_lvl, m_map in TRANSPORT_RULES.items()
        for m_lvl, r_map in m_map.items()
        for r_lvl, out in r_map.items()
    ]
    return ctrl.ControlSystem(rules)


def _build_balance_system() -> ctrl.ControlSystem:
    utilisation = ctrl.Antecedent(_U_UTIL, "utilisation")
    overcrowding = ctrl.Consequent(_U_OUT, "overcrowding")

    _util_levels(utilisation)
    _lmh_out(overcrowding)

    rules = [
        ctrl.Rule(utilisation[u_lvl], overcrowding[out]) for u_lvl, out in BALANCE_RULES.items()
    ]
    return ctrl.ControlSystem(rules)


# ---------------------------------------------------------------------------
# Module-level system instances (built once at import)
# ---------------------------------------------------------------------------

_FAIRNESS_SYSTEM = _build_fairness_system()
_TRANSPORT_SYSTEM = _build_transport_system()
_BALANCE_SYSTEM = _build_balance_system()


# ---------------------------------------------------------------------------
# Public evaluation functions
# ---------------------------------------------------------------------------


def _run_sim(system: ctrl.ControlSystem, inputs: dict[str, float]) -> float:
    """Create a fresh simulation, feed inputs, compute, return clipped output.

    Returns 0.5 (neutral) if the simulation fails (e.g. all-zero membership).
    """
    sim = ctrl.ControlSystemSimulation(system)
    for name, value in inputs.items():
        sim.input[name] = value
    try:
        sim.compute()
        key = next(iter(system.consequents)).label
        return float(np.clip(float(sim.output[key]), 0.0, 1.0))
    except Exception:  # noqa: BLE001
        return 0.5


def evaluate_fairness(vulnerability: float, service_level: float, distance: float) -> float:
    """Return unfairness of prioritisation in [0, 1]."""
    return _run_sim(
        _FAIRNESS_SYSTEM,
        {
            "vulnerability": float(np.clip(vulnerability, _EPS, 10 - _EPS)),
            "service_level": float(np.clip(service_level, _EPS, 10 - _EPS)),
            "distance": float(np.clip(distance, _EPS, 100 - _EPS)),
        },
    )


def evaluate_transport(distance: float, mobility: float, road_accessibility: float) -> float:
    """Return transportation infeasibility in [0, 1]."""
    return _run_sim(
        _TRANSPORT_SYSTEM,
        {
            "distance": float(np.clip(distance, _EPS, 100 - _EPS)),
            "mobility": float(np.clip(mobility, _EPS, 10 - _EPS)),
            "road_accessibility": float(np.clip(road_accessibility, _EPS, 10 - _EPS)),
        },
    )


def evaluate_overcrowding(utilisation: float) -> float:
    """Return centre overcrowding/imbalance in [0, 1] for a utilisation ratio."""
    return _run_sim(
        _BALANCE_SYSTEM,
        {"utilisation": float(np.clip(utilisation, _EPS, 2 - _EPS))},
    )
