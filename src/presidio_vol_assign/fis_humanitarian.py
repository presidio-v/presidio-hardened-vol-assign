"""Fuzzy Inference Systems for the humanitarian allocation model (v0.2.0).

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

.. warning::
    The membership-function break-points and rule bases below are **structural
    placeholders** generated from monotone severity heuristics. They must be
    replaced with the paper's final Tables 1-2 before publication. The shapes
    (triangular/trapezoidal, three linguistic levels, 27/27/3 rules) match the
    intended model; only the exact numbers are provisional.

Public API:
    evaluate_fairness(vulnerability, service_level, distance)       -> float [0,1]
    evaluate_transport(distance, mobility, road_accessibility)      -> float [0,1]
    evaluate_overcrowding(utilisation)                             -> float [0,1]
"""

from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------------------------------------------------------------------
# Universe arrays
# ---------------------------------------------------------------------------
_U_SCORE10 = np.linspace(0, 10, 101)  # any 0-10 variable
_U_DIST = np.linspace(0, 100, 101)  # distance in km
_U_UTIL = np.linspace(0, 2, 101)  # centre utilisation ratio (0-200 %)
_U_OUT = np.linspace(0, 1, 101)  # output 0-1

_EPS = 1e-4


# ---------------------------------------------------------------------------
# Membership-function builders (PLACEHOLDER break-points — see module warning)
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
# Rule-base helpers — map a monotone severity score to an output level.
# (PLACEHOLDER logic; the paper's explicit rule tables replace this.)
# ---------------------------------------------------------------------------

_LEVELS_3 = ("low", "medium", "high")


def _bucket(score: float, low_max: float, high_min: float) -> str:
    """Bucket a numeric severity into low / medium / high output level."""
    if score <= low_max:
        return "low"
    if score >= high_min:
        return "high"
    return "medium"


# ---------------------------------------------------------------------------
# FIS-A — Fairness in People Prioritization
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

    vmap = {"low": 0, "medium": 1, "high": 2}
    dmap = {"near": 0, "medium": 1, "far": 2}

    rules = []
    for v_lvl in _LEVELS_3:
        for s_lvl in _LEVELS_3:
            for d_lvl in ("near", "medium", "far"):
                # Poor allocation = low service + far distance; unfairness only
                # bites when the person is actually a priority (high vulnerability).
                poor = (2 - vmap[s_lvl]) + dmap[d_lvl]  # 0..4
                severity = vmap[v_lvl] * poor  # 0..8
                out = _bucket(severity, low_max=1, high_min=5)
                rules.append(
                    ctrl.Rule(
                        vulnerability[v_lvl] & service[s_lvl] & distance[d_lvl],
                        unfairness[out],
                    )
                )
    return ctrl.ControlSystem(rules)


# ---------------------------------------------------------------------------
# FIS-B — Transportation Feasibility
# ---------------------------------------------------------------------------


def _build_transport_system() -> ctrl.ControlSystem:
    distance = ctrl.Antecedent(_U_DIST, "distance")
    mobility = ctrl.Antecedent(_U_SCORE10, "mobility")
    road = ctrl.Antecedent(_U_SCORE10, "road_accessibility")
    infeasibility = ctrl.Consequent(_U_OUT, "transport_infeasibility")

    _nmf_distance(distance)
    _lmh_10(mobility)
    _lmh_10(road)
    _lmh_out(infeasibility)

    dmap = {"near": 0, "medium": 1, "far": 2}
    lmap = {"low": 0, "medium": 1, "high": 2}

    rules = []
    for d_lvl in ("near", "medium", "far"):
        for m_lvl in _LEVELS_3:
            for r_lvl in _LEVELS_3:
                # Infeasibility grows with distance (weighted), and with poor
                # personal mobility and poor road access.
                severity = dmap[d_lvl] * 2 + (2 - lmap[m_lvl]) + (2 - lmap[r_lvl])  # 0..8
                out = _bucket(severity, low_max=1, high_min=5)
                rules.append(
                    ctrl.Rule(
                        distance[d_lvl] & mobility[m_lvl] & road[r_lvl],
                        infeasibility[out],
                    )
                )
    return ctrl.ControlSystem(rules)


# ---------------------------------------------------------------------------
# FIS-C — Center Allocation Balance
# ---------------------------------------------------------------------------


def _build_balance_system() -> ctrl.ControlSystem:
    utilisation = ctrl.Antecedent(_U_UTIL, "utilisation")
    overcrowding = ctrl.Consequent(_U_OUT, "overcrowding")

    _util_levels(utilisation)
    _lmh_out(overcrowding)

    rules = [
        ctrl.Rule(utilisation["under"], overcrowding["low"]),
        ctrl.Rule(utilisation["balanced"], overcrowding["medium"]),
        ctrl.Rule(utilisation["over"], overcrowding["high"]),
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
