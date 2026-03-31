"""Fuzzy Inference Systems for the volunteer assignment model.

Implements the three FIS described in Rabiei et al. (ESWA, 2023):
    FIS1 — importance of unmet triage-nurse need at an ED
    FIS2 — importance of unmet primary-ER-nurse need at an ED
    FIS3 — degree of volunteer preference dissatisfaction

All three use Mamdani inference with centroid defuzzification.
Membership functions are triangular / trapezoidal, matching the
three-level linguistic variables (Low / Medium / High) from the paper.

Control systems are built once at module import; each call to an
evaluate_* function creates a fresh ControlSystemSimulation (required
because ControlSystemSimulation is stateful and not thread-safe).

Public API:
    evaluate_fis1(num_patients, ed_emergency_level, volunteer_skill) -> float [0, 1]
    evaluate_fis2(num_patients, ed_emergency_level, volunteer_skill) -> float [0, 1]
    evaluate_fis3(distance_to_ed, workload, difficulty_tolerance)    -> float [0, 1]
    compute_workload(num_patients, ed_emergency_level)               -> float [0, 10]
"""

from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------------------------------------------------------------------
# Universe arrays (linspace avoids float-step drift from arange)
# ---------------------------------------------------------------------------
_U_PATIENTS = np.linspace(0, 100, 101)  # num_patients
_U_SCORE10 = np.linspace(0, 10, 101)  # any 0-10 variable
_U_DIST = np.linspace(0, 100, 101)  # distance in km
_U_OUT = np.linspace(0, 1, 101)  # output 0-1

# Clip epsilon: keep inputs slightly inside universe to prevent zero-firing
_EPS = 1e-4


# ---------------------------------------------------------------------------
# Shared membership-function builders
# ---------------------------------------------------------------------------


def _lmh_10(var: ctrl.Antecedent | ctrl.Consequent) -> None:
    """Add Low / Medium / High MFs to a 0-10 variable."""
    var["low"] = fuzz.trapmf(var.universe, [0, 0, 2, 5])
    var["medium"] = fuzz.trimf(var.universe, [2, 5, 8])
    var["high"] = fuzz.trapmf(var.universe, [5, 8, 10, 10])


def _lmh_patients(var: ctrl.Antecedent) -> None:
    """Add Low / Medium / High MFs to num_patients (0-100)."""
    var["low"] = fuzz.trapmf(var.universe, [0, 0, 20, 45])
    var["medium"] = fuzz.trimf(var.universe, [20, 50, 80])
    var["high"] = fuzz.trapmf(var.universe, [55, 80, 100, 100])


def _lmh_distance(var: ctrl.Antecedent) -> None:
    """Add Near / Medium / Far MFs for distance (0-100 km)."""
    var["near"] = fuzz.trapmf(var.universe, [0, 0, 15, 35])
    var["medium"] = fuzz.trimf(var.universe, [15, 50, 85])
    var["far"] = fuzz.trapmf(var.universe, [65, 85, 100, 100])


def _lmh_out(var: ctrl.Consequent) -> None:
    """Add Low / Medium / High MFs to a 0-1 output."""
    var["low"] = fuzz.trapmf(var.universe, [0, 0, 0.2, 0.45])
    var["medium"] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
    var["high"] = fuzz.trapmf(var.universe, [0.55, 0.8, 1, 1])


# ---------------------------------------------------------------------------
# FIS1 and FIS2 — importance of unmet nursing need
# (FIS1 = triage nurse, FIS2 = primary ER nurse; same rule structure)
# ---------------------------------------------------------------------------


def _build_need_importance_system(suffix: str) -> ctrl.ControlSystem:
    """Build the control system for FIS1 or FIS2.

    Importance is high when the ED is critical (many patients, high emergency)
    AND the assigned volunteer's skill is low (poor match → unmet need).
    """
    patients = ctrl.Antecedent(_U_PATIENTS, f"num_patients_{suffix}")
    emergency = ctrl.Antecedent(_U_SCORE10, f"ed_emergency_level_{suffix}")
    skill = ctrl.Antecedent(_U_SCORE10, f"volunteer_skill_{suffix}")
    importance = ctrl.Consequent(_U_OUT, f"importance_{suffix}")

    _lmh_patients(patients)
    _lmh_10(emergency)
    _lmh_10(skill)
    _lmh_out(importance)

    # 27 rules (3 patients × 3 emergency × 3 skill levels)
    rules = [
        # High patients
        ctrl.Rule(patients["high"] & emergency["high"] & skill["low"], importance["high"]),
        ctrl.Rule(patients["high"] & emergency["high"] & skill["medium"], importance["medium"]),
        ctrl.Rule(patients["high"] & emergency["high"] & skill["high"], importance["low"]),
        ctrl.Rule(patients["high"] & emergency["medium"] & skill["low"], importance["high"]),
        ctrl.Rule(patients["high"] & emergency["medium"] & skill["medium"], importance["medium"]),
        ctrl.Rule(patients["high"] & emergency["medium"] & skill["high"], importance["low"]),
        ctrl.Rule(patients["high"] & emergency["low"] & skill["low"], importance["medium"]),
        ctrl.Rule(patients["high"] & emergency["low"] & skill["medium"], importance["low"]),
        ctrl.Rule(patients["high"] & emergency["low"] & skill["high"], importance["low"]),
        # Medium patients
        ctrl.Rule(patients["medium"] & emergency["high"] & skill["low"], importance["high"]),
        ctrl.Rule(patients["medium"] & emergency["high"] & skill["medium"], importance["medium"]),
        ctrl.Rule(patients["medium"] & emergency["high"] & skill["high"], importance["low"]),
        ctrl.Rule(patients["medium"] & emergency["medium"] & skill["low"], importance["medium"]),
        ctrl.Rule(patients["medium"] & emergency["medium"] & skill["medium"], importance["medium"]),
        ctrl.Rule(patients["medium"] & emergency["medium"] & skill["high"], importance["low"]),
        ctrl.Rule(patients["medium"] & emergency["low"] & skill["low"], importance["low"]),
        ctrl.Rule(patients["medium"] & emergency["low"] & skill["medium"], importance["low"]),
        ctrl.Rule(patients["medium"] & emergency["low"] & skill["high"], importance["low"]),
        # Low patients
        ctrl.Rule(patients["low"] & emergency["high"] & skill["low"], importance["medium"]),
        ctrl.Rule(patients["low"] & emergency["high"] & skill["medium"], importance["low"]),
        ctrl.Rule(patients["low"] & emergency["high"] & skill["high"], importance["low"]),
        ctrl.Rule(patients["low"] & emergency["medium"] & skill["low"], importance["low"]),
        ctrl.Rule(patients["low"] & emergency["medium"] & skill["medium"], importance["low"]),
        ctrl.Rule(patients["low"] & emergency["medium"] & skill["high"], importance["low"]),
        ctrl.Rule(patients["low"] & emergency["low"] & skill["low"], importance["low"]),
        ctrl.Rule(patients["low"] & emergency["low"] & skill["medium"], importance["low"]),
        ctrl.Rule(patients["low"] & emergency["low"] & skill["high"], importance["low"]),
    ]
    return ctrl.ControlSystem(rules)


# ---------------------------------------------------------------------------
# FIS3 — volunteer preference dissatisfaction
# ---------------------------------------------------------------------------


def _build_preference_system() -> ctrl.ControlSystem:
    """Build the control system for FIS3.

    Dissatisfaction is high when the volunteer is far from the ED,
    workload is high, and the volunteer's difficulty tolerance is low.
    """
    distance = ctrl.Antecedent(_U_DIST, "distance_to_ed")
    workload = ctrl.Antecedent(_U_SCORE10, "workload")
    tolerance = ctrl.Antecedent(_U_SCORE10, "difficulty_tolerance")
    dissatisfaction = ctrl.Consequent(_U_OUT, "preference_dissatisfaction")

    _lmh_distance(distance)
    _lmh_10(workload)
    _lmh_10(tolerance)
    _lmh_out(dissatisfaction)

    rules = [
        # Far distance
        ctrl.Rule(distance["far"] & workload["high"] & tolerance["low"], dissatisfaction["high"]),
        ctrl.Rule(
            distance["far"] & workload["high"] & tolerance["medium"], dissatisfaction["high"]
        ),
        ctrl.Rule(
            distance["far"] & workload["high"] & tolerance["high"], dissatisfaction["medium"]
        ),
        ctrl.Rule(distance["far"] & workload["medium"] & tolerance["low"], dissatisfaction["high"]),
        ctrl.Rule(
            distance["far"] & workload["medium"] & tolerance["medium"], dissatisfaction["medium"]
        ),
        ctrl.Rule(distance["far"] & workload["medium"] & tolerance["high"], dissatisfaction["low"]),
        ctrl.Rule(distance["far"] & workload["low"] & tolerance["low"], dissatisfaction["medium"]),
        ctrl.Rule(distance["far"] & workload["low"] & tolerance["medium"], dissatisfaction["low"]),
        ctrl.Rule(distance["far"] & workload["low"] & tolerance["high"], dissatisfaction["low"]),
        # Medium distance
        ctrl.Rule(
            distance["medium"] & workload["high"] & tolerance["low"], dissatisfaction["medium"]
        ),
        ctrl.Rule(
            distance["medium"] & workload["high"] & tolerance["medium"], dissatisfaction["medium"]
        ),
        ctrl.Rule(
            distance["medium"] & workload["high"] & tolerance["high"], dissatisfaction["low"]
        ),
        ctrl.Rule(
            distance["medium"] & workload["medium"] & tolerance["low"], dissatisfaction["medium"]
        ),
        ctrl.Rule(
            distance["medium"] & workload["medium"] & tolerance["medium"], dissatisfaction["low"]
        ),
        ctrl.Rule(
            distance["medium"] & workload["medium"] & tolerance["high"], dissatisfaction["low"]
        ),
        ctrl.Rule(distance["medium"] & workload["low"] & tolerance["low"], dissatisfaction["low"]),
        ctrl.Rule(
            distance["medium"] & workload["low"] & tolerance["medium"], dissatisfaction["low"]
        ),
        ctrl.Rule(distance["medium"] & workload["low"] & tolerance["high"], dissatisfaction["low"]),
        # Near distance
        ctrl.Rule(
            distance["near"] & workload["high"] & tolerance["low"], dissatisfaction["medium"]
        ),
        ctrl.Rule(
            distance["near"] & workload["high"] & tolerance["medium"], dissatisfaction["low"]
        ),
        ctrl.Rule(distance["near"] & workload["high"] & tolerance["high"], dissatisfaction["low"]),
        ctrl.Rule(distance["near"] & workload["medium"] & tolerance["low"], dissatisfaction["low"]),
        ctrl.Rule(
            distance["near"] & workload["medium"] & tolerance["medium"], dissatisfaction["low"]
        ),
        ctrl.Rule(
            distance["near"] & workload["medium"] & tolerance["high"], dissatisfaction["low"]
        ),
        ctrl.Rule(distance["near"] & workload["low"] & tolerance["low"], dissatisfaction["low"]),
        ctrl.Rule(distance["near"] & workload["low"] & tolerance["medium"], dissatisfaction["low"]),
        ctrl.Rule(distance["near"] & workload["low"] & tolerance["high"], dissatisfaction["low"]),
    ]
    return ctrl.ControlSystem(rules)


# ---------------------------------------------------------------------------
# Module-level system instances (built once at import)
# ---------------------------------------------------------------------------

_FIS1_SYSTEM = _build_need_importance_system("fis1")
_FIS2_SYSTEM = _build_need_importance_system("fis2")
_FIS3_SYSTEM = _build_preference_system()


# ---------------------------------------------------------------------------
# Public evaluation functions
# ---------------------------------------------------------------------------


def _run_sim(system: ctrl.ControlSystem, inputs: dict[str, float]) -> float:
    """Create a fresh simulation, feed inputs, compute, return output.

    Returns 0.5 (neutral) if the simulation fails (e.g. all zero membership).
    """
    sim = ctrl.ControlSystemSimulation(system)
    for name, value in inputs.items():
        sim.input[name] = value
    try:
        sim.compute()
        # Output key is the single consequent in the system
        key = next(iter(system.consequents)).label
        result = float(sim.output[key])
        return float(np.clip(result, 0.0, 1.0))
    except Exception:  # noqa: BLE001
        return 0.5  # fallback for zero-firing edge cases


def evaluate_fis1(
    num_patients: float,
    ed_emergency_level: float,
    volunteer_skill: float,
) -> float:
    """Return importance of unmet triage-nurse need in [0, 1]."""
    return _run_sim(
        _FIS1_SYSTEM,
        {
            "num_patients_fis1": float(np.clip(num_patients, _EPS, 100 - _EPS)),
            "ed_emergency_level_fis1": float(np.clip(ed_emergency_level, _EPS, 10 - _EPS)),
            "volunteer_skill_fis1": float(np.clip(volunteer_skill, _EPS, 10 - _EPS)),
        },
    )


def evaluate_fis2(
    num_patients: float,
    ed_emergency_level: float,
    volunteer_skill: float,
) -> float:
    """Return importance of unmet primary-ER-nurse need in [0, 1]."""
    return _run_sim(
        _FIS2_SYSTEM,
        {
            "num_patients_fis2": float(np.clip(num_patients, _EPS, 100 - _EPS)),
            "ed_emergency_level_fis2": float(np.clip(ed_emergency_level, _EPS, 10 - _EPS)),
            "volunteer_skill_fis2": float(np.clip(volunteer_skill, _EPS, 10 - _EPS)),
        },
    )


def evaluate_fis3(
    distance_to_ed: float,
    workload: float,
    difficulty_tolerance: float,
) -> float:
    """Return volunteer preference dissatisfaction in [0, 1]."""
    return _run_sim(
        _FIS3_SYSTEM,
        {
            "distance_to_ed": float(np.clip(distance_to_ed, _EPS, 100 - _EPS)),
            "workload": float(np.clip(workload, _EPS, 10 - _EPS)),
            "difficulty_tolerance": float(np.clip(difficulty_tolerance, _EPS, 10 - _EPS)),
        },
    )


def compute_workload(num_patients: int, ed_emergency_level: float) -> float:
    """Derive a 0-10 workload score from ED characteristics.

    Combines normalised patient count and emergency level as equal-weight average.
    Used as the workload input to FIS3.
    """
    normalised_patients = (float(num_patients) / 100.0) * 10.0
    return (normalised_patients + float(ed_emergency_level)) / 2.0
