"""Input-turbulence harness for the allocation model (Paper B, RQ1).

This module owns one transformation: a clean ``AllocationProblem`` plus a
``PerturbationSpec`` become a *degraded* problem, deterministically, given the
random generator the caller supplies. It perturbs the raw input fields
(situational data collected mid-shock), which is a different axis from the
elicitation-weight sweep in ``allocation.sensitivity`` — that perturbs how the
decision-maker weighs criteria; this perturbs the data the decision is made on.

Design:
- Parse, don't validate: the target field and mode are resolved against a
  pinned registry at entry; an unknown field or a mode that does not fit the
  field's kind is denied with a raise, never coerced into a silent no-op.
- Determinism at the boundary: all randomness comes from the caller's
  ``numpy`` generator, so a run is reproducible from its seed.
- Values are clipped to each field's pinned plausible range, so a perturbation
  can degrade data but never manufacture an out-of-range input.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from enum import Enum

import numpy as np

from presidio_vol_assign.allocation.models import (
    AllocationProblem,
    DisabilityStatus,
    HazardLevel,
    InjuryLevel,
    LivingStatus,
    RoadCondition,
)


class TurbulenceMode(str, Enum):
    """How an input field is degraded."""

    NOISE = "noise"  # additive Gaussian, sigma = level * field range
    BIAS = "bias"  # systematic signed shift, level * field range
    MISSINGNESS = "missingness"  # with prob level, impute (median / mode)
    FLIP = "flip"  # categorical only: with prob level, relabel


# Pinned plausible ranges: used to scale noise/bias and to clip results, so a
# perturbation stays within the domain the FIS membership functions expect.
# (owner, attribute, low, high)
_CONTINUOUS: dict[str, tuple[str, str, float, float]] = {
    "age": ("person", "age", 0.0, 120.0),
    "infrastructure_damage_level": ("person", "infrastructure_damage_level", 0.0, 100.0),
    "resource_time_remaining": ("person", "resource_time_remaining", 0.0, 72.0),
    "center_occupancy_rate": ("center", "center_occupancy_rate", 0.0, 100.0),
    "resource_depletion_rate": ("center", "resource_depletion_rate", 0.0, 100.0),
    "travel_duration": ("travel", "travel_duration", 0.0, 180.0),
}

# (owner, attribute, enum type)
_CATEGORICAL: dict[str, tuple[str, str, type[Enum]]] = {
    "disability_status": ("person", "disability_status", DisabilityStatus),
    "injury_level": ("person", "injury_level", InjuryLevel),
    "living_status": ("person", "living_status", LivingStatus),
    "road_condition": ("travel", "road_condition", RoadCondition),
    "possible_hazard": ("travel", "possible_hazard", HazardLevel),
}

CONTINUOUS_FIELDS: tuple[str, ...] = tuple(_CONTINUOUS)
CATEGORICAL_FIELDS: tuple[str, ...] = tuple(_CATEGORICAL)


@dataclass(frozen=True)
class PerturbationSpec:
    """One turbulence perturbation: which field, which mode, how strong.

    ``level`` is a fraction of the field's plausible range for NOISE/BIAS
    (BIAS may be signed), or a probability in [0, 1] for MISSINGNESS/FLIP.
    """

    field: str
    mode: TurbulenceMode
    level: float


def _entities(problem: AllocationProblem, owner: str) -> list:
    if owner == "person":
        return problem.people
    if owner == "center":
        return problem.centers
    return list(problem.travel.values())


def _perturb_continuous(
    values: list[float],
    mode: TurbulenceMode,
    level: float,
    lo: float,
    hi: float,
    rng: np.random.Generator,
) -> list[float]:
    arr = np.asarray(values, dtype=float)
    span = hi - lo
    if mode is TurbulenceMode.NOISE:
        out = arr + rng.normal(0.0, level * span, size=arr.shape)
    elif mode is TurbulenceMode.BIAS:
        out = arr + level * span
    else:  # MISSINGNESS — impute the instance median for the blanked entries
        median = float(np.median(arr))
        blanked = rng.random(arr.shape) < level
        out = np.where(blanked, median, arr)
    return np.clip(out, lo, hi).tolist()


def _perturb_categorical(
    values: list[Enum],
    mode: TurbulenceMode,
    level: float,
    enum_cls: type[Enum],
    rng: np.random.Generator,
) -> list[Enum]:
    members = list(enum_cls)
    if mode is TurbulenceMode.FLIP:
        out: list[Enum] = []
        for current in values:
            if rng.random() < level:
                others = [m for m in members if m != current]
                out.append(others[int(rng.integers(len(others)))])
            else:
                out.append(current)
        return out
    # MISSINGNESS — impute the instance mode for the blanked entries
    most_common = Counter(values).most_common(1)[0][0]
    return [most_common if rng.random() < level else current for current in values]


def apply_turbulence(
    problem: AllocationProblem,
    spec: PerturbationSpec,
    rng: np.random.Generator,
) -> AllocationProblem:
    """Return a deep copy of *problem* with *spec* applied to one input field.

    Fail closed: an unknown field, a mode that does not fit the field's kind,
    or an out-of-range probability is denied with ``ValueError`` rather than
    silently ignored — a turbulence run that quietly did nothing would be a
    false negative in the degradation study.
    """
    if not np.isfinite(spec.level):
        raise ValueError("turbulence level must be finite")

    if spec.field in _CONTINUOUS:
        owner, attr, lo, hi = _CONTINUOUS[spec.field]
        if spec.mode is TurbulenceMode.FLIP:
            raise ValueError(f"FLIP is categorical-only; {spec.field!r} is continuous")
        if spec.mode is TurbulenceMode.NOISE and spec.level < 0.0:
            raise ValueError("NOISE level (sigma fraction) must be non-negative")
        kind = "continuous"
    elif spec.field in _CATEGORICAL:
        owner, attr, enum_cls = _CATEGORICAL[spec.field]
        if spec.mode in (TurbulenceMode.NOISE, TurbulenceMode.BIAS):
            raise ValueError(f"{spec.mode.value} is continuous-only; {spec.field!r} is categorical")
        kind = "categorical"
    else:
        raise ValueError(f"unknown turbulence field: {spec.field!r}")

    if spec.mode in (TurbulenceMode.MISSINGNESS, TurbulenceMode.FLIP) and not (
        0.0 <= spec.level <= 1.0
    ):
        raise ValueError("missingness/flip level must be a probability in [0, 1]")

    perturbed = copy.deepcopy(problem)
    entities = _entities(perturbed, owner)
    values = [getattr(e, attr) for e in entities]

    if kind == "continuous":
        new_values = _perturb_continuous(values, spec.mode, spec.level, lo, hi, rng)
    else:
        new_values = _perturb_categorical(values, spec.mode, spec.level, enum_cls, rng)

    for entity, value in zip(entities, new_values):
        setattr(entity, attr, value)
    return perturbed
