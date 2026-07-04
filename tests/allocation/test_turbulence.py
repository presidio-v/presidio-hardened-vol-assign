"""Tests for the input-turbulence harness (Paper B, RQ1)."""

from __future__ import annotations

import numpy as np
import pytest

from presidio_vol_assign.allocation.models import LivingStatus
from presidio_vol_assign.allocation.turbulence import (
    PerturbationSpec,
    TurbulenceMode,
    apply_turbulence,
)


def _idl(problem) -> list[float]:
    return [p.infrastructure_damage_level for p in problem.people]


def test_noise_is_deterministic_from_the_generator(problem) -> None:
    spec = PerturbationSpec("infrastructure_damage_level", TurbulenceMode.NOISE, 0.2)
    a = apply_turbulence(problem, spec, np.random.default_rng(0))
    b = apply_turbulence(problem, spec, np.random.default_rng(0))
    assert _idl(a) == _idl(b)


def test_zero_noise_is_identity(problem) -> None:
    spec = PerturbationSpec("infrastructure_damage_level", TurbulenceMode.NOISE, 0.0)
    out = apply_turbulence(problem, spec, np.random.default_rng(0))
    assert _idl(out) == _idl(problem)


def test_noise_stays_in_range(problem) -> None:
    spec = PerturbationSpec("infrastructure_damage_level", TurbulenceMode.NOISE, 1.0)
    out = apply_turbulence(problem, spec, np.random.default_rng(1))
    assert all(0.0 <= v <= 100.0 for v in _idl(out))


def test_does_not_mutate_the_input(problem) -> None:
    before = _idl(problem)
    apply_turbulence(
        problem,
        PerturbationSpec("infrastructure_damage_level", TurbulenceMode.BIAS, 0.3),
        np.random.default_rng(0),
    )
    assert _idl(problem) == before


def test_bias_shifts_systematically(problem) -> None:
    out = apply_turbulence(
        problem,
        PerturbationSpec("infrastructure_damage_level", TurbulenceMode.BIAS, 0.2),
        np.random.default_rng(0),
    )
    assert np.mean(_idl(out)) > np.mean(_idl(problem))


def test_full_missingness_imputes_the_median(problem) -> None:
    out = apply_turbulence(
        problem,
        PerturbationSpec("infrastructure_damage_level", TurbulenceMode.MISSINGNESS, 1.0),
        np.random.default_rng(0),
    )
    median = float(np.median(_idl(problem)))
    assert all(v == median for v in _idl(out))


def test_full_flip_relabels_every_value(problem) -> None:
    out = apply_turbulence(
        problem,
        PerturbationSpec("living_status", TurbulenceMode.FLIP, 1.0),
        np.random.default_rng(0),
    )
    for original, perturbed in zip(problem.people, out.people):
        assert perturbed.living_status != original.living_status
        assert isinstance(perturbed.living_status, LivingStatus)


def test_unknown_field_is_denied(problem) -> None:
    with pytest.raises(ValueError, match="unknown turbulence field"):
        apply_turbulence(
            problem,
            PerturbationSpec("not_a_field", TurbulenceMode.NOISE, 0.1),
            np.random.default_rng(0),
        )


def test_flip_on_continuous_is_denied(problem) -> None:
    with pytest.raises(ValueError, match="categorical-only"):
        apply_turbulence(
            problem,
            PerturbationSpec("travel_duration", TurbulenceMode.FLIP, 0.1),
            np.random.default_rng(0),
        )


def test_noise_on_categorical_is_denied(problem) -> None:
    with pytest.raises(ValueError, match="continuous-only"):
        apply_turbulence(
            problem,
            PerturbationSpec("road_condition", TurbulenceMode.NOISE, 0.1),
            np.random.default_rng(0),
        )


def test_out_of_range_probability_is_denied(problem) -> None:
    with pytest.raises(ValueError, match="probability in"):
        apply_turbulence(
            problem,
            PerturbationSpec("living_status", TurbulenceMode.FLIP, 1.5),
            np.random.default_rng(0),
        )


def test_non_finite_level_is_denied(problem) -> None:
    with pytest.raises(ValueError, match="finite"):
        apply_turbulence(
            problem,
            PerturbationSpec("infrastructure_damage_level", TurbulenceMode.NOISE, float("nan")),
            np.random.default_rng(0),
        )
