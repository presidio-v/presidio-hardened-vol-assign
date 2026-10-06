"""Load-coupled CAIL: reduces to the published model when uncoupled, penalises load otherwise."""

from __future__ import annotations

import math

import numpy as np
import pytest

from presidio_vol_assign.allocation.load_coupling import (
    centre_capacities,
    overload,
    post_occupancy,
    precompute_load_coupled_cache,
)
from presidio_vol_assign.allocation.solvers import evaluate_pairs, precompute_fis_cache


def test_zero_load_equals_static_cail(problem, base_config) -> None:
    static = precompute_fis_cache(problem, base_config)
    coupled = precompute_load_coupled_cache(problem, base_config, kappa=2.0, static=static)
    np.testing.assert_allclose(coupled.cail_load[:, :, 0], static.cail, rtol=0, atol=1e-9)


def test_infinite_kappa_reproduces_separable_model(problem, base_config) -> None:
    static = precompute_fis_cache(problem, base_config)
    coupled = precompute_load_coupled_cache(problem, base_config, kappa=math.inf, static=static)
    pairs = [(0, 0), (1, 0), (2, 0), (3, 1)]
    k = base_config.objectives
    assert evaluate_pairs(pairs, coupled, k) == pytest.approx(evaluate_pairs(pairs, static, k))


def test_load_raises_cail_above_the_static_value(problem, base_config) -> None:
    static = precompute_fis_cache(problem, base_config)
    coupled = precompute_load_coupled_cache(problem, base_config, kappa=1.0, static=static)
    k = base_config.objectives
    piled = [(0, 0), (1, 0), (2, 0), (3, 0)]
    # same allocation, now scored at the occupancy it creates: CAIL must not fall
    assert evaluate_pairs(piled, coupled, k)[-1] > evaluate_pairs(piled, static, k)[-1]
    assert (coupled.cail_load[:, :, -1] >= coupled.cail_load[:, :, 0] - 1e-9).all()


def test_post_occupancy_clips_and_overload_counts(problem) -> None:
    cap = centre_capacities(problem, kappa=1.0)
    assert post_occupancy(np.array([90.0]), cap[:1], np.array([50.0]))[0] == 100.0
    assert overload([(j, 0) for j in range(problem.n_dir)], problem, kappa=1.0) > 0.0
    with pytest.raises(ValueError):
        centre_capacities(problem, kappa=0.0)
