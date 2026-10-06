"""The compiled FIS3 must reproduce scikit-fuzzy to floating-point precision."""

from __future__ import annotations

import numpy as np
import pytest

from presidio_vol_assign.allocation.fast_fis import compiled_fis3, evaluate_fis3_cail_fast
from presidio_vol_assign.allocation.fis import evaluate_fis3_cail, fis_overrides


def _sample() -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(7)
    random_pts = [
        (float(c), float(r), float(t))
        for c, r, t in zip(
            rng.uniform(-5, 105, 400), rng.uniform(-5, 105, 400), rng.uniform(-5, 185, 400)
        )
    ]
    # MF breakpoints and universe edges, where the upsampling and clipping branches bite
    knots = [0.0, 25.0, 37.5, 50.0, 62.5, 75.0, 100.0]
    td_knots = [0.0, 30.0, 45.0, 60.0, 90.0, 180.0]
    grid = [(c, r, t) for c in knots for r in knots[::2] for t in td_knots]
    return random_pts + grid


@pytest.mark.parametrize("point", _sample())
def test_fast_fis3_matches_skfuzzy(point) -> None:
    assert evaluate_fis3_cail_fast(*point) == pytest.approx(evaluate_fis3_cail(*point), abs=1e-9)


def test_fast_fis3_rejects_rule_overrides() -> None:
    with fis_overrides({"fis3": [0]}), pytest.raises(RuntimeError):
        evaluate_fis3_cail_fast(50.0, 50.0, 60.0)


def test_compiled_fis3_checks_arity() -> None:
    with pytest.raises(ValueError):
        compiled_fis3()(50.0, 50.0)


def test_batch_matches_scalar_path() -> None:
    fis3 = compiled_fis3()
    pts = _sample()
    cor, rdr, td = (np.array(col) for col in zip(*pts, strict=True))
    batch = fis3.batch(cor, rdr, td, chunk=97)  # odd chunk size exercises the chunk seams
    scalar = np.array([fis3(*p) for p in pts])
    np.testing.assert_allclose(batch, scalar, rtol=0, atol=1e-9)


def test_batch_rejects_ragged_inputs() -> None:
    with pytest.raises(ValueError):
        compiled_fis3().batch(np.zeros(3), np.zeros(3), np.zeros(2))


@pytest.mark.parametrize("objectives", [3, 4])
def test_fast_cache_matches_skfuzzy_cache(problem, base_config, objectives) -> None:
    import dataclasses

    from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
    from presidio_vol_assign.allocation.solvers import precompute_fis_cache

    config = dataclasses.replace(base_config, objectives=objectives)
    slow = precompute_fis_cache(problem, config)
    fast = precompute_fis_cache_fast(problem, config)
    for field in ("ulpp", "til", "trd", "rpd", "cail"):
        np.testing.assert_allclose(getattr(fast, field), getattr(slow, field), rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", ["fis1", "fis2_til", "fis2a_trd", "fis2b_rpd"])
def test_other_compiled_fis_match_skfuzzy(name) -> None:
    from presidio_vol_assign.allocation import fis
    from presidio_vol_assign.allocation.fast_fis import compiled

    rng = np.random.default_rng(11)
    wrappers = {
        "fis1": (fis.evaluate_fis1_ulpp, [(0, 1), (0, 100), (0, 48)]),
        "fis2_til": (fis.evaluate_fis2_til, [(0, 180), (0, 1)]),
        "fis2a_trd": (fis.evaluate_fis2a_trd, [(0, 1), (0, 1)]),
        "fis2b_rpd": (fis.evaluate_fis2b_rpd, [(0, 180)]),
    }
    slow, ranges = wrappers[name]
    pts = np.column_stack([rng.uniform(lo, hi, 300) for lo, hi in ranges])
    want = np.array([slow(*p) for p in pts])
    np.testing.assert_allclose(compiled(name).batch(*pts.T), want, rtol=0, atol=1e-9)
