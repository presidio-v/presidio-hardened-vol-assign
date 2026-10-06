"""Unit tests for the weight-LHS sensitivity sweeper."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from presidio_vol_assign.allocation.models import AllocationSolverType, Weights
from presidio_vol_assign.allocation.sensitivity import (
    WeightSample,
    lhs_weight_samples,
    run_weight_sweep,
)


class TestLHSWeightSamples:
    def test_sample_count(self):
        samples = lhs_weight_samples(Weights(), n_samples=10, bound=0.2, seed=42)
        assert len(samples) == 10
        assert all(isinstance(s, WeightSample) for s in samples)

    def test_samples_within_bound(self):
        samples = lhs_weight_samples(Weights(), n_samples=20, bound=0.2, seed=42)
        for s in samples:
            for name in ("was", "wds", "wil", "wls", "wrc", "wph"):
                v = getattr(s.weights, name)
                # baseline=1, bound=0.2 → [0.8, 1.0] (clipped at 1.0)
                assert 0.8 <= v <= 1.0

    def test_seed_determinism(self):
        a = lhs_weight_samples(Weights(), n_samples=5, bound=0.1, seed=99)
        b = lhs_weight_samples(Weights(), n_samples=5, bound=0.1, seed=99)
        # Same seed → same draws
        for sa, sb in zip(a, b):
            assert sa.weights == sb.weights

    def test_zero_samples_rejected(self):
        with pytest.raises(ValueError, match="n_samples"):
            lhs_weight_samples(Weights(), n_samples=0)

    @pytest.mark.parametrize("bad", [-0.1, 0.0, 1.5])
    def test_bad_bound_rejected(self, bad):
        with pytest.raises(ValueError, match="bound"):
            lhs_weight_samples(Weights(), n_samples=5, bound=bad)


class TestRunWeightSweep:
    def test_manifest_written_with_one_row_per_sample(self, problem, base_config, tmp_path):
        # Tiny sweep: 3 LHS samples with cheap solver
        cfg = dataclasses.replace(
            base_config,
            solver=AllocationSolverType.NSGA2,
            objectives=4,
            pop_size=15,
            generations=5,
        )
        samples = lhs_weight_samples(Weights(), n_samples=3, bound=0.1, seed=7)
        manifest = run_weight_sweep(problem, cfg, samples, tmp_path)
        assert manifest.exists()
        df = pd.read_csv(manifest)
        assert len(df) == 3
        assert {"sample_id", "was", "wph", "hv", "mn_ulpp_min", "mn_trd_min"} <= set(df.columns)
        # Per-sample subdirs created
        for sid in range(3):
            assert (tmp_path / f"sample_{sid:04d}").exists()

    def test_3obj_sweep_writes_til_columns(self, problem, base_config, tmp_path):
        cfg = dataclasses.replace(
            base_config,
            solver=AllocationSolverType.NSGA2,
            objectives=3,
            pop_size=15,
            generations=5,
        )
        samples = lhs_weight_samples(Weights(), n_samples=2, bound=0.1, seed=7)
        manifest = run_weight_sweep(problem, cfg, samples, tmp_path)
        df = pd.read_csv(manifest)
        assert "mn_til_min" in df.columns
        assert "mn_trd_min" not in df.columns
        assert "mn_rpd_min" not in df.columns
