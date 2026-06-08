"""Sensitivity-analysis drivers for the allocation model.

Implements the weight-perturbation sweep needed for H3b (the parametric
sensitivity hypothesis) of the MDPI extended paper:

    For each Latin-Hypercube sample of the six VS/RWS weights drawn from
    baseline·(1±bound), run the configured solver once and record its
    Pareto-front summary. The post-hoc analysis then computes the
    coefficient of variation of mean objectives across samples; H3b is
    confirmed when CV ≤ 10% for every objective.

The driver is deliberately solver-agnostic: it accepts an
`AllocationConfig` and varies only the `weights` field per sample. Other
hyper-parameters (solver, objectives, pop_size, generations, seed) stay
fixed across the sweep, so each sample's variation reflects only the
weight perturbation, not GA stochasticity.

Public API:
    lhs_weight_samples(baseline, n_samples, bound, seed) -> list[Weights]
    run_weight_sweep(problem, config, samples, output_dir) -> Path

Outputs: one set of `pareto_alloc_*.csv` / `metrics_alloc_*.json` per
sample plus a top-level `weight_sweep_manifest.csv` recording the six
weights, the run identifier, the four-or-three mean objectives, and HV
for every sample.
"""

from __future__ import annotations

import csv
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from presidio_vol_assign.allocation.metrics import compute_allocation_metrics
from presidio_vol_assign.allocation.models import (
    AllocationConfig,
    AllocationProblem,
    Weights,
)
from presidio_vol_assign.allocation.solvers import solve
from presidio_vol_assign.allocation.writers import (
    write_allocation_metrics_json,
    write_allocation_pareto_csv,
)

# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------


_WEIGHT_FIELDS: tuple[str, ...] = ("was", "wds", "wil", "wls", "wrc", "wph")


@dataclass(frozen=True)
class WeightSample:
    """One LHS sample point: a sample id paired with the Weights instance."""

    sample_id: int
    weights: Weights


def lhs_weight_samples(
    baseline: Weights,
    n_samples: int,
    bound: float = 0.2,
    seed: int | None = None,
) -> list[WeightSample]:
    """Draw `n_samples` Latin-Hypercube samples in 6D weight space.

    Each weight is scaled to the interval `baseline_value · (1 ± bound)`,
    clipped to [0, 1] (the domain of `Weights`). LHS gives uniform
    coverage along each axis without the clustering of pure random
    sampling, making the per-axis CV estimate more efficient.

    Args:
        baseline: Centre of the perturbation box (default Weights() = all 1.0).
        n_samples: Number of samples to draw.
        bound: Fractional perturbation per axis (default 0.2 → ±20%).
        seed: Optional seed for the LHS RNG; None = nondeterministic.

    Returns:
        A list of `WeightSample` objects in sample order.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if not (0.0 < bound <= 1.0):
        raise ValueError(f"bound must be in (0, 1], got {bound}")

    sampler = qmc.LatinHypercube(d=len(_WEIGHT_FIELDS), seed=seed)
    unit = sampler.random(n=n_samples)  # shape (n_samples, 6) ∈ [0, 1)

    base_values = np.array([getattr(baseline, name) for name in _WEIGHT_FIELDS], dtype=float)
    lo = np.clip(base_values * (1.0 - bound), 0.0, 1.0)
    hi = np.clip(base_values * (1.0 + bound), 0.0, 1.0)
    scaled = lo + unit * (hi - lo)

    samples: list[WeightSample] = []
    for sid, row in enumerate(scaled):
        kwargs = {name: float(row[i]) for i, name in enumerate(_WEIGHT_FIELDS)}
        samples.append(WeightSample(sample_id=sid, weights=Weights(**kwargs)))
    return samples


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def run_weight_sweep(
    problem: AllocationProblem,
    config: AllocationConfig,
    samples: list[WeightSample],
    output_dir: Path,
) -> Path:
    """Run the configured solver once per sample; write per-sample outputs.

    Each sample reuses every field of `config` except `weights`. Per-sample
    artefacts are written to `output_dir/sample_<id>/`; the top-level
    `weight_sweep_manifest.csv` aggregates the per-sample weights and
    Pareto-front summary statistics.

    Returns the Path to the manifest CSV.

    Manifest schema:
        sample_id, was, wds, wil, wls, wrc, wph,
        solver, objectives, nns, mid, sm, hv, cpu_time_sec,
        mn_ulpp_min, mn_ulpp_mean,
        mn_<obj2>_min, mn_<obj2>_mean,   # obj2 = til (3-obj) or trd (4-obj)
        mn_rpd_min,  mn_rpd_mean,        # 4-obj only (left blank for 3-obj)
        mn_cail_min, mn_cail_mean
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "weight_sweep_manifest.csv"

    if config.objectives == 4:
        obj_min_cols = (
            "mn_ulpp_min",
            "mn_ulpp_mean",
            "mn_trd_min",
            "mn_trd_mean",
            "mn_rpd_min",
            "mn_rpd_mean",
            "mn_cail_min",
            "mn_cail_mean",
        )
    else:
        obj_min_cols = (
            "mn_ulpp_min",
            "mn_ulpp_mean",
            "mn_til_min",
            "mn_til_mean",
            "mn_cail_min",
            "mn_cail_mean",
        )

    header = (
        "sample_id",
        *_WEIGHT_FIELDS,
        "solver",
        "objectives",
        "nns",
        "mid",
        "sm",
        "hv",
        "cpu_time_sec",
        "wall_time_sec",
        *obj_min_cols,
    )

    with manifest_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for sample in samples:
            sample_dir = output_dir / f"sample_{sample.sample_id:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sweep_config = dataclasses.replace(
                config,
                weights=sample.weights,
                output_dir=str(sample_dir),
            )

            t0 = time.monotonic()
            front = solve(problem, sweep_config)
            wall = time.monotonic() - t0

            write_allocation_pareto_csv(front, sample_dir)
            metrics = compute_allocation_metrics(front)
            write_allocation_metrics_json(metrics, sample_dir)

            row = [
                sample.sample_id,
                *(round(getattr(sample.weights, name), 6) for name in _WEIGHT_FIELDS),
                front.solver.value,
                front.objectives_count,
                metrics.nns,
                round(metrics.mid, 6),
                round(metrics.sm, 6),
                round(metrics.hv, 6),
                round(metrics.cpu_time_sec, 3),
                round(wall, 3),
                *_objective_summary(front),
            ]
            writer.writerow(row)
            fh.flush()

    return manifest_path


def _objective_summary(front) -> list:
    """Return min/mean per objective from the front, in the canonical column order."""
    if not front.solutions:
        if front.objectives_count == 4:
            return [""] * 8
        return [""] * 6

    if front.objectives_count == 4:
        ulpp = [s.mn_ulpp for s in front.solutions]
        trd = [s.mn_trd for s in front.solutions]
        rpd = [s.mn_rpd for s in front.solutions]
        cail = [s.mn_cail for s in front.solutions]
        return [
            round(min(ulpp), 6),
            round(sum(ulpp) / len(ulpp), 6),
            round(min(trd), 6),
            round(sum(trd) / len(trd), 6),
            round(min(rpd), 6),
            round(sum(rpd) / len(rpd), 6),
            round(min(cail), 6),
            round(sum(cail) / len(cail), 6),
        ]
    ulpp = [s.mn_ulpp for s in front.solutions]
    til = [s.mn_til for s in front.solutions]
    cail = [s.mn_cail for s in front.solutions]
    return [
        round(min(ulpp), 6),
        round(sum(ulpp) / len(ulpp), 6),
        round(min(til), 6),
        round(sum(til) / len(til), 6),
        round(min(cail), 6),
        round(sum(cail) / len(cail), 6),
    ]
