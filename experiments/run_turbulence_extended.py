"""Turbulence on the exact decider across instances and conditions (Paper B revision).

Answers R2.6 and R3 ("Strengthen the basis for generalisation"): the exact-decider study of
``run_turbulence_exact`` is repeated

* on five further generated instances per size (generator seeds 101-105, CRC-32-salted),
  besides the reference instance;
* under three demand conditions (n_dir = people/6, /3, /2: slack, reference, scarce) and
  two occupancy conditions (as generated; stressed, every centre's occupancy raised by 25
  points, clipped at 100);
* with the original eight cells, the systematic-bias cells (both signs, the four
  continuous fields), and three joint perturbations applied together at the same level:
  all four continuous fields with noise; the transport triple (travel-duration noise,
  road-condition and hazard flips); and all eight original cells at once.

Bias is deterministic, so it has one realisation per level. Regret is the clean-quality
loss of the exact decider; trend tests in ``analyze_turbulence_extended`` use levels
0.05-0.4 only (level 0 is the identity check and is not run here).

Writes ``<out>/<size>/manifest_<instance>.csv``.

Usage::

    python -m experiments.run_turbulence_extended --size small --instance ref
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.generate_instances import SIZES, generate_instance
from experiments.run_h1_h2_h4 import BASE_SEED
from experiments.run_turbulence import OBJECTIVES, _config
from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
from presidio_vol_assign.allocation.decisions import decision_stability
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.turbulence import (
    PerturbationSpec,
    TurbulenceMode,
    apply_turbulence,
)
from presidio_vol_assign.allocation.validation import load_allocation_problem

LEVELS = (0.05, 0.1, 0.2, 0.4)
REALIZATIONS = 20
INSTANCE_SEEDS = (101, 102, 103, 104, 105)
DEMAND = {"slack": 6, "reference": 3, "scarce": 2}  # n_dir = people // divisor
OCCUPANCY_SHIFT = 0.25  # "stressed": occupancy + 25 points (bias on the 0-100 field)

SINGLE = (
    ("infrastructure_damage_level", "noise"),
    ("resource_time_remaining", "noise"),
    ("center_occupancy_rate", "noise"),
    ("travel_duration", "noise"),
    ("infrastructure_damage_level", "missingness"),
    ("center_occupancy_rate", "missingness"),
    ("road_condition", "flip"),
    ("possible_hazard", "flip"),
)
CONTINUOUS = (
    "infrastructure_damage_level",
    "resource_time_remaining",
    "center_occupancy_rate",
    "travel_duration",
)
JOINT = {
    "joint_continuous_noise": tuple((f, "noise") for f in CONTINUOUS),
    "joint_transport": (
        ("travel_duration", "noise"),
        ("road_condition", "flip"),
        ("possible_hazard", "flip"),
    ),
    "joint_all": SINGLE,
}

HEADER = ["instance", "demand", "occupancy", "cell", "level", "realization", "regret", "churn"]


def cells() -> list[tuple[str, tuple[tuple[str, str, float], ...], int]]:
    """(name, ((field, mode, sign), ...), realisations) for every cell."""
    out = [(f"{f}_{m}", ((f, m, 1.0),), REALIZATIONS) for f, m in SINGLE]
    for f in CONTINUOUS:
        out.append((f"{f}_bias_up", ((f, "bias", 1.0),), 1))
        out.append((f"{f}_bias_down", ((f, "bias", -1.0),), 1))
    for name, parts in JOINT.items():
        out.append((name, tuple((f, m, 1.0) for f, m in parts), REALIZATIONS))
    return out


def _load(size: str, instance: str, root: Path):
    if instance == "ref":
        base = Path("experiments/instances") / size
    else:
        spec = SIZES[size]
        base = root / f"seed{instance}" / spec.name
        if not (base / "travel.csv").exists():
            generate_instance(spec, seed=int(instance), out_dir=root / f"seed{instance}")
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="small", choices=("small", "large"))
    parser.add_argument("--instance", default="ref", help="'ref' or a generator seed")
    parser.add_argument("--out", type=Path, default=Path("experiments/results/turbulence_extended"))
    args = parser.parse_args()
    base = _load(args.size, args.instance, args.out / "instances")
    out = args.out / args.size
    out.mkdir(parents=True, exist_ok=True)
    cfg = _config(100, 150)

    with (out / f"manifest_{args.instance}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for demand, divisor in DEMAND.items():
            people = sum(1 for _ in open(base / "people.csv")) - 1
            problem = load_allocation_problem(
                base / "people.csv",
                base / "centers.csv",
                base / "travel.csv",
                n_dir=people // divisor,
            )
            for occupancy in ("generated", "stressed"):
                cond = problem
                if occupancy == "stressed":
                    spec = PerturbationSpec(
                        "center_occupancy_rate", TurbulenceMode.BIAS, OCCUPANCY_SHIFT
                    )
                    cond = apply_turbulence(problem, spec, np.random.default_rng(0))
                clean_cache = precompute_fis_cache_fast(cond, cfg)
                clean = exact_weighted_sum_pairs(clean_cache, cond.n_dir, OBJECTIVES)
                for name, parts, n_real in cells():
                    for level in LEVELS:
                        for r in range(n_real):
                            rng = np.random.default_rng(BASE_SEED + r)
                            pert = cond
                            for field, mode, sign in parts:
                                spec = PerturbationSpec(field, TurbulenceMode(mode), sign * level)
                                pert = apply_turbulence(pert, spec, rng)
                            cache = precompute_fis_cache_fast(pert, cfg)
                            decided = exact_weighted_sum_pairs(cache, cond.n_dir, OBJECTIVES)
                            m = decision_stability(
                                clean, decided, clean_cache, OBJECTIVES, cond.n_centers
                            )
                            writer.writerow(
                                [
                                    args.instance,
                                    demand,
                                    occupancy,
                                    name,
                                    level,
                                    r,
                                    m["quality_loss"],
                                    m["allocation_churn"],
                                ]
                            )
                fh.flush()
                print(f"  {args.size} {args.instance} {demand} {occupancy}: done", flush=True)


if __name__ == "__main__":
    main()
