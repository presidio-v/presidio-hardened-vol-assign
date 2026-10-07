"""Seed-only null for RQ1 (Paper B): how much does the committed decision move with no turbulence?

The turbulence driver compares a decision made on degraded inputs against the clean
decision for the same solver seed. Any change to the FIS cache decorrelates the GA
trajectory, so that comparison cannot separate input sensitivity from solver
stochasticity. This driver measures the second on its own: it solves the *clean*
instance under N seeds (the first seeds coincide with the turbulence driver's reps)
and writes, under ``<out>/``:

* ``fronts.jsonl`` — one record per seed: seed, front signature, solve latency, and every
  front solution's fitness and (person, centre) index pairs. The decision rules and the
  seed-ensemble guardrail are computed offline from these, so nothing here is re-solved;
* ``meta.json`` — instance hash, solver budget, and environment fingerprint;
* ``<rule>/seed_floor_decisions.csv`` — per seed: realised clean objectives, their sum,
  and the committed pairs (JSON);
* ``<rule>/seed_floor_pairwise.csv`` — the RQ1 stability metrics plus directed-set Jaccard
  for every unordered seed pair: the floor that any turbulence effect must clear.

Usage::

    python -m experiments.run_seed_floor --size small --seeds 20
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from experiments.generate_instances import SIZES
from experiments.run_h1_h2_h4 import BASE_SEED, SEED_STEP
from experiments.run_reproducibility import _env_fingerprint
from experiments.run_turbulence import _DECISION_RULES, OBJECTIVES, _config
from presidio_vol_assign.allocation.decisions import decision_stability, pairs_of
from presidio_vol_assign.allocation.models import AllocationConfig, AllocationSolverType
from presidio_vol_assign.allocation.repro import allocation_front_signature
from presidio_vol_assign.allocation.solvers import evaluate_pairs, precompute_fis_cache, solve
from presidio_vol_assign.allocation.validation import load_allocation_problem

_FIELDS = ["objective_drift", "quality_loss", "allocation_churn", "load_rank_stability"]
_INSTANCE_FILES = ("people.csv", "centers.csv", "travel.csv")


def _instance_hash(base: Path) -> str:
    digest = hashlib.sha256()
    for name in _INSTANCE_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((base / name).read_bytes())
    return digest.hexdigest()


def directed_jaccard(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    """Jaccard overlap of the two decisions' directed-person sets (ignores centres)."""
    sa = {person for person, _ in a}
    sb = {person for person, _ in b}
    union = sa | sb
    if not union:
        raise ValueError("cannot compare two empty decisions")
    return len(sa & sb) / len(union)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="small", choices=sorted(SIZES.keys()))
    parser.add_argument("--seeds", type=int, default=20, help="clean solves (seed index 0..N-1)")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--solver", default="nsga2", choices=["nsga2", "nrga", "nsga3"])
    parser.add_argument(
        "--instances", type=Path, default=None, help="instance dir (default: the reference one)"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2 to form a pair")

    tag = f"{args.size}_gen{args.generations}"
    if (args.solver, args.pop_size) != ("nsga2", 100):
        tag += f"_{args.solver}_pop{args.pop_size}"
    if args.instances is not None:
        tag += f"_{args.instances.name}"
    out = args.out or Path("experiments/results/seed_floor") / tag
    out.mkdir(parents=True, exist_ok=True)

    base = args.instances or Path("experiments/instances") / args.size
    problem = load_allocation_problem(
        base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=SIZES[args.size].n_dir
    )

    def config(seed: int | None = None) -> AllocationConfig:
        return dataclasses.replace(
            _config(args.pop_size, args.generations, seed=seed),
            solver=AllocationSolverType(args.solver),
        )

    cache = precompute_fis_cache(problem, config())
    meta = {
        "solver": args.solver,
        "instances": str(base),
        "size": args.size,
        "seeds": args.seeds,
        "pop_size": args.pop_size,
        "generations": args.generations,
        "objectives": OBJECTIVES,
        "instance_sha256": _instance_hash(base),
        "env": _env_fingerprint(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(
        f"RQ1 seed floor: {args.size} {args.seeds} seeds, {args.solver} pop={args.pop_size} "
        f"gen={args.generations} instances={base}",
        flush=True,
    )
    fronts = {}
    with (out / "fronts.jsonl").open("w") as fh:
        for k in range(args.seeds):
            seed = BASE_SEED + k * SEED_STEP
            start = time.perf_counter()
            front = solve(problem, config(seed), cache=cache)
            latency = time.perf_counter() - start
            fronts[k] = front
            record = {
                "seed_index": k,
                "seed": seed,
                "signature": allocation_front_signature(front),
                "latency_sec": latency,
                "solutions": [
                    {"fitness": list(s.fitness), "pairs": pairs_of(s, problem)}
                    for s in front.solutions
                ],
            }
            fh.write(json.dumps(record) + "\n")
            print(f"  seed {k}: |front|={len(front.solutions)} {latency:.1f}s", flush=True)

    for rule, decide in sorted(_DECISION_RULES.items()):
        rule_dir = out / rule
        rule_dir.mkdir(exist_ok=True)
        decisions = {k: pairs_of(decide(front), problem) for k, front in fronts.items()}
        with (rule_dir / "seed_floor_decisions.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["seed_index", "seed", "f1", "f2", "f3", "quality", "pairs"])
            for k, pairs in decisions.items():
                obj = np.asarray(evaluate_pairs(pairs, cache, OBJECTIVES), dtype=float)
                seed = BASE_SEED + k * SEED_STEP
                writer.writerow([k, seed, *obj.tolist(), float(obj.sum()), json.dumps(pairs)])
        with (rule_dir / "seed_floor_pairwise.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["seed_a", "seed_b", *_FIELDS, "directed_jaccard"])
            for a, b in itertools.combinations(sorted(decisions), 2):
                m = decision_stability(
                    decisions[a], decisions[b], cache, OBJECTIVES, problem.n_centers
                )
                jac = directed_jaccard(decisions[a], decisions[b])
                writer.writerow([a, b, *[m[f] for f in _FIELDS], jac])


if __name__ == "__main__":
    main()
