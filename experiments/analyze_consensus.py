"""Consensus-of-k guardrail (Paper B, RQ2 appendix), computed offline from stored decisions.

Each ensemble directs people by plurality vote over k committed decisions (top n_dir by
vote count, centre by plurality; ties to the lower index). Disjoint ensembles from the
20-seed clean floor are compared pairwise. Writes ``experiments/results/audit/consensus.csv``.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.generate_instances import SIZES
from experiments.run_turbulence import OBJECTIVES, _config
from presidio_vol_assign.allocation.decisions import decision_stability
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.solvers import evaluate_pairs
from presidio_vol_assign.allocation.validation import load_allocation_problem


def plurality(decisions: list[list[tuple[int, int]]], n_dir: int) -> list[tuple[int, int]]:
    votes: Counter = Counter()
    centre: dict[int, Counter] = defaultdict(Counter)
    for decision in decisions:
        for person, c in decision:
            votes[person] += 1
            centre[person][c] += 1
    people = sorted(votes, key=lambda j: (-votes[j], j))[:n_dir]
    return [(j, min(centre[j], key=lambda c: (-centre[j][c], c))) for j in people]


def main() -> None:
    out = Path("experiments/results/audit")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in ("small", "large"):
        base = Path("experiments/instances") / size
        problem = load_allocation_problem(
            base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=SIZES[size].n_dir
        )
        cache = precompute_fis_cache_fast(problem, _config(100, 150))
        path = Path(f"experiments/results/seed_floor/{size}_gen150/equal-weight")
        decisions = [
            [tuple(p) for p in json.loads(r["pairs"])]
            for r in csv.DictReader((path / "seed_floor_decisions.csv").open())
        ]
        for k in (1, 3, 5, 10):
            groups = [decisions[g * k : (g + 1) * k] for g in range(len(decisions) // k)]
            ensembles = [plurality(g, problem.n_dir) for g in groups]
            quality = [float(np.sum(evaluate_pairs(e, cache, OBJECTIVES))) for e in ensembles]
            churn = [
                decision_stability(a, b, cache, OBJECTIVES, problem.n_centers)["allocation_churn"]
                for a, b in itertools.combinations(ensembles, 2)
            ]
            row = {
                "size": size,
                "k": k,
                "ensembles": len(ensembles),
                "churn_mean": float(np.mean(churn)),
                "churn_min": float(np.min(churn)),
                "churn_max": float(np.max(churn)),
                "quality_mean": float(np.mean(quality)),
            }
            rows.append(row)
            print(row, flush=True)
    with (out / "consensus.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
