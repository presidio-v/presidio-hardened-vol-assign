"""RQ1/RQ2 audit summary (Paper B): necessity and identifiability for every seed-floor run.

For each ``experiments/results/seed_floor/<run>/`` this computes, on the run's own instance:

* the exact equal-weight optimum (separable greedy) and the exact *supported* front from a
  simplex weight sweep (21 steps per axis, 231 weight vectors);
* per seed: how many front points the single exact optimum dominates, the front's
  hypervolume as a share of the exact supported front's, and the best front point's sum;
* the seed floor of the committed (equal-weight) decision: pairwise churn and directed-set
  Jaccard, and the clean quality mean / sd.

Hypervolume uses a reference point at 1.1x the component-wise maximum over the exact front
and every front of the same instance, so shares are comparable within an instance.
Writes ``experiments/results/audit/audit_summary.csv``.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV

from experiments.generate_instances import SIZES
from experiments.run_turbulence import OBJECTIVES, _config
from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.solvers import evaluate_pairs
from presidio_vol_assign.allocation.validation import load_allocation_problem

FLOOR = Path("experiments/results/seed_floor")
OUT = Path("experiments/results/audit")
_STEPS = 20


def nondominated(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(points, 9), axis=0)
    keep = [
        i for i, p in enumerate(pts) if not ((pts <= p).all(axis=1) & (pts < p).any(axis=1)).any()
    ]
    return pts[keep]


def exact_supported_front(cache, n_dir: int) -> np.ndarray:  # noqa: ANN001
    pts = []
    for a in range(_STEPS + 1):
        for b in range(_STEPS + 1 - a):
            w = (a / _STEPS, b / _STEPS, (_STEPS - a - b) / _STEPS)
            pairs = exact_weighted_sum_pairs(cache, n_dir, OBJECTIVES, w)
            pts.append(evaluate_pairs(pairs, cache, OBJECTIVES))
    return nondominated(np.array(pts))


def _instance_label(base: Path) -> str:
    """'reference' for the published instances, else the alternate-instance seed folder."""
    return "reference" if base.parent == Path("experiments/instances") else base.parent.name


def _pairwise(rows: list[dict], metric: str) -> float:
    return float(np.mean([float(r[metric]) for r in rows]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # a run is complete once its pairwise table is written (it is the driver's last output)
    runs = sorted(
        p for p in FLOOR.iterdir() if (p / "equal-weight" / "seed_floor_pairwise.csv").exists()
    )
    by_instance: dict[str, list[Path]] = defaultdict(list)
    metas = {}
    for run in runs:
        meta = json.loads((run / "meta.json").read_text())
        metas[run] = meta
        by_instance[meta["instance_sha256"]].append(run)

    rows_out = []
    for sha, inst_runs in by_instance.items():
        meta0 = metas[inst_runs[0]]
        base = Path(meta0.get("instances") or f"experiments/instances/{meta0['size']}")
        problem = load_allocation_problem(
            base / "people.csv",
            base / "centers.csv",
            base / "travel.csv",
            n_dir=SIZES[meta0["size"]].n_dir,
        )
        cache = precompute_fis_cache_fast(problem, _config(100, 150))
        exact_pt = np.array(
            evaluate_pairs(exact_weighted_sum_pairs(cache, problem.n_dir, OBJECTIVES), cache, 3)
        )
        exact_front = exact_supported_front(cache, problem.n_dir)
        fronts = {
            run: [
                np.array([s["fitness"] for s in json.loads(line)["solutions"]])
                for line in (run / "fronts.jsonl").open()
            ]
            for run in inst_runs
        }
        ref = np.vstack([exact_front, *[f for fs in fronts.values() for f in fs]]).max(axis=0) * 1.1
        hv = HV(ref_point=ref)
        hv_exact = float(hv(exact_front))
        for run in inst_runs:
            meta = metas[run]
            fs = fronts[run]
            dominated = [
                int(((exact_pt <= f + 1e-9).all(axis=1) & (exact_pt < f - 1e-9).any(axis=1)).sum())
                for f in fs
            ]
            shares = [float(hv(nondominated(f))) / hv_exact for f in fs]
            pairwise = list(
                csv.DictReader((run / "equal-weight" / "seed_floor_pairwise.csv").open())
            )
            quality = [
                float(r["quality"])
                for r in csv.DictReader((run / "equal-weight" / "seed_floor_decisions.csv").open())
            ]
            rows_out.append(
                {
                    "run": run.name,
                    "size": meta["size"],
                    "instance": _instance_label(base),
                    "solver": meta.get("solver", "nsga2"),
                    "pop_size": meta["pop_size"],
                    "generations": meta["generations"],
                    "seeds": meta["seeds"],
                    "exact_sum": float(exact_pt.sum()),
                    "exact_front_points": len(exact_front),
                    "dominated_points": sum(dominated),
                    "front_points": sum(len(f) for f in fs),
                    "hv_share_mean": float(np.mean(shares)),
                    "hv_share_min": float(np.min(shares)),
                    "hv_share_max": float(np.max(shares)),
                    "best_front_sum": float(min(f.sum(axis=1).min() for f in fs)),
                    "committed_q_mean": float(np.mean(quality)),
                    "committed_q_sd": float(np.std(quality, ddof=1)),
                    "churn_floor": _pairwise(pairwise, "allocation_churn"),
                    "jaccard_floor": _pairwise(pairwise, "directed_jaccard"),
                    "mean_latency_s": float(
                        np.mean(
                            [
                                json.loads(line)["latency_sec"]
                                for line in (run / "fronts.jsonl").open()
                            ]
                        )
                    ),
                }
            )
            print(
                f"{run.name:38s} exact {exact_pt.sum():6.2f} | dominated "
                f"{sum(dominated):4d}/{sum(len(f) for f in fs):4d} | HV share "
                f"{np.mean(shares):.2f} | committed Q {np.mean(quality):6.2f}±"
                f"{np.std(quality, ddof=1):.2f} | churn {rows_out[-1]['churn_floor']:.3f} "
                f"jac {rows_out[-1]['jaccard_floor']:.2f}",
                flush=True,
            )

    with (OUT / "audit_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(sorted(rows_out, key=lambda r: (r["size"], r["run"])))


if __name__ == "__main__":
    main()
