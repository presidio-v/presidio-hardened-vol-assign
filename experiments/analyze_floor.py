"""Floor-adjusted RQ1 analysis (Paper B): turbulence effects measured above the seed-only null.

The original RQ1 analysis compared each turbulence-driven decision with the clean
decision for the same solver seed. A clean re-solve under a *different* seed already
moves the decision about as much (``run_seed_floor.py``), so churn and drift must be
read against that floor. This script:

1. summarises the floor per size: pairwise churn, directed-set Jaccard and drift over
   all seed pairs (CI by cluster bootstrap over seeds), plus the clean quality spread;
2. checks bit-for-bit that the floor run reproduces the turbulence run's clean
   decisions (one manifest row is recomputed from scratch and must match exactly);
3. per (cell, level), reports fuzzy churn minus the floor mean with a 95% CI;
4. per cell, tests the realised-quality harm: ``quality_loss`` is already paired
   (decision on degraded inputs minus clean decision, same seed, scored on clean truth),
   so the null is zero; trend over levels 0.05–0.4 by one-sided Spearman (realisation =
   unit, reps averaged), Holm across cells within a size; effect size is the mean loss in
   units of the clean-seed quality s.d.;
5. places the crisp baseline's clean quality inside the fuzzy clean-seed distribution.

Usage::

    python -m experiments.analyze_floor --size small
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.generate_instances import SIZES
from experiments.run_h1_h2_h4 import BASE_SEED
from experiments.run_turbulence import OBJECTIVES, _config
from presidio_vol_assign.allocation.baselines import crisp_greedy_pairs
from presidio_vol_assign.allocation.decisions import (
    canonical_decision,
    decision_stability,
    pairs_of,
)
from presidio_vol_assign.allocation.solvers import evaluate_pairs, precompute_fis_cache, solve
from presidio_vol_assign.allocation.turbulence import (
    PerturbationSpec,
    TurbulenceMode,
    apply_turbulence,
)
from presidio_vol_assign.allocation.validation import load_allocation_problem

_RNG = np.random.default_rng(20261005)
_BOOT = 4000
_FLOOR_LEVELS = (0.05, 0.1, 0.2, 0.4)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _seed_cluster_boot(pairwise: list[dict], metric: str, n_seeds: int) -> tuple[float, float]:
    """95% CI of the mean pairwise metric, resampling *seeds* (pairs share seeds)."""
    table = np.full((n_seeds, n_seeds), np.nan)
    for r in pairwise:
        a, b, v = int(r["seed_a"]), int(r["seed_b"]), float(r[metric])
        table[a, b] = table[b, a] = v
    means = []
    for _ in range(_BOOT):
        idx = _RNG.integers(0, n_seeds, n_seeds)
        sub = table[np.ix_(idx, idx)]
        # drop self-pairs, including those created by resampling one seed twice
        mask = idx[:, None] != idx[None, :]
        vals = sub[mask]
        vals = vals[~np.isnan(vals)]
        if vals.size:
            means.append(vals.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _boot_mean_ci(values: np.ndarray) -> tuple[float, float]:
    draws = _RNG.choice(values, size=(_BOOT, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adjusted = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def _load_problem(size: str):
    base = Path("experiments/instances") / size
    return load_allocation_problem(
        base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=SIZES[size].n_dir
    )


def _canonical_from_record(record: dict) -> list[tuple[int, int]]:
    """Equal-weight rule applied to a stored front (mirrors decisions.canonical_decision)."""
    fit = np.array([s["fitness"] for s in record["solutions"]], dtype=float)
    lo, hi = fit.min(axis=0), fit.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    best = int(np.argmin(((fit - lo) / span).sum(axis=1)))
    return [tuple(p) for p in record["solutions"][best]["pairs"]]


def _consistency_check(size: str, floor_dir: Path, turb_dir: Path, gen: int) -> dict:
    """Recompute one manifest row (IDL noise, level 0.05, realisation 0, rep 0) from scratch."""
    problem = _load_problem(size)
    cfg = _config(100, gen)
    clean_cache = precompute_fis_cache(problem, cfg)
    with (floor_dir / "fronts.jsonl").open() as fh:
        rec0 = json.loads(fh.readline())
    clean_pairs = _canonical_from_record(rec0)
    spec = PerturbationSpec("infrastructure_damage_level", TurbulenceMode("noise"), 0.05)
    perturbed = apply_turbulence(problem, spec, np.random.default_rng(BASE_SEED + 0))
    pert_cache = precompute_fis_cache(perturbed, cfg)
    pert_pairs = pairs_of(
        canonical_decision(solve(perturbed, _config(100, gen, seed=BASE_SEED), cache=pert_cache)),
        perturbed,
    )
    got = decision_stability(clean_pairs, pert_pairs, clean_cache, OBJECTIVES, problem.n_centers)
    rows = _read_csv(turb_dir / "infrastructure_damage_level_noise" / "turbulence_manifest.csv")
    want = next(
        r
        for r in rows
        if r["system"] == "fuzzy"
        and r["rep"] == "0"
        and r["realization"] == "0"
        and float(r["level"]) == 0.05
    )
    match = all(
        np.isclose(got[k], float(want[k]), rtol=0, atol=1e-12, equal_nan=True)
        for k in ("objective_drift", "quality_loss", "allocation_churn")
    )
    return {"match": bool(match), "recomputed": got, "manifest": {k: want[k] for k in got}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="small", choices=sorted(SIZES.keys()))
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--rule", default="equal-weight")
    parser.add_argument("--turbulence", type=Path, default=None)
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    floor_dir = Path("experiments/results/seed_floor") / f"{args.size}_gen{args.generations}"
    turb_dir = args.turbulence or Path("experiments/results/turbulence") / args.size
    out_dir = Path("experiments/results/floor_analysis") / args.size
    out_dir.mkdir(parents=True, exist_ok=True)

    pairwise = _read_csv(floor_dir / args.rule / "seed_floor_pairwise.csv")
    decisions = _read_csv(floor_dir / args.rule / "seed_floor_decisions.csv")
    n_seeds = len(decisions)
    q_clean = np.array([float(d["quality"]) for d in decisions])
    sd_clean = float(q_clean.std(ddof=1))

    # 1. floor summary
    floor = {"size": args.size, "rule": args.rule, "seeds": n_seeds, "pairs": len(pairwise)}
    floor_metrics = ("allocation_churn", "directed_jaccard", "objective_drift")
    floor_metrics += ("load_rank_stability",)
    for metric in floor_metrics:
        vals = np.array([float(r[metric]) for r in pairwise])
        vals = vals[~np.isnan(vals)]
        lo, hi = _seed_cluster_boot(pairwise, metric, n_seeds)
        floor[metric] = {"mean": float(vals.mean()), "ci95": [lo, hi]}
    floor["clean_quality"] = {
        "mean": float(q_clean.mean()),
        "sd": sd_clean,
        "min": float(q_clean.min()),
        "max": float(q_clean.max()),
    }
    # Crisp baseline's clean quality, located in the fuzzy clean-seed distribution.
    problem = _load_problem(args.size)
    cfg = _config(100, args.generations)
    cache = precompute_fis_cache(problem, cfg)
    crisp_q = float(np.sum(evaluate_pairs(crisp_greedy_pairs(problem, cfg), cache, OBJECTIVES)))
    floor["crisp_clean_quality"] = crisp_q
    floor["crisp_percentile_in_fuzzy_seeds"] = float((q_clean < crisp_q).mean() * 100)
    if not args.skip_check:
        floor["consistency_check"] = _consistency_check(
            args.size, floor_dir, turb_dir, args.generations
        )
    (out_dir / "floor_summary.json").write_text(json.dumps(floor, indent=2) + "\n")
    churn_floor = floor["allocation_churn"]["mean"]

    # 2. per-cell churn above floor and quality-loss trend
    churn_rows, quality_rows = [], []
    for manifest in sorted(turb_dir.glob("*/turbulence_manifest.csv")):
        rows = _read_csv(manifest)
        cell = manifest.parent.name
        for system in ("fuzzy", "crisp"):
            # realisation -> level -> metric -> [values over reps]
            by: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for r in rows:
                if r["system"] == system:
                    lvl = float(r["level"])
                    for m in ("allocation_churn", "quality_loss"):
                        by[int(r["realization"])][lvl][m].append(float(r[m]))
            levels_x, loss_y = [], []
            for lvl in _FLOOR_LEVELS:
                churn = np.array([np.mean(by[z][lvl]["allocation_churn"]) for z in by])
                loss = np.array([np.mean(by[z][lvl]["quality_loss"]) for z in by])
                levels_x += [lvl] * loss.size
                loss_y += loss.tolist()
                c_lo, c_hi = _boot_mean_ci(churn)
                l_lo, l_hi = _boot_mean_ci(loss)
                churn_rows.append(
                    {
                        "cell": cell,
                        "system": system,
                        "level": lvl,
                        "churn_mean": float(churn.mean()),
                        "churn_ci_lo": c_lo,
                        "churn_ci_hi": c_hi,
                        "churn_minus_floor": float(churn.mean() - churn_floor)
                        if system == "fuzzy"
                        else float("nan"),
                        "quality_loss_mean": float(loss.mean()),
                        "quality_loss_ci_lo": l_lo,
                        "quality_loss_ci_hi": l_hi,
                        "loss_in_clean_sd": float(loss.mean() / sd_clean),
                    }
                )
            rho, p_two = spearmanr(levels_x, loss_y)
            p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
            quality_rows.append(
                {
                    "cell": cell,
                    "system": system,
                    "spearman_rho": float(rho),
                    "p_one_sided": float(p_one),
                }
            )
    for system in ("fuzzy", "crisp"):
        idx = [i for i, q in enumerate(quality_rows) if q["system"] == system]
        for i, p in zip(idx, _holm([quality_rows[i]["p_one_sided"] for i in idx])):
            quality_rows[i]["p_holm"] = p

    outputs = (("churn_quality_by_level.csv", churn_rows), ("quality_trend.csv", quality_rows))
    for name, data in outputs:
        with (out_dir / name).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    print(json.dumps({k: v for k, v in floor.items() if k != "consistency_check"}, indent=1))
    if "consistency_check" in floor:
        print("consistency check match:", floor["consistency_check"]["match"])
    print(f"\n{'cell':45s} sys   lvl  churn  -floor  qloss[CI]  /sd")
    for r in churn_rows:
        print(
            f"{r['cell']:45s} {r['system']:5s} {r['level']:.2f} {r['churn_mean']:.3f} "
            f"{r['churn_minus_floor']:+.3f} {r['quality_loss_mean']:+.2f}"
            f"[{r['quality_loss_ci_lo']:+.2f},{r['quality_loss_ci_hi']:+.2f}] "
            f"{r['loss_in_clean_sd']:+.2f}"
        )
    print("\ntrend (Spearman level vs quality_loss, one-sided, Holm within system):")
    for q in quality_rows:
        print(
            f"  {q['cell']:45s} {q['system']:5s} "
            f"rho={q['spearman_rho']:+.2f} p_holm={q['p_holm']:.4f}"
        )


if __name__ == "__main__":
    main()
