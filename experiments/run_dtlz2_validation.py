"""DTLZ2 validation for the DEAP NSGA-II and NSGA-III selectors.

Addresses Critical issue C1 in the adversarial review: the H2a refutation
in our humanitarian allocation problem (NSGA-II beating NSGA-III on HV)
must not be an artifact of `selNSGA3` being broken in our DEAP-based
framework. This script runs both selectors on the standard DTLZ2 benchmark
with DEAP's published continuous variation operators
(`cxSimulatedBinaryBounded` + `mutPolynomialBounded`), where the
literature consensus is that NSGA-III equals or outperforms NSGA-II at
$M=4$ objectives.

If NSGA-III achieves significantly higher HV than NSGA-II on DTLZ2-M4,
the selector is functioning correctly and the H2a result is a property
of our problem, not a bug. If not, our selector pairing is suspect and
the H2 narrative must be revised.

Usage:
    .venv/bin/python -m experiments.run_dtlz2_validation [--reps 30]
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
from deap import base, creator, tools
from pymoo.indicators.hv import HV
from pymoo.problems import get_problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--n-var", type=int, default=10, help="Decision variables (k+M-1).")
    parser.add_argument("--n-obj", type=int, default=4)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--p-divisions", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/results/dtlz2_validation"),
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "dtlz2_manifest.csv"

    problem = get_problem("dtlz2", n_var=args.n_var, n_obj=args.n_obj)

    def evaluate(individual: list) -> tuple[float, ...]:
        x = np.asarray(individual, dtype=float)
        f = problem.evaluate(x)
        return tuple(float(v) for v in f)

    # DEAP creator setup (idempotent)
    if not hasattr(creator, "DTLZFitness"):
        creator.create("DTLZFitness", base.Fitness, weights=(-1.0,) * args.n_obj)
    if not hasattr(creator, "DTLZIndividual"):
        creator.create("DTLZIndividual", list, fitness=creator.DTLZFitness)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register(
        "individual", tools.initRepeat, creator.DTLZIndividual, toolbox.attr_float, args.n_var
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=15.0, low=0.0, up=1.0)
    toolbox.register(
        "mutate", tools.mutPolynomialBounded, eta=20.0, low=0.0, up=1.0, indpb=1.0 / args.n_var
    )

    ref_dirs = tools.uniform_reference_points(nobj=args.n_obj, p=args.p_divisions)
    print(
        f"DTLZ2 validation: M={args.n_obj}, n_var={args.n_var}, "
        f"pop={args.pop_size}, gen={args.generations}, reps={args.reps}, "
        f"NSGA-III ref points={len(ref_dirs)}",
        flush=True,
    )

    # HV reference point — standard for DTLZ2 with M=4
    hv_ref = np.array([1.1] * args.n_obj)
    hv_indicator = HV(ref_point=hv_ref)

    def run_one(selector_kind: str, rep: int) -> dict:
        random.seed(1000 + rep * 7919)
        np.random.seed(1000 + rep * 7919)

        if selector_kind == "nsga2":
            select = tools.selNSGA2
        else:

            def select(inds: list, k: int) -> list:
                return tools.selNSGA3(inds, k, ref_dirs)

        pop = toolbox.population(n=args.pop_size)
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        t0 = time.monotonic()
        for _ in range(args.generations):
            offspring = [creator.DTLZIndividual(list(ind)) for ind in pop]
            for ind, parent in zip(offspring, pop):
                ind.fitness.values = parent.fitness.values
            for i in range(1, len(offspring), 2):
                if random.random() < 0.9:
                    toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values
            for ind in offspring:
                if random.random() < 1.0:
                    toolbox.mutate(ind)
                    if not ind.fitness.valid:
                        del ind.fitness.values  # already invalid; harmless
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)
            pop = select(pop + offspring, args.pop_size)
        cpu = time.monotonic() - t0

        front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        pts = np.asarray([ind.fitness.values for ind in front], dtype=float)
        # Drop points outside the reference box (rare in DTLZ2 but safe)
        keep = (pts < hv_ref).all(axis=1)
        pts = pts[keep]
        hv = float(hv_indicator(pts)) if pts.size else 0.0
        return {"hv": hv, "nns": len(front), "cpu_time_sec": cpu}

    rows = []
    with manifest_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["selector", "rep", "hv", "nns", "cpu_time_sec"])

        for selector_kind in ("nsga2", "nsga3"):
            for rep in range(args.reps):
                r = run_one(selector_kind, rep)
                writer.writerow(
                    [selector_kind, rep, round(r["hv"], 6), r["nns"], round(r["cpu_time_sec"], 3)]
                )
                fh.flush()
                rows.append({"selector": selector_kind, "rep": rep, **r})
                print(
                    f"  {selector_kind:7s} rep{rep:02d}  HV={r['hv']:.4f}  "
                    f"NNS={r['nns']}  cpu={r['cpu_time_sec']:.2f}s",
                    flush=True,
                )

    # Summary
    print()
    print("DTLZ2-M4 validation summary:")
    for kind in ("nsga2", "nsga3"):
        sub = [r for r in rows if r["selector"] == kind]
        hvs = np.array([r["hv"] for r in sub])
        cpus = np.array([r["cpu_time_sec"] for r in sub])
        print(
            f"  {kind:7s}  HV {hvs.mean():.4f} ± {hvs.std():.4f}  "
            f"CPU {cpus.mean():.2f} ± {cpus.std():.2f}s"
        )
    n2 = np.array([r["hv"] for r in rows if r["selector"] == "nsga2"])
    n3 = np.array([r["hv"] for r in rows if r["selector"] == "nsga3"])
    from scipy.stats import mannwhitneyu

    u = mannwhitneyu(n3, n2, alternative="two-sided")
    print(f"  Mann-Whitney U (NSGA-III vs NSGA-II HV): U={u.statistic:.0f}, p={u.pvalue:.4f}")
    print(f"  Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
