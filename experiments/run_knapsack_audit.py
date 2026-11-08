"""Second case, third-party benchmark (Paper B revision, R2.1): the protocol on the
multi-objective 0/1 knapsack problem of Zitzler & Thiele (1999).

The problem is not ours and S1 does not settle it: k knapsacks, each item has a profit and
a weight per knapsack, every knapsack has a capacity, and the objectives are the k total
profits (maximised). The capacity constraints couple the items, so a weighted sum is a
multi-constraint knapsack, NP-hard in general. Instances follow the published recipe:
integer profits and weights uniform in [10, 100], capacity half of the total weight per
knapsack; k = 2, with 250 and 500 items.

S2 solves every weighted sum on a lattice of weight vectors exactly with HiGHS. S3 runs
NSGA-II (pymoo; binary encoding, uniform crossover, bit-flip mutation at 1/n, and the
Zitzler-Thiele greedy repair that drops items by increasing max profit/weight ratio until
every knapsack fits) under 20 seeds at two budgets, and compares committed decisions
(front point minimising the equal-weight, min-max-normalised objective sum) across seeds:
churn is the Jaccard distance between the two selected item sets.

Writes ``experiments/results/knapsack/{fronts,exact,floor}.csv``.
"""

from __future__ import annotations

import csv
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from scipy.optimize import Bounds, LinearConstraint, milp

OUT = Path("experiments/results/knapsack")
SIZES = (250, 500)
KNAPSACKS = 2
INSTANCE_SEED = 1999
SEEDS = tuple(1000 + 7919 * k for k in range(20))
BUDGETS = (200, 2000)  # generations, population 100
SWEEP = 100  # weight vectors (w, 1 - w), w = 0, 1/100, ..., 1


def make_instance(n: int, k: int = KNAPSACKS, seed: int = INSTANCE_SEED):
    rng = np.random.Generator(np.random.PCG64(seed + n))
    profit = rng.integers(10, 101, size=(k, n)).astype(float)
    weight = rng.integers(10, 101, size=(k, n)).astype(float)
    capacity = weight.sum(axis=1) / 2.0
    return profit, weight, capacity


class _Knapsack(Problem):
    def __init__(self, profit, weight, capacity):
        self.profit, self.weight, self.capacity = profit, weight, capacity
        super().__init__(n_var=profit.shape[1], n_obj=profit.shape[0], xl=0, xu=1, vtype=bool)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -(x.astype(float) @ self.profit.T)


class _GreedyRepair(Repair):
    """Zitzler-Thiele repair: drop items by increasing max_j p_ij / w_ij until feasible."""

    def _do(self, problem, x, **kwargs):
        order = np.argsort((problem.profit / problem.weight).max(axis=0), kind="stable")
        x = x.astype(bool).copy()
        for row in x:
            load = problem.weight @ row
            for item in order:
                if (load <= problem.capacity).all():
                    break
                if row[item]:
                    row[item] = False
                    load -= problem.weight[:, item]
        return x


def exact_weighted(profit, weight, capacity, w: np.ndarray) -> np.ndarray:
    n = profit.shape[1]
    res = milp(
        -(w @ profit),
        constraints=LinearConstraint(weight, -np.inf, capacity),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        options={"time_limit": 60, "disp": False},
    )
    if res.status != 0:
        raise RuntimeError(f"weighted knapsack not solved to optimality: {res.message}")
    return res.x > 0.5


def committed(front_x: np.ndarray, front_f: np.ndarray) -> np.ndarray:
    lo, hi = front_f.min(axis=0), front_f.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return front_x[int(np.argmin(((front_f - lo) / span).sum(axis=1)))]


def jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 - np.logical_and(a, b).sum() / union if union else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exact_rows, floor_rows, front_rows = [], [], []
    for n in SIZES:
        profit, weight, capacity = make_instance(n)
        t = time.perf_counter()
        sweep = []
        for a in range(SWEEP + 1):
            w = np.array([a / SWEEP, 1 - a / SWEEP])
            x = exact_weighted(profit, weight, capacity, w)
            sweep.append((x, profit @ x))
        sweep_s = time.perf_counter() - t
        eq = exact_weighted(profit, weight, capacity, np.array([0.5, 0.5]))
        eq_f = profit @ eq
        supported = np.unique(np.array([f for _, f in sweep]), axis=0)
        exact_rows.append(
            {
                "items": n,
                "eq_profit_1": eq_f[0],
                "eq_profit_2": eq_f[1],
                "eq_sum": eq_f.sum(),
                "supported_points": len(supported),
                "sweep_seconds": sweep_s,
            }
        )
        print(exact_rows[-1], flush=True)
        problem = _Knapsack(profit, weight, capacity)
        for gens in BUDGETS:
            commits = []
            for seed in SEEDS:
                algo = NSGA2(
                    pop_size=100,
                    sampling=BinaryRandomSampling(),
                    crossover=UniformCrossover(),
                    mutation=BitflipMutation(prob=1.0, prob_var=1.0 / n),
                    repair=_GreedyRepair(),
                    eliminate_duplicates=True,
                )
                t = time.perf_counter()
                res = minimize(problem, algo, ("n_gen", gens), seed=seed, verbose=False)
                secs = time.perf_counter() - t
                fx, ff = np.atleast_2d(res.X).astype(bool), -np.atleast_2d(res.F)
                dominated = int(sum((eq_f >= f).all() and (eq_f > f).any() for f in ff))
                commits.append(committed(fx, -ff))
                front_rows.append(
                    {
                        "items": n,
                        "generations": gens,
                        "seed": seed,
                        "points": len(ff),
                        "dominated_by_exact": dominated,
                        "best_sum": float(ff.sum(axis=1).max()),
                        "exact_sum": float(eq_f.sum()),
                        "seconds": secs,
                    }
                )
            churn = [jaccard_distance(a, b) for a, b in combinations(commits, 2)]
            q = [float((profit @ c).sum()) for c in commits]
            floor_rows.append(
                {
                    "items": n,
                    "generations": gens,
                    "seeds": len(SEEDS),
                    "churn_mean": float(np.mean(churn)),
                    "churn_min": float(np.min(churn)),
                    "churn_max": float(np.max(churn)),
                    "committed_sum_mean": float(np.mean(q)),
                    "committed_sum_sd": float(np.std(q, ddof=1)),
                    "exact_sum": float(eq_f.sum()),
                    "dominated_share": sum(
                        r["dominated_by_exact"]
                        for r in front_rows
                        if r["items"] == n and r["generations"] == gens
                    )
                    / sum(
                        r["points"]
                        for r in front_rows
                        if r["items"] == n and r["generations"] == gens
                    ),
                }
            )
            print(floor_rows[-1], flush=True)
    for name, rows in (("exact", exact_rows), ("floor", floor_rows), ("fronts", front_rows)):
        with (OUT / f"{name}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
