"""Second case, a coupled model (Paper B revision, R2.1): the protocol on the ED
volunteer-staffing model of Rabiei et al. (2023).

S1. Each emergency department posts one vacancy of one skill type; every vacancy must be
filled by exactly one volunteer of that type, and no volunteer fills two. The objectives
(mean importance of unmet needs Z1, mean preference dissatisfaction Z2) are means of
per-(volunteer, vacancy) FIS scores, but the one-to-one constraint couples the choices:
the model is not separable. A weighted sum is a rectangular linear assignment problem,
so S2 is still exact, in polynomial time (``scipy.optimize.linear_sum_assignment``).

S2 sweeps w in {0, 0.01, ..., 1} for the exact supported reference front and tests the
evolutionary fronts against the exact equal-weight optimum. S3 runs NSGA-II and NRGA (the
published permutation encoding and operators, population 100, 200 generations) under 20
seeds and compares committed decisions (the front point minimising the equal-weight,
min-max-normalised sum): churn is the fraction of vacancies filled by a different
volunteer. Instances come from the package's deterministic benchmark generator.

Writes ``experiments/results/ed_audit/{exact,fronts,floor}.csv``.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from presidio_vol_assign.benchmark import generate_instance
from presidio_vol_assign.domains import get_domain
from presidio_vol_assign.engine import run
from presidio_vol_assign.models import RunConfig

OUT = Path("experiments/results/ed_audit")
INSTANCES = {"small": 1000, "large": 101000}  # benchmark seeds (size offset 0 / 100000)
SEEDS = tuple(1000 + 7919 * k for k in range(20))
SOLVERS = ("nsga2", "nrga")
INFEASIBLE = 1e6


def score_matrices(problem, cache):  # noqa: ANN001, ANN201
    nv, nj = len(problem.volunteers), len(problem.vacancies)
    imp, pref = np.ones((nv, nj)), np.ones((nv, nj))
    feasible = np.zeros((nv, nj), dtype=bool)
    for (vi, vj), entry in cache.items():
        if entry is not None:
            imp[vi, vj], pref[vi, vj] = entry
            feasible[vi, vj] = True
    return imp, pref, feasible


def exact_assignment(imp, pref, feasible, w: float):  # noqa: ANN001, ANN201
    cost = np.where(feasible, w * imp + (1.0 - w) * pref, INFEASIBLE)
    rows, cols = linear_sum_assignment(cost.T)  # rows: vacancies, cols: volunteers
    pairs = sorted((int(v), int(j)) for j, v in zip(rows, cols, strict=True))
    if any(not feasible[v, j] for v, j in pairs):
        raise RuntimeError("no feasible assignment")
    return (
        pairs,
        float(np.mean([imp[v, j] for v, j in pairs])),
        float(np.mean([pref[v, j] for v, j in pairs])),
    )


def _committed(front, problem):  # noqa: ANN001, ANN202
    pts = np.array([(s.z1, s.z2) for s in front.solutions])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    sol = front.solutions[int(np.argmin(((pts - lo) / span).sum(axis=1)))]
    vid = {v.volunteer_id: i for i, v in enumerate(problem.volunteers)}
    jid = {v.ed_id: j for j, v in enumerate(problem.vacancies)}
    return {(vid[a.volunteer_id], jid[a.ed_id]) for a in sol.assignments}, sol.z1 + sol.z2


def _cluster_ci(
    values: dict[tuple[int, int], float], n: int, reps: int = 2000
) -> tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64(0))
    means = []
    for _ in range(reps):
        pick = rng.integers(0, n, size=n)
        vals = [values[(min(a, b), max(a, b))] for a, b in combinations(pick, 2) if a != b]
        means.append(np.mean(vals))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    domain = get_domain("ed-staffing")
    exact_rows, front_rows, floor_rows = [], [], []
    for size, seed in INSTANCES.items():
        problem = generate_instance("ed-staffing", size, seed)
        cache = domain.precompute(problem)
        imp, pref, feasible = score_matrices(problem, cache)
        _, e1, e2 = exact_assignment(imp, pref, feasible, 0.5)
        supported = {
            (round(z1, 12), round(z2, 12))
            for z1, z2 in (exact_assignment(imp, pref, feasible, a / 100)[1:] for a in range(101))
        }
        exact_rows.append(
            {
                "size": size,
                "volunteers": len(problem.volunteers),
                "vacancies": len(problem.vacancies),
                "exact_z1": e1,
                "exact_z2": e2,
                "exact_sum": e1 + e2,
                "supported_points": len(supported),
            }
        )
        print(exact_rows[-1], flush=True)
        for solver in SOLVERS:
            commits, sums = [], []
            for s in SEEDS:
                front = run(
                    problem, RunConfig(solver=solver, pop_size=100, generations=200, seed=s), domain
                )[0]
                pts = [(x.z1, x.z2) for x in front.solutions]
                dominated = sum(1 for a, b in pts if e1 <= a and e2 <= b and (e1 < a or e2 < b))
                front_rows.append(
                    {
                        "size": size,
                        "solver": solver,
                        "seed": s,
                        "points": len(pts),
                        "dominated_by_exact": dominated,
                        "best_sum": min(a + b for a, b in pts),
                        "exact_sum": e1 + e2,
                        "seconds": front.cpu_time_sec,
                    }
                )
                c, q = _committed(front, problem)
                commits.append(c)
                sums.append(q)
            nj = len(problem.vacancies)
            churn = {
                (a, b): 1.0 - len(commits[a] & commits[b]) / nj
                for a, b in combinations(range(len(SEEDS)), 2)
            }
            lo, hi = _cluster_ci(churn, len(SEEDS))
            rows = [r for r in front_rows if r["size"] == size and r["solver"] == solver]
            floor_rows.append(
                {
                    "size": size,
                    "solver": solver,
                    "seeds": len(SEEDS),
                    "churn_mean": float(np.mean(list(churn.values()))),
                    "churn_ci_lo": lo,
                    "churn_ci_hi": hi,
                    "committed_sum_mean": float(np.mean(sums)),
                    "committed_sum_sd": float(np.std(sums, ddof=1)),
                    "exact_sum": e1 + e2,
                    "dominated": sum(r["dominated_by_exact"] for r in rows),
                    "points": sum(r["points"] for r in rows),
                    "best_sum_mean": float(np.mean([r["best_sum"] for r in rows])),
                    "seeds_reaching_exact": sum(1 for r in rows if r["best_sum"] <= e1 + e2 + 1e-9),
                }
            )
            print(floor_rows[-1], flush=True)
    for name, rows in (("exact", exact_rows), ("fronts", front_rows), ("floor", floor_rows)):
        with (OUT / f"{name}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
