"""Statistical significance testing for benchmark comparisons.

The ATRES reviewers noted that the benchmark reported only mean ± std and ran no
statistical test. Following the common practice in the multi-objective
literature (and the reviewer's own suggestion), this module compares solvers'
per-instance **hypervolume** distributions with the **Wilcoxon rank-sum test**
(a.k.a. Mann–Whitney U; ``scipy.stats.ranksums``, already a dependency — no new
package). Higher HV is better, so the solver with the larger median HV is the
"better" side; the two-sided p-value says whether the difference is significant.

Each ``(size)`` group is compared against a single reference solver: the greedy
baseline when present (the headline framework-vs-baseline comparison), otherwise
NSGA-II, otherwise the first solver alphabetically.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ranksums

# Minimum per-solver sample count (instances) for a meaningful rank-sum test.
MIN_SAMPLES = 5


@dataclass
class HvComparison:
    """Outcome of one Wilcoxon rank-sum HV comparison between two solvers."""

    size: str
    reference: str  # solver_b — the baseline/reference side
    solver: str  # solver_a — the side being tested against the reference
    n: int  # samples per side
    median: float  # median HV of ``solver``
    reference_median: float  # median HV of ``reference``
    statistic: float
    p_value: float
    better: str  # which solver has the higher median HV ("tie" if equal)
    significant: bool  # p_value < alpha


def _reference_solver(solvers: set[str]) -> str:
    """Pick the reference (baseline) solver for a size group."""
    if "greedy" in solvers:
        return "greedy"
    if "nsga2" in solvers:
        return "nsga2"
    return sorted(solvers)[0]


def wilcoxon_hv_tests(rows: list, *, alpha: float = 0.05) -> list[HvComparison]:
    """Pairwise Wilcoxon rank-sum HV tests vs. a reference solver, per size.

    ``rows`` are :class:`~presidio_vol_assign.benchmark.BenchmarkRow` objects
    carrying ``hv_samples``. Returns one :class:`HvComparison` per (size, solver)
    pair where both the solver and the reference have at least ``MIN_SAMPLES``
    HV samples. Returns an empty list when no group qualifies (e.g. too few
    instances, or only one solver), so callers can simply skip the report.
    """
    by_size: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for r in rows:
        if r.hv_samples:
            by_size[r.size][r.solver] = list(r.hv_samples)

    comparisons: list[HvComparison] = []
    for size, samples in by_size.items():
        eligible = {s: hv for s, hv in samples.items() if len(hv) >= MIN_SAMPLES}
        if len(eligible) < 2:
            continue
        reference = _reference_solver(set(eligible))
        ref_hv = eligible[reference]
        ref_median = float(np.median(ref_hv))
        for solver, hv in sorted(eligible.items()):
            if solver == reference:
                continue
            statistic, p_value = ranksums(hv, ref_hv)
            median = float(np.median(hv))
            if median > ref_median:
                better = solver
            elif median < ref_median:
                better = reference
            else:
                better = "tie"
            comparisons.append(
                HvComparison(
                    size=size,
                    reference=reference,
                    solver=solver,
                    n=min(len(hv), len(ref_hv)),
                    median=median,
                    reference_median=ref_median,
                    statistic=float(statistic),
                    p_value=float(p_value),
                    better=better,
                    significant=bool(p_value < alpha),
                )
            )
    return comparisons


def write_hv_tests_csv(comparisons: list[HvComparison], output_dir: Path) -> Path:
    """Write the HV significance comparisons to ``stats_<ts>.csv``.

    The filename deliberately avoids the ``benchmark_`` prefix so it is not
    confused with (or globbed alongside) the Table-3 summary files.
    """
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    csv_path = output_dir / f"stats_{ts}.csv"
    pd.DataFrame([asdict(c) for c in comparisons]).to_csv(csv_path, index=False)
    return csv_path
