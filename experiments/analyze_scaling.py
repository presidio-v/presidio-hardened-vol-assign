"""Summarise the deployment-envelope runs (Paper B, RQ4) into one table per size.

Reads ``experiments/results/scaling/scaling_manifest*.csv`` (rows tagged with the manifest's
time limit) and adds the deterministic fallback (published-model greedy scored in the soft
repaired model) where a manifest lacks it. Writes ``scaling_summary.csv``.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.make_audit_figures import _greedy_fallback

DIR = Path("experiments/results/scaling")


def main() -> None:
    rows = []
    for path in sorted(DIR.glob("scaling_manifest*.csv")):
        match = re.search(r"_t(\d+)", path.stem)
        limit = int(match.group(1)) if match else 60
        rows += [{**r, "time_limit": limit} for r in csv.DictReader(path.open())]
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(int(r["people"]), int(r["centres"]), r["time_limit"])].append(r)
    out = []
    for (people, centres, limit), rs in sorted(groups.items()):
        seeds = sorted({int(r["instance_seed"]) for r in rs})

        def vals(method: str, field: str, rs=rs) -> list[float]:
            return [float(r[field]) for r in rs if r["method"] == method and r[field] != ""]

        def solved(method: str, rs=rs) -> int:
            return sum(1 for r in rs if r["method"] == method and r["quality"] != "")

        greedy = vals("greedy_static", "quality") or [
            _greedy_fallback(people, centres, s) for s in seeds
        ]
        row = {
            "people": people,
            "centres": centres,
            "time_limit_s": limit,
            "seeds": len(seeds),
            "mip_soft_q": np.mean(vals("mip_soft", "quality") or [np.nan]),
            "mip_soft_solved": solved("mip_soft"),
            "mip_soft_proven": sum(
                1 for r in rs if r["method"] == "mip_soft" and r["optimal"] == "True"
            ),
            "mip_soft_gap_max": max(vals("mip_soft", "gap") or [np.nan]),
            "mip_soft_s": np.mean(vals("mip_soft", "seconds") or [np.nan]),
            "mip_hard_solved": solved("mip_hard"),
            "mip_hard_proven": sum(
                1 for r in rs if r["method"] == "mip_hard" and r["optimal"] == "True"
            ),
            "mip_hard_s": np.mean(vals("mip_hard", "seconds") or [np.nan]),
            "moea_q": np.mean(vals("moea_committed", "quality") or [np.nan]),
            "moea_generations": np.mean(vals("moea_committed", "generations") or [np.nan]),
            "greedy_q": float(np.mean(greedy)),
        }
        out.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    with (DIR / "scaling_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    main()
