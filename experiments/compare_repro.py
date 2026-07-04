"""Cross-environment reproducibility comparison (Paper B, RQ2).

Given a directory holding one ``repro_manifest.csv`` per environment (e.g. the
downloaded artifacts of a CI matrix), compute cross-environment REP for each
(size, seed): 1.0 iff every environment produced the same allocation-front
signature. Prints a table and the overall verdict and writes cross_env_repro.csv.

Divergence is reported, not gated: cross-platform bit-for-bit reproducibility of a
float/library-dependent solver is not guaranteed, and *which* (size, seed,
environment) diverged is precisely the audit trail this study argues for.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifests = sorted(args.results_dir.rglob("repro_manifest.csv"))
    if not manifests:
        raise SystemExit(f"no repro_manifest.csv under {args.results_dir}")

    # (size, seed) -> {environment label: signature}
    by_key: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for manifest in manifests:
        for r in csv.DictReader(manifest.open()):
            env = f"{r['platform']}|py{r['python']}|np{r['numpy']}|deap{r['deap']}"
            by_key[(r["size"], r["seed"])][env] = r["signature"]

    summary = []
    for (size, seed), sigs in sorted(by_key.items()):
        distinct = len(set(sigs.values()))
        summary.append(
            {
                "size": size,
                "seed": seed,
                "n_envs": len(sigs),
                "distinct_sigs": distinct,
                "cross_env_rep": 1.0 if distinct == 1 else 0.0,
            }
        )

    overall = min((s["cross_env_rep"] for s in summary), default=0.0)
    n_env = max((s["n_envs"] for s in summary), default=0)
    print(f"{'size':8}{'seed':>10}{'n_envs':>8}{'distinct':>10}{'rep':>6}")
    for s in summary:
        print(
            f"{s['size']:8}{s['seed']:>10}{s['n_envs']:>8}{s['distinct_sigs']:>10}{s['cross_env_rep']:>6}"
        )
    print(
        f"\ncross-environment REP over {len(summary)} (size,seed) keys "
        f"across up to {n_env} environments: {overall}"
    )

    out = args.out or (args.results_dir / "cross_env_repro.csv")
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
