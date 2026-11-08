"""Trend tests for the exact decider's regret (Paper B, RQ3; revision R2.6).

One-sided Spearman test of regret against turbulence level per (size, field, mode) cell,
with realisations as units and Holm correction across all cells. Level 0 is the identity
check (zero regret by construction) and is excluded, so the test cannot be driven by it; the
script also prints the values with level 0 included, for comparison.

Reads ``<root>/<size>/<cell>/turbulence_manifest.csv`` (from ``run_turbulence_exact``);
writes ``experiments/results/turbulence_exact/trend_tests.csv``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adjusted[i] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/results/turbulence_exact"))
    args = parser.parse_args()
    rows = []
    for size in ("small", "large"):
        for cell in sorted(p for p in (args.root / size).iterdir() if p.is_dir()):
            data = [
                r
                for r in csv.DictReader((cell / "turbulence_manifest.csv").open())
                if r["system"] == "exact"
            ]
            for include_zero in (False, True):
                sel = [r for r in data if include_zero or float(r["level"]) > 0]
                rho, p_two = spearmanr(
                    [float(r["level"]) for r in sel], [float(r["quality_loss"]) for r in sel]
                )
                rows.append(
                    {
                        "size": size,
                        "cell": cell.name,
                        "level_0_included": include_zero,
                        "n": len(sel),
                        "rho": float(rho),
                        "p_one_sided": float(p_two / 2 if rho > 0 else 1 - p_two / 2),
                    }
                )
    for include_zero in (False, True):
        sub = [r for r in rows if r["level_0_included"] is include_zero]
        for r, adj in zip(sub, holm(np.array([r["p_one_sided"] for r in sub])), strict=True):
            r["p_holm"] = float(adj)
        rhos = [r["rho"] for r in sub]
        print(
            f"level 0 included={include_zero}: n={sub[0]['n']} per cell, "
            f"rho {min(rhos):.2f}-{max(rhos):.2f}, max Holm p {max(r['p_holm'] for r in sub):.2e}"
        )
    out = Path("experiments/results/turbulence_exact")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trend_tests.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
