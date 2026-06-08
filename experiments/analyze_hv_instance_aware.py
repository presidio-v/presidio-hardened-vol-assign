"""Instance-aware HV recomputation (M3).

Addresses Moderate issue M3 in the adversarial review: HV in the main
matrix uses the fixed (100,100,100,100) reference point at the FIS output-
universe maxima. A reviewer experienced in MOEA benchmarking will note
that an instance-aware reference --- the worst-per-objective across the
union of all algorithms' fronts on the same (size, rep) instance --- is
more discriminating between near-optimal fronts.

For each (size, rep) cell, we union the three algorithms' Pareto fronts
and take the per-objective max as a tight reference (then add a 5%
margin to keep all points strictly inside the reference box). HV is
recomputed per (size, algorithm, rep) and the rankings are compared
against the fixed-reference HV from the main matrix.

If the H2a verdict (NSGA-II > NSGA-III on HV) holds under the
instance-aware reference, the original analysis is robust to the
methodological concern. If it flips, §5.4 needs a different narrative.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from scipy.stats import mannwhitneyu


def _front_points(run_dir: Path) -> np.ndarray:
    csv = next(run_dir.glob("pareto_alloc_*.csv"))
    df = pd.read_csv(csv)
    return df[["mn_ulpp", "mn_trd", "mn_rpd", "mn_cail"]].to_numpy(dtype=float)


def main() -> None:
    manifest = pd.read_csv("experiments/results/h1_h2_h4/manifest.csv")
    manifest = manifest[manifest["objectives"] == 4]

    rows = []
    for (size, rep), cell in manifest.groupby(["size", "rep"]):
        # Union of points across all three algorithms in this cell
        union = np.vstack(
            [
                _front_points(Path("experiments/results/h1_h2_h4") / row["run_id"])
                for _, row in cell.iterrows()
            ]
        )
        ref = union.max(axis=0) * 1.05  # 5% margin to keep all points inside box
        indicator = HV(ref_point=ref)

        for _, row in cell.iterrows():
            pts = _front_points(Path("experiments/results/h1_h2_h4") / row["run_id"])
            keep = (pts < ref).all(axis=1)
            pts_in = pts[keep]
            hv_aware = float(indicator(pts_in)) if pts_in.size else 0.0
            rows.append(
                {
                    "size": size,
                    "algorithm": row["algorithm"],
                    "rep": rep,
                    "hv_fixed_100": float(row["hv"]),
                    "hv_instance_aware": hv_aware,
                    "ref_ulpp": ref[0],
                    "ref_trd": ref[1],
                    "ref_rpd": ref[2],
                    "ref_cail": ref[3],
                }
            )

    df = pd.DataFrame(rows)
    out = Path("experiments/results/h1_h2_h4/hv_instance_aware.csv")
    df.to_csv(out, index=False)

    # Summary
    print("=" * 88)
    print("HV per (size × algorithm), fixed ref vs instance-aware ref")
    print("=" * 88)
    print(f"{'size':<8} {'algorithm':<10} {'fixed mean':>14} {'aware mean':>14} {'ratio':>8}")
    for size in ["small", "medium", "large"]:
        for algo in ["nsga2", "nrga", "nsga3"]:
            sub = df[(df["size"] == size) & (df["algorithm"] == algo)]
            mfixed = sub["hv_fixed_100"].mean()
            maware = sub["hv_instance_aware"].mean()
            print(f"{size:<8} {algo:<10} {mfixed:>14.0f} {maware:>14.0f} {maware / mfixed:>8.4f}")

    print()
    print("=" * 88)
    print("H2a verdict under instance-aware reference (NSGA-III vs NSGA-II)")
    print("=" * 88)
    for size in ["small", "medium", "large"]:
        sub = df[df["size"] == size]
        n2 = sub[sub["algorithm"] == "nsga2"]["hv_instance_aware"].values
        n3 = sub[sub["algorithm"] == "nsga3"]["hv_instance_aware"].values
        u = mannwhitneyu(n3, n2, alternative="two-sided")
        direction = "NSGA-III > NSGA-II" if n3.mean() > n2.mean() else "NSGA-II > NSGA-III"
        print(
            f"  {size:<6}  NSGA-II={n2.mean():>14.0f}  NSGA-III={n3.mean():>14.0f}  "
            f"{direction}  p={u.pvalue:.4f}"
        )

    print(f"\nDetail → {out}")


if __name__ == "__main__":
    main()
