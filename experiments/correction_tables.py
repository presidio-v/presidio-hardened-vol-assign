"""Tables 3, 4, A7/A8 and the H3b CVs of Paper A, from a results root.

Usage: python -m experiments.correction_tables <results_root> <out_prefix>
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

ROOT, OUT = Path(sys.argv[1]), sys.argv[2]
ALGS = ["nsga2", "nrga", "nsga3"]
METRICS = ["nns", "hv", "sm", "cpu_time_sec", "mid"]


def table3(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (size, alg), g in m[m.objectives == 4].groupby(["size", "algorithm"]):
        r = {"size": size, "alg": alg}
        for c in METRICS:
            x = g[c] / 1e6 if c == "hv" else g[c]
            r[c] = f"{x.mean():.2f} ± {x.std():.2f}"
        rows.append(r)
    return pd.DataFrame(rows)


def pvalues(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (obj, size), g in m.groupby(["objectives", "size"]):
        for a, b in [("nsga2", "nsga3"), ("nsga2", "nrga"), ("nsga3", "nrga")]:
            r = {"obj": obj, "size": size, "pair": f"{a}-{b}"}
            for c in METRICS:
                x, y = g[g.algorithm == a][c], g[g.algorithm == b][c]
                if x.nunique() == 1 and y.nunique() == 1 and x.iloc[0] == y.iloc[0]:
                    r[c] = "---"
                else:
                    p = mannwhitneyu(x, y, alternative="two-sided").pvalue
                    r[c] = "<0.001*" if p < 0.001 else f"{p:.3f}" + ("*" if p < 0.05 else "")
            rows.append(r)
    return pd.DataFrame(rows)


def table4(root: Path) -> pd.DataFrame:
    out = {}
    for size, d in [("small", "h3a_small"), ("medium", "h3a"), ("large", "h3a_large")]:
        h = pd.read_csv(root / d / "h3a_manifest.csv")
        h = h[h.fis_name != "baseline"]
        out[size] = h.groupby("fis_name").delta_hv_pct.apply(lambda s: s.abs().median())
        out[size + "_max"] = h.groupby("fis_name").delta_hv_pct.apply(lambda s: s.abs().max())
        out[size + "_n>5"] = h.groupby("fis_name").delta_hv_pct.apply(lambda s: (s.abs() > 5).sum())
    return pd.DataFrame(out).round(2)


def h3b(root: Path) -> pd.DataFrame:
    rows = {}
    for size, d in [("small", "h3b_small"), ("medium", "h3b"), ("large", "h3b_large")]:
        h = pd.read_csv(root / d / "h3b_manifest.csv")
        cols = ["mn_ulpp_avg", "mn_trd_avg", "mn_rpd_avg", "mn_cail_avg", "hv_avg", "nns_avg"]
        r = {c: 100 * h[c].std() / h[c].mean() for c in cols}
        r["ulpp_range"] = f"{h.mn_ulpp_avg.min():.2f}-{h.mn_ulpp_avg.max():.2f}"
        r["ulpp_min_range"] = f"{h.mn_ulpp_min_avg.min():.2f}-{h.mn_ulpp_min_avg.max():.2f}"
        r["rpd_spread"] = round(h.mn_rpd_avg.max() - h.mn_rpd_avg.min(), 3)
        rows[size] = r
    return pd.DataFrame(rows).T


m = pd.read_csv(ROOT / "h1_h2_h4" / "manifest.csv")
pd.set_option("display.width", 200)
for name, df in [
    ("table3", table3(m)),
    ("pvalues", pvalues(m)),
    ("table4", table4(ROOT)),
    ("h3b", h3b(ROOT)),
]:
    df.to_csv(f"{OUT}_{name}.csv")
    print(f"=== {name}\n{df.to_string()}\n")
