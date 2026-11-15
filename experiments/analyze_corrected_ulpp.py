"""RQ3 and RQ4 under the corrected ULPP objective against the published one (Paper B, round 2).

Reads the exact-decider turbulence manifests of ``run_turbulence_exact`` for both objectives
(``results/turbulence_exact`` and ``results/turbulence_exact_left_out``) and the RQ4 grids of
``run_repair_sweep`` (``results/repair/kappa_sweep.csv`` and ``kappa_sweep_left_out.csv``).
Checks, per cell, that level 0 reproduces the clean decision (zero regret and churn) and that no
regret is negative, then writes

* ``results/turbulence_exact_left_out/comparison.csv``: mean regret per cell and level, and
  mean churn at level 0.4, for both objectives;
* ``--table``: the LaTeX table of the paper (regret at levels 0.05 and 0.4, churn at 0.4);
* ``--extended-table``: the supplementary table of the 36-condition study, both objectives
  (needs ``analyze_turbulence_extended`` run on both result roots).

Usage::

    python -m experiments.analyze_corrected_ulpp --table <dir>/tab_corrected_rq3.tex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

RESULTS = Path("experiments/results")
ROOTS = {
    "published": RESULTS / "turbulence_exact",
    "corrected": RESULTS / "turbulence_exact_left_out",
}
CELLS = (
    ("travel_duration_noise", "Travel time, noise"),
    ("resource_time_remaining_noise", "Resource time, noise"),
    ("infrastructure_damage_level_noise", "Damage, noise"),
    ("center_occupancy_rate_noise", "Occupancy, noise"),
    ("road_condition_flip", "Road, flip"),
    ("possible_hazard_flip", "Hazard, flip"),
    ("infrastructure_damage_level_missingness", "Damage, missing"),
    ("center_occupancy_rate_missingness", "Occupancy, missing"),
)
LEVELS = (0.05, 0.1, 0.2, 0.4)
_TOL = 1e-9


def _cell_stats(manifest: Path) -> dict[str, float]:
    rows = [r for r in csv.DictReader(manifest.open()) if r["system"] == "exact"]
    if not rows:
        raise ValueError(f"{manifest}: no exact-decider rows")

    def at(level: float, key: str) -> list[float]:
        return [float(r[key]) for r in rows if abs(float(r["level"]) - level) < _TOL]

    # fail closed: a decider that moves on clean data, or a negative regret, means a broken run
    if any(abs(x) > _TOL for x in at(0.0, "quality_loss") + at(0.0, "allocation_churn")):
        raise ValueError(f"{manifest}: level 0 does not reproduce the clean decision")
    if min(float(r["quality_loss"]) for r in rows) < -_TOL:
        raise ValueError(f"{manifest}: negative regret")
    out = {f"regret_{lv:g}": mean(at(lv, "quality_loss")) for lv in LEVELS}
    out["churn_0.4"] = mean(at(0.4, "allocation_churn"))
    return out


def compare() -> list[dict]:
    rows = []
    for size in ("small", "large"):
        for cell, label in CELLS:
            row = {"size": size, "cell": cell, "label": label}
            for name, root in ROOTS.items():
                stats = _cell_stats(root / size / cell / "turbulence_manifest.csv")
                row.update({f"{name}_{k}": v for k, v in stats.items()})
            rows.append(row)
    return rows


def latex(rows: list[dict]) -> str:
    by = {(r["size"], r["cell"]): r for r in rows}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{RQ3 on the reference instances under the published and the corrected ULPP"
        r" objective: mean regret of the exact decider over $50$ realisations at levels $0.05$ and"
        r" $0.4$, and mean allocation churn at $0.4$. Published: the values behind"
        r" Section~\ref{sec:rq3}; corrected: ULPP averaged over the people left out."
        r"\label{tab:corrected}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l rrr rrr rrr rrr}",
        r"\toprule",
        r"& \multicolumn{6}{c}{Small} & \multicolumn{6}{c}{Large}\\",
        r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
        r"& \multicolumn{3}{c}{Published} & \multicolumn{3}{c}{Corrected}"
        r" & \multicolumn{3}{c}{Published} & \multicolumn{3}{c}{Corrected}\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        r"Field, mode" + r" & $0.05$ & $0.4$ & churn" * 4 + r"\\",
        r"\midrule",
    ]
    for cell, label in CELLS:
        vals = []
        for size in ("small", "large"):
            r = by[(size, cell)]
            for name in ("published", "corrected"):
                vals += [
                    f"{r[f'{name}_regret_0.05']:.2f}",
                    f"{r[f'{name}_regret_0.4']:.1f}",
                    f"{r[f'{name}_churn_0.4']:.2f}",
                ]
        lines.append(label + " & " + " & ".join(vals) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


EXTENDED = {
    "published": RESULTS / "turbulence_extended" / "summary.csv",
    "corrected": RESULTS / "turbulence_extended_left_out" / "summary.csv",
}
EXT_CELLS = (
    *CELLS,
    ("travel_duration_bias_up", r"Travel time, bias $+$"),
    ("travel_duration_bias_down", r"Travel time, bias $-$"),
    ("resource_time_remaining_bias_up", r"Resource time, bias $+$"),
    ("resource_time_remaining_bias_down", r"Resource time, bias $-$"),
    ("infrastructure_damage_level_bias_up", r"Damage, bias $+$"),
    ("infrastructure_damage_level_bias_down", r"Damage, bias $-$"),
    ("center_occupancy_rate_bias_up", r"Occupancy, bias $+$"),
    ("center_occupancy_rate_bias_down", r"Occupancy, bias $-$"),
    ("joint_continuous_noise", "Joint: continuous noise"),
    ("joint_transport", "Joint: transport"),
    ("joint_all", "Joint: all eight"),
)


def extended_latex() -> str:
    """Supplementary table: the 36-condition study, published vs corrected, at level 0.4."""
    data = {}
    for name, path in EXTENDED.items():
        for r in csv.DictReader(path.open()):
            if abs(float(r["level"]) - 0.4) < _TOL:
                data[(name, r["size"], r["cell"])] = r
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{RQ3 across $36$ conditions per size under the published and the corrected ULPP"
        r" objective, level $0.4$: median over conditions of the per-condition mean regret, and the"
        r" share of conditions whose trend test is significant after Holm correction (bias cells:"
        r" one realisation per level, little power).}\label{tab:s-extended}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l rrrr rrrr}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Small} & \multicolumn{4}{c}{Large}\\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"& \multicolumn{2}{c}{Published} & \multicolumn{2}{c}{Corrected}"
        r" & \multicolumn{2}{c}{Published} & \multicolumn{2}{c}{Corrected}\\",
        r"Field, mode" + r" & regret & sig." * 4 + r"\\",
        r"\midrule",
    ]
    for cell, label in EXT_CELLS:
        vals = []
        for size in ("small", "large"):
            for name in ("published", "corrected"):
                r = data[(name, size, cell)]
                vals += [f"{float(r['regret_median']):.1f}", f"{float(r['share_significant']):.2f}"]
        lines.append(label + " & " + " & ".join(vals) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--extended-table", type=Path, default=None)
    args = parser.parse_args()
    rows = compare()
    out = ROOTS["corrected"] / "comparison.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    for size in ("small", "large"):
        for name in ROOTS:
            sub = sorted(
                (r for r in rows if r["size"] == size), key=lambda r: -r[f"{name}_regret_0.4"]
            )
            print(f"{size} {name} ranking at 0.4: " + ", ".join(r["label"] for r in sub))
    if args.table is not None:
        args.table.write_text(latex(rows))
        print(f"wrote {args.table}")
    if args.extended_table is not None:
        args.extended_table.write_text(extended_latex())
        print(f"wrote {args.extended_table}")


if __name__ == "__main__":
    main()
