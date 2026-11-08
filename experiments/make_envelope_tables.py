# ruff: noqa: E501  (LaTeX table rows and captions are long string literals)
"""LaTeX tables of the revised deployment envelope (Paper B revision, R2.7 / R3).

Reads ``experiments/results/envelope/envelope_t*_s*.csv`` and writes two fragments into
``--out``: ``tab_envelope.tex`` (means over instances, counts of solved and proven
instances, overload, latency) and ``tab_envelope_instances.tex`` (every instance, for the
appendix). Unsolved instances are counted and never averaged.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

ENV = Path("experiments/results/envelope")


def _load() -> list[dict]:
    rows = []
    for path in sorted(ENV.glob("envelope_t*_s*.csv")):
        rows += list(csv.DictReader(path.open()))
    for r in rows:
        r["limit"] = str(int(float(r["limit"])))
    return rows


def _cell(rows, size, limit, method):  # noqa: ANN001, ANN202
    return [
        r
        for r in rows
        if (int(r["people"]), int(r["centres"])) == size
        and r["limit"] == limit
        and r["method"] == method
    ]


def _mean(vals: list[float], fmt: str = "{:.1f}") -> str:
    return fmt.format(st.mean(vals)) if vals else "--"


def table_main(rows: list[dict]) -> str:
    sizes = sorted({(int(r["people"]), int(r["centres"])) for r in rows})
    out = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Deployment envelope on the repaired model, three instances per size; every method on"
        r" the same instances. Soft capacity ($\kappa=2$): quality and overload (people beyond free"
        r" capacity) in the soft model. Hard capacity ($\kappa_f=1.5$): quality in the hard model."
        r" Means over solved instances; $s/p$: instances with an incumbent / proven optimal. LS: local"
        r" search; seeded: NSGA-II seeded with exact solutions of the published model; fallback: exact"
        r" optimum of the published model. Latency: total decision time of the hard-capacity local"
        r" search, preparation of scores and load table included (s).\label{tab:envelope}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lr|rcrrrrr|rcrr}",
        r"\toprule",
        r"& & \multicolumn{7}{c|}{Soft capacity} & \multicolumn{4}{c}{Hard capacity}\\",
        r"Size & Limit & MIP & $s/p$ & LS & NSGA-II & Seeded & Fallback & Overload$^\ast$ & MIP & $s/p$ & LS & Latency\\",
        r"\midrule",
    ]
    for limit in ("60", "300"):
        for size in sizes:
            c = {
                m: _cell(rows, size, limit, m)
                for m in (
                    "mip_soft",
                    "ls_soft",
                    "nsga2_best_point",
                    "nsga2_seeded_best_point",
                    "fallback",
                    "mip_hard",
                    "ls_hard",
                )
            }
            if not c["fallback"]:
                continue

            def q(method: str, key: str = "quality_soft") -> list[float]:
                return [float(r[key]) for r in c[method] if r["solved"] == "True"]

            def sp(method: str) -> str:
                cell = c[method]
                return f"{sum(r['solved'] == 'True' for r in cell)}/{sum(r['optimal'] == 'True' for r in cell)}"

            ovl = "/".join(
                _mean([float(r["overload_soft"]) for r in c[m] if r["solved"] == "True"], "{:.0f}")
                for m in (
                    "mip_soft",
                    "ls_soft",
                    "nsga2_best_point",
                    "nsga2_seeded_best_point",
                    "fallback",
                )
            )
            latency = _mean([float(r["total_s"]) for r in c["ls_hard"]], "{:.0f}")
            out.append(
                f"${size[0]}\\times{size[1]}$ & {limit}\\,s & {_mean(q('mip_soft'))} & {sp('mip_soft')} & "
                f"{_mean(q('ls_soft'))} & {_mean(q('nsga2_best_point'))} & {_mean(q('nsga2_seeded_best_point'))} & "
                f"{_mean(q('fallback'))} & {ovl} & {_mean(q('mip_hard', 'quality_hard'))} & {sp('mip_hard')} & "
                f"{_mean(q('ls_hard', 'quality_hard'))} & {latency}\\\\"
            )
        out.append(r"\midrule" if limit == "60" else r"\bottomrule")
    out += [
        r"\multicolumn{13}{l}{$^\ast$ Overload of MIP / LS / NSGA-II / seeded / fallback, soft model. Hard-capacity"
        r" methods have zero overload by construction.}\\",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    return "\n".join(out) + "\n"


def table_instances(rows: list[dict]) -> str:
    methods = (
        ("mip_soft", "MIP soft"),
        ("ls_soft", "LS soft"),
        ("nsga2_best_point", "NSGA-II"),
        ("nsga2_seeded_best_point", "seeded"),
        ("fallback", "fallback"),
        ("mip_hard", "MIP hard"),
        ("ls_hard", "LS hard"),
    )
    out = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Deployment envelope per instance. Quality (soft model for soft methods, hard model"
        r" for hard ones); --: no incumbent within the limit; $^\dagger$: not proven optimal."
        r" Instances identified by size and seed $s$; their SHA-256 hashes are in the archived manifests.\label{tab:envelope-instances}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrr" + "r" * len(methods) + "}",
        r"\toprule",
        "Size & $s$ & Limit & " + " & ".join(label for _, label in methods) + r"\\",
        r"\midrule",
    ]
    keyset = sorted(
        {
            (int(r["people"]), int(r["centres"]), int(r["instance_seed"]), int(r["limit"]))
            for r in rows
        },
        key=lambda k: (k[3], k[0], k[2]),
    )
    for people, centres, seed, limit in keyset:
        cells = []
        for method, _ in methods:
            r = [
                x
                for x in rows
                if (
                    int(x["people"]),
                    int(x["centres"]),
                    int(x["instance_seed"]),
                    int(x["limit"]),
                    x["method"],
                )
                == (people, centres, seed, limit, method)
            ]
            if not r or r[0]["solved"] != "True":
                cells.append("--")
                continue
            key = "quality_hard" if method.endswith("hard") else "quality_soft"
            mark = "$^\\dagger$" if method.startswith("mip") and r[0]["optimal"] != "True" else ""
            cells.append(f"{float(r[0][key]):.2f}{mark}")
        out.append(f"${people}\\times{centres}$ & {seed} & {limit} & " + " & ".join(cells) + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def table_timing(rows: list[dict]) -> str:
    """Solver time, total latency and overload of every method, means over instances."""
    methods = (
        ("mip_soft", "MIP soft"),
        ("ls_soft", "LS soft"),
        ("nsga2_best_point", "NSGA-II"),
        ("nsga2_seeded_best_point", "seeded"),
        ("fallback", "fallback"),
        ("mip_hard", "MIP hard"),
        ("ls_hard", "LS hard"),
    )
    out = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Deployment envelope: solver time / total decision latency in seconds (preparation of"
        r" fuzzy scores and, for methods that see the load, CAIL-per-load tables included), and overload"
        r" (people beyond free capacity, soft model; hard-capacity methods have none). Means over the three"
        r" instances; a MIP that finds no incumbent counts with its time limit. All methods use the compiled"
        r" fuzzy evaluator.\label{tab:envelope-timing}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lr" + "r" * len(methods) + "|" + "r" * 5 + "}",
        r"\toprule",
        r"& & \multicolumn{"
        + str(len(methods))
        + r"}{c|}{Solver / total (s)} & \multicolumn{5}{c}{Overload}\\",
        "Size & Limit & "
        + " & ".join(label for _, label in methods)
        + r" & MIP & LS & NSGA-II & seeded & fallback\\",
        r"\midrule",
    ]
    sizes = sorted({(int(r["people"]), int(r["centres"])) for r in rows})
    for limit in ("60", "300"):
        for size in sizes:
            if not _cell(rows, size, limit, "fallback"):
                continue
            cells = []
            for method, _ in methods:
                c = _cell(rows, size, limit, method)
                cells.append(
                    _mean(
                        [float(r["solver_s"]) for r in c],
                        "{:.0f}" if method.startswith(("mip", "nsga")) else "{:.1f}",
                    )
                    + "/"
                    + _mean([float(r["total_s"]) for r in c], "{:.0f}")
                )
            for method in (
                "mip_soft",
                "ls_soft",
                "nsga2_best_point",
                "nsga2_seeded_best_point",
                "fallback",
            ):
                c = [r for r in _cell(rows, size, limit, method) if r["solved"] == "True"]
                cells.append(_mean([float(r["overload_soft"]) for r in c], "{:.0f}"))
            out.append(f"${size[0]}\\times{size[1]}$ & {limit}\\,s & " + " & ".join(cells) + r"\\")
        out.append(r"\midrule" if limit == "60" else r"\bottomrule")
    out += [r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = _load()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tab_envelope.tex").write_text(table_main(rows))
    (args.out / "tab_envelope_instances.tex").write_text(table_instances(rows))
    (args.out / "tab_envelope_timing.tex").write_text(table_timing(rows))
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
