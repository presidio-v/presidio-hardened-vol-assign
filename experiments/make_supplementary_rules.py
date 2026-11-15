"""Rule bases of the audited fuzzy systems as LaTeX tables (Paper B, supplementary S1).

Generated from the rule constants the implementation runs (``allocation.fis``), so the
supplementary tables cannot drift from the code. Each table is a grid: rows and columns are
the input terms, cells the output term (VL, L, M, H, VH: very low to very high).

Usage::

    python -m experiments.make_supplementary_rules --out <dir>/rules.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from presidio_vol_assign.allocation.fis import (
    FIS1_RULES,
    FIS2_TIL_RULES,
    FIS2A_TRD_RULES,
    FIS2B_RPD_RULES,
    FIS3_RULES,
)

_TERM = {
    "very_low": "VL",
    "low": "L",
    "moderate": "M",
    "medium": "M",
    "high": "H",
    "very_high": "VH",
}


def _t(term: str) -> str:
    """Output term as its abbreviation (VL, L, M, H, VH)."""
    return _TERM[term]


def _in(term: str) -> str:
    """Input term spelled out, as in the rule base."""
    return term.replace("_", " ")


def _grid(rules, row_name: str, col_name: str, caption: str, label: str) -> str:
    """One table from 3-tuples (row term, column term, output term), order as in the code."""
    rows = list(dict.fromkeys(r for r, _, _ in rules))
    cols = list(dict.fromkeys(c for _, c, _ in rules))
    cell = {(r, c): o for r, c, o in rules}
    if len(cell) != len(rules):  # fail closed: a duplicate antecedent would be silently lost
        raise ValueError(f"{label}: duplicate antecedent combination")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        rf"{row_name} $\backslash$ {col_name} & " + " & ".join(_in(c) for c in cols) + r"\\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(_in(r) + " & " + " & ".join(_t(cell.get((r, c), "--")) for c in cols) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build() -> str:
    parts = []
    for vs in dict.fromkeys(v for v, *_ in FIS1_RULES):
        sub = [(i, r, o) for v, i, r, o in FIS1_RULES if v == vs]
        parts.append(
            _grid(
                sub,
                "IDL",
                "RTR",
                rf"FIS$_1$ (VS, IDL, RTR $\to$ ULPP), vulnerability score {_in(vs)}.",
                f"tab:s-fis1-{vs}",
            )
        )
    parts.append(
        _grid(
            FIS2_TIL_RULES,
            "TD",
            "RWS",
            r"FIS$_2$ (TD, RWS $\to$ TIL), used by the three-objective model.",
            "tab:s-fis2",
        )
    )
    parts.append(
        _grid(
            FIS2A_TRD_RULES,
            "RCS",
            "PHS",
            r"FIS$_{2a}$ (RCS, PHS $\to$ TRD), four-objective model.",
            "tab:s-fis2a",
        )
    )
    parts.append(
        _grid(
            [(t, "rpd", o) for t, o in FIS2B_RPD_RULES],
            "TD",
            "",
            r"FIS$_{2b}$ (TD $\to$ RPD), four-objective model.",
            "tab:s-fis2b",
        )
    )
    for cor in dict.fromkeys(c for c, *_ in FIS3_RULES):
        sub = [(r, t, o) for c, r, t, o in FIS3_RULES if c == cor]
        parts.append(
            _grid(
                sub,
                "RDR",
                "TD",
                rf"FIS$_3$ (COR, RDR, TD $\to$ CAIL), centre occupancy rate {_in(cor)}.",
                f"tab:s-fis3-{cor}",
            )
        )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
