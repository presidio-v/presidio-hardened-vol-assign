"""Supplementary S1 tables: headers as published in the corrected Systems supplement."""

from __future__ import annotations

from experiments.make_supplementary_rules import build


def _header(tex: str, label: str) -> str:
    block = tex.split(rf"\label{{{label}}}", 1)[1]
    return block.split(r"\toprule", 1)[1].split(r"\midrule", 1)[0].strip()


def test_single_input_table_has_plain_headers():
    assert _header(build(), "tab:s-fis2b") == r"TD & RPD\\"


def test_two_input_table_keeps_corner_label():
    assert _header(build(), "tab:s-fis2") == r"TD $\backslash$ RWS & low & moderate & high\\"
