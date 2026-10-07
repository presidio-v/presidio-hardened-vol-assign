"""Compiled Mamdani evaluator that reproduces scikit-fuzzy's ``ControlSystemSimulation``.

A fresh skfuzzy simulation costs ~2.6 ms per call. That is fine for a static FIS cache
(one call per person or pair) but not for the load-coupled model, where CAIL must be
tabulated per (person, centre, load level), nor for a turbulence study that rebuilds the
cache for every perturbation. This module compiles one of the project's rule tables into
plain numpy and replays skfuzzy's arithmetic step for step:

1. fuzzify each crisp input by linear interpolation of the term's sampled MF;
2. fire each rule with ``min`` over its antecedents (skfuzzy's default AND);
3. accumulate rules sharing a consequent term with ``max``;
4. upsample the output universe at every point where a term's MF meets its cut level
   (skfuzzy's ``find_memberships``), clip each term at its cut, aggregate with ``max``;
5. defuzzify by skfuzzy's piecewise-linear centroid.

It is a pure performance substitute: ``tests/allocation/test_fast_fis.py`` pins it to the
skfuzzy result over a dense input sample. Rule-drop overrides (``fis_overrides``) are not
supported here — callers that need them must stay on the skfuzzy path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np
from skfuzzy import control as ctrl

from presidio_vol_assign.allocation import fis as _fis
from presidio_vol_assign.allocation.models import AllocationConfig, AllocationProblem
from presidio_vol_assign.allocation.solvers import FISCache

_NEUTRAL = 50.0  # matches fis._run_sim's zero-firing fallback


@dataclass(frozen=True)
class _Input:
    universe: np.ndarray
    lo: float
    hi: float
    terms: dict[str, np.ndarray]


@dataclass(frozen=True)
class CompiledMamdani:
    """A Mamdani FIS reduced to sampled MFs and an index-based rule table."""

    inputs: tuple[_Input, ...]
    out_universe: np.ndarray
    out_terms: tuple[str, ...]
    out_mfs: np.ndarray  # (n_out_terms, len(out_universe))
    rule_antecedents: tuple[tuple[str, ...], ...]
    rule_consequent: np.ndarray  # index into out_terms, one per rule

    def __call__(self, *values: float) -> float:
        if len(values) != len(self.inputs):
            raise ValueError(f"expected {len(self.inputs)} inputs, got {len(values)}")
        degrees = []
        for spec, value in zip(self.inputs, values, strict=True):
            x = float(np.clip(value, spec.lo, spec.hi))
            degrees.append(
                {name: float(np.interp(x, spec.universe, mf)) for name, mf in spec.terms.items()}
            )
        cuts = np.full(len(self.out_terms), np.nan)  # NaN = term fired by no rule
        for labels, k in zip(self.rule_antecedents, self.rule_consequent, strict=True):
            strength = min(d[label] for d, label in zip(degrees, labels, strict=True))
            cuts[k] = strength if np.isnan(cuts[k]) else max(cuts[k], strength)
        return _defuzz(self.out_universe, self.out_mfs, cuts)

    def batch(self, *values: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """Evaluate many input tuples at once; same arithmetic as ``__call__``.

        Arguments are equal-length arrays, one per input. Upsampling is done per base
        segment: within one segment every term MF is linear, so skfuzzy's global cut-point
        insertion is equivalent to inserting each term's crossing inside that segment.
        """
        if len(values) != len(self.inputs):
            raise ValueError(f"expected {len(self.inputs)} inputs, got {len(values)}")
        arrays = [np.asarray(v, dtype=float).ravel() for v in values]
        n = arrays[0].size
        if any(a.size != n for a in arrays):
            raise ValueError("all input arrays must have the same length")
        out = np.empty(n)
        for start in range(0, n, chunk):
            sl = slice(start, start + chunk)
            out[sl] = self._batch_chunk([a[sl] for a in arrays])
        return out

    def _batch_chunk(self, arrays: list[np.ndarray]) -> np.ndarray:
        n = arrays[0].size
        degrees = []
        for spec, x in zip(self.inputs, arrays, strict=True):
            xc = np.clip(x, spec.lo, spec.hi)
            degrees.append({t: np.interp(xc, spec.universe, mf) for t, mf in spec.terms.items()})
        n_terms = len(self.out_terms)
        cuts = np.full((n, n_terms), np.nan)
        for labels, k in zip(self.rule_antecedents, self.rule_consequent, strict=True):
            strength = np.minimum.reduce(
                [d[label] for d, label in zip(degrees, labels, strict=True)]
            )
            cuts[:, k] = np.where(np.isnan(cuts[:, k]), strength, np.fmax(cuts[:, k], strength))
        return _defuzz_batch(self.out_universe, self.out_mfs, cuts)


def _cut_points(x: np.ndarray, mf: np.ndarray, y: float) -> np.ndarray:
    """skfuzzy ``_interp_universe_fast``: universe points where ``mf`` crosses level ``y``."""
    idx = np.where(np.diff(mf > y))[0] if y == 0.0 else np.where(np.diff(mf >= y))[0]
    return x[idx] + (y - mf[idx]) * (x[idx + 1] - x[idx]) / (mf[idx + 1] - mf[idx])


def _defuzz(universe: np.ndarray, mfs: np.ndarray, cuts: np.ndarray) -> float:
    active = ~np.isnan(cuts)
    if not active.any():
        return _NEUTRAL
    extra = [_cut_points(universe, mfs[k], cuts[k]) for k in np.flatnonzero(active)]
    x = np.union1d(universe, np.concatenate(extra))
    out = np.zeros_like(x)
    for k in np.flatnonzero(active):
        np.maximum(out, np.minimum(cuts[k], np.interp(x, universe, mfs[k])), out=out)
    value = _centroid(x, out)
    return _NEUTRAL if value is None else float(np.clip(value, 0.0, 100.0))


def _defuzz_batch(universe: np.ndarray, mfs: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Vectorised ``_defuzz`` over rows of ``cuts`` (shape (N, n_terms))."""
    x0, x1 = universe[:-1], universe[1:]  # (S,)
    m0, m1 = mfs[:, :-1], mfs[:, 1:]  # (K, S)
    active = ~np.isnan(cuts)  # (N, K)
    c = np.where(active, cuts, 0.0)[:, :, None]  # (N, K, 1)
    # skfuzzy cut-crossing rule: strict ">" for a zero cut, ">=" otherwise
    above0 = np.where(c == 0.0, m0[None] > c, m0[None] >= c)
    above1 = np.where(c == 0.0, m1[None] > c, m1[None] >= c)
    crosses = (above0 != above1) & active[:, :, None]  # (N, K, S)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x0 + (c - m0[None]) * (x1 - x0) / (m1 - m0)[None]
    xc = np.where(crosses, xc, np.nan)  # (N, K, S)
    pts = np.concatenate(
        [
            np.broadcast_to(x0, (cuts.shape[0], 1, x0.size)),
            xc,
            np.broadcast_to(x1, (cuts.shape[0], 1, x1.size)),
        ],
        axis=1,
    )  # (N, K+2, S)
    pts = np.sort(pts, axis=1)  # NaN sorts last
    pts = np.where(np.isnan(pts), x1, pts)  # padding collapses to zero-width pieces
    # every term MF is linear on its base segment
    frac = (pts - x0) / (x1 - x0)  # (N, K+2, S)
    y = np.zeros_like(pts)
    for k in range(mfs.shape[0]):
        mf_k = m0[k] + frac * (m1[k] - m0[k])
        clipped = np.minimum(cuts[:, k][:, None, None], mf_k)
        y = np.where(active[:, k][:, None, None], np.maximum(y, clipped), y)
    xa, xb = pts[:, :-1, :], pts[:, 1:, :]
    ya, yb = y[:, :-1, :], y[:, 1:, :]
    w = xb - xa
    keep = ~(((ya == 0.0) & (yb == 0.0)) | (w == 0.0))
    rect = keep & (ya == yb)
    rise = keep & ~rect & (ya == 0.0)
    fall = keep & ~rect & ~rise & (yb == 0.0)
    trap = keep & ~(rect | rise | fall)
    with np.errstate(divide="ignore", invalid="ignore"):
        moment = np.select(
            [rect, rise, fall, trap],
            [
                0.5 * (xa + xb),
                2.0 / 3.0 * w + xa,
                1.0 / 3.0 * w + xa,
                (2.0 / 3.0 * w * (yb + 0.5 * ya)) / (ya + yb) + xa,
            ],
            0.0,
        )
    area = np.select(
        [rect, rise, fall, trap], [w * ya, 0.5 * w * yb, 0.5 * w * ya, 0.5 * w * (ya + yb)], 0.0
    )
    total = area.sum(axis=(1, 2))
    value = (moment * area).sum(axis=(1, 2)) / np.fmax(total, np.finfo(float).eps)
    ok = active.any(axis=1) & (total > 0.0)
    return np.where(ok, np.clip(value, 0.0, 100.0), _NEUTRAL)


def _centroid(x: np.ndarray, y: np.ndarray) -> float | None:
    """skfuzzy's exact piecewise-linear centroid; None when the area is empty."""
    x1, x2, y1, y2 = x[:-1], x[1:], y[:-1], y[1:]
    width = x2 - x1
    keep = ~(((y1 == 0.0) & (y2 == 0.0)) | (width == 0.0))
    x1, x2, y1, y2, width = x1[keep], x2[keep], y1[keep], y2[keep], width[keep]
    if x1.size == 0:
        return None
    rect = y1 == y2
    rise = (y1 == 0.0) & ~rect
    fall = (y2 == 0.0) & ~rect & ~rise
    trap = ~(rect | rise | fall)
    moment = np.empty_like(x1)
    area = np.empty_like(x1)
    moment[rect] = 0.5 * (x1[rect] + x2[rect])
    area[rect] = width[rect] * y1[rect]
    moment[rise] = 2.0 / 3.0 * width[rise] + x1[rise]
    area[rise] = 0.5 * width[rise] * y2[rise]
    moment[fall] = 1.0 / 3.0 * width[fall] + x1[fall]
    area[fall] = 0.5 * width[fall] * y1[fall]
    t = trap
    moment[t] = (2.0 / 3.0 * width[t] * (y2[t] + 0.5 * y1[t])) / (y1[t] + y2[t]) + x1[t]
    area[t] = 0.5 * width[t] * (y1[t] + y2[t])
    total = float(area.sum())
    if total <= 0.0:
        return None
    return float((moment * area).sum() / max(total, np.finfo(float).eps))


def compile_mamdani(
    antecedents: Sequence[ctrl.Antecedent],
    consequent: ctrl.Consequent,
    table: Sequence[tuple[str, ...]],
    eps: float = _fis._EPS,
) -> CompiledMamdani:
    """Compile a rule table (antecedent labels..., consequent label) into numpy form.

    Inputs are clipped to ``[min + eps, max - eps]`` exactly as the ``evaluate_fis*``
    wrappers do before handing values to skfuzzy.
    """
    inputs = tuple(
        _Input(
            universe=np.asarray(a.universe, dtype=float),
            lo=float(a.universe.min()) + eps,
            hi=float(a.universe.max()) - eps,
            terms={name: np.asarray(t.mf, dtype=float) for name, t in a.terms.items()},
        )
        for a in antecedents
    )
    out_terms = tuple(consequent.terms)
    out_mfs = np.array([consequent.terms[t].mf for t in out_terms], dtype=float)
    n_in = len(inputs)
    for row in table:
        if len(row) != n_in + 1:
            raise ValueError(f"rule {row!r} does not have {n_in} antecedents + 1 consequent")
    return CompiledMamdani(
        inputs=inputs,
        out_universe=np.asarray(consequent.universe, dtype=float),
        out_terms=out_terms,
        out_mfs=out_mfs,
        rule_antecedents=tuple(tuple(row[:n_in]) for row in table),
        rule_consequent=np.array([out_terms.index(row[n_in]) for row in table]),
    )


# (builder, rule table, antecedent labels in rule-table column order)
_SPECS = {
    "fis1": (_fis._build_fis1, _fis.FIS1_RULES, ("vs_fis1", "idl_fis1", "rtr_fis1")),
    "fis2_til": (_fis._build_fis2_til, _fis.FIS2_TIL_RULES, ("td_fis2", "rws_fis2")),
    "fis2a_trd": (_fis._build_fis2a_trd, _fis.FIS2A_TRD_RULES, ("rcs_fis2a", "phs_fis2a")),
    "fis2b_rpd": (_fis._build_fis2b_rpd, _fis.FIS2B_RPD_RULES, ("td_fis2b",)),
    "fis3": (_fis._build_fis3, _fis.FIS3_RULES, ("cor_fis3", "rdr_fis3", "td_fis3")),
}


@cache
def compiled(name: str) -> CompiledMamdani:
    """Compile one of the project's FISs from the same builder and rule table skfuzzy uses."""
    builder, table, labels = _SPECS[name]
    system = builder()
    antecedents = {a.label: a for a in system.antecedents}
    (consequent,) = tuple(system.consequents)
    return compile_mamdani(tuple(antecedents[label] for label in labels), consequent, table)


def compiled_fis3() -> CompiledMamdani:
    """FIS3 (COR, RDR, TD) -> CAIL."""
    return compiled("fis3")


def _require_defaults() -> None:
    for name in _SPECS:
        if _fis._active(name) is not _fis._DEFAULTS[name]:
            raise RuntimeError(
                f"fast FIS does not honour rule-drop overrides ({name} is overridden)"
            )


def evaluate_fis3_cail_fast(cor: float, rdr: float, td: float) -> float:
    """Drop-in fast equivalent of ``fis.evaluate_fis3_cail`` (default rule base only)."""
    _require_defaults()
    return compiled_fis3()(cor, rdr, td)


def precompute_fis_cache_fast(problem: AllocationProblem, config: AllocationConfig) -> FISCache:
    """Batched equivalent of ``solvers.precompute_fis_cache`` (default rule bases only).

    Same inputs, same clipping, same FIS — pinned to the skfuzzy path by tests — but one
    vectorised call per FIS instead of one simulation per person or pair.
    """
    _require_defaults()
    m, n = problem.n_people, problem.n_centers
    weights = config.weights
    vs = np.array([_fis.compute_vs(p, weights) for p in problem.people])
    idl = np.array([p.infrastructure_damage_level for p in problem.people], dtype=float)
    rtr = np.array([p.resource_time_remaining for p in problem.people], dtype=float)
    ulpp = compiled("fis1").batch(vs, idl, rtr)

    person_idx = {p.person_id: j for j, p in enumerate(problem.people)}
    centre_idx = {c.center_id: i for i, c in enumerate(problem.centers)}
    keys = list(problem.travel.items())
    rows = np.array([person_idx[pid] for (pid, _), _ in keys])
    cols = np.array([centre_idx[cid] for (_, cid), _ in keys])
    td = np.array([t.travel_duration for _, t in keys], dtype=float)
    cor = np.array([problem.centers[i].center_occupancy_rate for i in cols], dtype=float)
    rdr = np.array([problem.centers[i].resource_depletion_rate for i in cols], dtype=float)

    def scatter(values: np.ndarray) -> np.ndarray:
        out = np.zeros((m, n))
        out[rows, cols] = values
        return out

    til = trd = rpd = np.zeros((m, n))
    cail = scatter(compiled("fis3").batch(cor, rdr, td))
    if config.objectives == 3:
        rws = np.array([_fis.compute_rws(t, weights) for _, t in keys])
        til = scatter(compiled("fis2_til").batch(td, rws))
    else:
        rcs = np.array([t.road_condition.score for _, t in keys], dtype=float)
        phs = np.array([t.possible_hazard.score for _, t in keys], dtype=float)
        trd = scatter(compiled("fis2a_trd").batch(rcs, phs))
        rpd = scatter(compiled("fis2b_rpd").batch(td))
    return FISCache(ulpp=ulpp, til=til, trd=trd, rpd=rpd, cail=cail)
