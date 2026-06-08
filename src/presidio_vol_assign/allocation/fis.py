"""Fuzzy Inference Systems for the allocation model.

Implements four Mamdani FIS:
    FIS1   — (VS, IDL, RTR)              → ULPP   (Resourcefulness)
    FIS2   — (TD, RWS)                   → TIL    (legacy 3-obj only)
    FIS2a  — (RCS, PHS)                  → TRD    (Robustness, 4-obj)
    FIS2b  — (TD)                        → RPD    (Rapidity, 4-obj)
    FIS3   — (COR, RDR, TD)              → CAIL   (Redundancy)

Membership functions and rule bases match Rabiei, Arias-Aranda, Stantchev
(ATRes 2026, in press) Tables 3–6 for FIS1, FIS2 (legacy), and FIS3. The
new FIS2a and FIS2b are introduced for the MDPI Applied Sciences extended
version (4-objective formulation, 4R-mapped).

Two derived helpers compute the inputs the paper expresses in closed form:

    compute_vs(person, weights) → VS_j ∈ [0, 1]            (Eq. 2)
    compute_rws(travel, weights) → RWS_{j,i} ∈ [0, 1]      (Eq. 5)

All evaluation functions return values in [0, 100] (the output universe of
the FIS), matching the ATRes index ranges. Each call creates a fresh
`ControlSystemSimulation` because that class is stateful and not
thread-safe.

Fail-safe: if a simulation produces no output (zero-firing edge case from
extreme inputs), evaluators return 50.0 — the neutral mid-point of the
output universe.

Rule-base override (for H3a sensitivity analysis):
Rule tables are exposed as module-level constants (e.g. `FIS1_RULES`).
`build_fis_with_drops(name, drop_indices)` rebuilds a named FIS with a
subset of rules removed by 0-indexed position. The `fis_overrides({...})`
context manager temporarily swaps in overridden systems for the duration
of an enclosed `solve()` call; outside the context, evaluators fall back
to the module-default systems built once at import.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from presidio_vol_assign.allocation.models import Person, TravelInfo, Weights

# ---------------------------------------------------------------------------
# Universe arrays (linspace avoids float-step drift from arange)
# ---------------------------------------------------------------------------

_U_VS = np.linspace(0, 1, 101)  # vulnerability score [0, 1]
_U_IDL = np.linspace(0, 100, 101)  # infrastructure damage level [0, 100]
_U_RTR = np.linspace(0, 48, 101)  # resource time remaining [0, 48 hours]
_U_TD = np.linspace(0, 180, 181)  # travel duration [0, 180 minutes]
_U_RWS = np.linspace(0, 1, 101)  # roadworthiness score [0, 1]
_U_RCS = np.linspace(0, 1, 101)  # road condition score [0, 1]
_U_PHS = np.linspace(0, 1, 101)  # possible hazard score [0, 1]
_U_COR = np.linspace(0, 100, 101)  # center occupancy rate [0, 100]
_U_RDR = np.linspace(0, 100, 101)  # resource depletion rate [0, 100]
_U_OUT = np.linspace(0, 100, 101)  # ULPP / TIL / TRD / RPD / CAIL [0, 100]

_EPS = 1e-4


# ---------------------------------------------------------------------------
# Shared membership-function builders
# ---------------------------------------------------------------------------


def _vs_lmh(var: ctrl.Antecedent) -> None:
    """Vulnerability Score: Low / Medium / High."""
    var["low"] = fuzz.trapmf(var.universe, [0, 0, 0.25, 0.5])
    var["medium"] = fuzz.trimf(var.universe, [0.25, 0.5, 0.75])
    var["high"] = fuzz.trapmf(var.universe, [0.5, 0.75, 1.0, 1.0])


def _idl_levels(var: ctrl.Antecedent) -> None:
    """Infrastructure Damage Level: Minor / Moderate / Severe."""
    var["minor"] = fuzz.trapmf(var.universe, [0, 0, 25, 50])
    var["moderate"] = fuzz.trimf(var.universe, [25, 50, 75])
    var["severe"] = fuzz.trapmf(var.universe, [50, 75, 100, 100])


def _rtr_levels(var: ctrl.Antecedent) -> None:
    """Resource Time Remaining (hours): Short / Moderate / Long."""
    var["short"] = fuzz.trapmf(var.universe, [0, 0, 12, 24])
    var["moderate"] = fuzz.trimf(var.universe, [12, 24, 36])
    var["long"] = fuzz.trapmf(var.universe, [24, 36, 48, 48])


def _td_levels(var: ctrl.Antecedent) -> None:
    """Travel Duration (minutes): Short / Moderate / Long."""
    var["short"] = fuzz.trapmf(var.universe, [0, 0, 30, 60])
    var["moderate"] = fuzz.trimf(var.universe, [30, 60, 90])
    var["long"] = fuzz.trapmf(var.universe, [60, 90, 180, 180])


def _rws_lmh(var: ctrl.Antecedent) -> None:
    """Roadworthiness Score: Low / Moderate / High."""
    var["low"] = fuzz.trapmf(var.universe, [0, 0, 0.25, 0.5])
    var["moderate"] = fuzz.trimf(var.universe, [0.25, 0.5, 0.75])
    var["high"] = fuzz.trapmf(var.universe, [0.5, 0.75, 1.0, 1.0])


def _rcs_lmh(var: ctrl.Antecedent) -> None:
    """Road Condition Score: Low (clear) / Moderate / High (blocked)."""
    var["low"] = fuzz.trapmf(var.universe, [0, 0, 0.2, 0.5])
    var["moderate"] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
    var["high"] = fuzz.trapmf(var.universe, [0.5, 0.8, 1.0, 1.0])


def _phs_lmh(var: ctrl.Antecedent) -> None:
    """Possible Hazard Score: Low / Moderate / High (3-level coarsening
    of the underlying 5-level discrete PHS for FIS2a tractability)."""
    var["low"] = fuzz.trapmf(var.universe, [0, 0, 0.2, 0.4])
    var["moderate"] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
    var["high"] = fuzz.trapmf(var.universe, [0.6, 0.8, 1.0, 1.0])


def _cor_levels(var: ctrl.Antecedent) -> None:
    """Center Occupancy Rate: Low / Moderate / High."""
    var["low"] = fuzz.trapmf(var.universe, [0, 0, 25, 50])
    var["moderate"] = fuzz.trimf(var.universe, [25, 50, 75])
    var["high"] = fuzz.trapmf(var.universe, [50, 75, 100, 100])


def _rdr_levels(var: ctrl.Antecedent) -> None:
    """Resource Depletion Rate: Slow / Moderate / Rapid."""
    var["slow"] = fuzz.trapmf(var.universe, [0, 0, 25, 50])
    var["moderate"] = fuzz.trimf(var.universe, [25, 50, 75])
    var["rapid"] = fuzz.trapmf(var.universe, [50, 75, 100, 100])


def _out_5level(var: ctrl.Consequent) -> None:
    """Output universe: VeryLow / Low / Moderate / High / VeryHigh on [0, 100]."""
    var["very_low"] = fuzz.trapmf(var.universe, [0, 0, 10, 30])
    var["low"] = fuzz.trimf(var.universe, [10, 30, 50])
    var["moderate"] = fuzz.trimf(var.universe, [30, 50, 70])
    var["high"] = fuzz.trimf(var.universe, [50, 70, 90])
    var["very_high"] = fuzz.trapmf(var.universe, [70, 90, 100, 100])


# ---------------------------------------------------------------------------
# Rule tables — module-level constants exposed for sensitivity analysis
# ---------------------------------------------------------------------------

# FIS1 — (VS, IDL, RTR) → ULPP, ATRes Table 4 (27 rules)
FIS1_RULES: tuple[tuple[str, str, str, str], ...] = (
    # (vs, idl, rtr, ulpp)
    ("low", "minor", "short", "moderate"),
    ("low", "minor", "moderate", "low"),
    ("low", "minor", "long", "very_low"),
    ("low", "moderate", "short", "high"),
    ("low", "moderate", "moderate", "moderate"),
    ("low", "moderate", "long", "low"),
    ("low", "severe", "short", "very_high"),
    ("low", "severe", "moderate", "high"),
    ("low", "severe", "long", "moderate"),
    ("medium", "minor", "short", "high"),
    ("medium", "minor", "moderate", "moderate"),
    ("medium", "minor", "long", "low"),
    ("medium", "moderate", "short", "very_high"),
    ("medium", "moderate", "moderate", "high"),
    ("medium", "moderate", "long", "moderate"),
    ("medium", "severe", "short", "very_high"),
    ("medium", "severe", "moderate", "very_high"),
    ("medium", "severe", "long", "high"),
    ("high", "minor", "short", "very_high"),
    ("high", "minor", "moderate", "high"),
    ("high", "minor", "long", "moderate"),
    ("high", "moderate", "short", "very_high"),
    ("high", "moderate", "moderate", "very_high"),
    ("high", "moderate", "long", "high"),
    ("high", "severe", "short", "very_high"),
    ("high", "severe", "moderate", "very_high"),
    ("high", "severe", "long", "very_high"),
)

# FIS2 (legacy) — (TD, RWS) → TIL, ATRes Table 5 (9 rules)
FIS2_TIL_RULES: tuple[tuple[str, str, str], ...] = (
    # (td, rws, til)
    ("short", "low", "moderate"),
    ("short", "moderate", "low"),
    ("short", "high", "very_low"),
    ("moderate", "low", "high"),
    ("moderate", "moderate", "moderate"),
    ("moderate", "high", "low"),
    ("long", "low", "very_high"),
    ("long", "moderate", "high"),
    ("long", "high", "moderate"),
)

# FIS2a — (RCS, PHS) → TRD (9 rules), MDPI extension
FIS2A_TRD_RULES: tuple[tuple[str, str, str], ...] = (
    # (rcs, phs, trd)
    ("low", "low", "very_low"),
    ("low", "moderate", "low"),
    ("low", "high", "moderate"),
    ("moderate", "low", "low"),
    ("moderate", "moderate", "moderate"),
    ("moderate", "high", "high"),
    ("high", "low", "moderate"),
    ("high", "moderate", "high"),
    ("high", "high", "very_high"),
)

# FIS2b — (TD) → RPD (3 rules), MDPI extension
FIS2B_RPD_RULES: tuple[tuple[str, str], ...] = (
    # (td, rpd)
    ("short", "low"),
    ("moderate", "moderate"),
    ("long", "high"),
)

# FIS3 — (COR, RDR, TD) → CAIL, ATRes Table 6 (27 rules)
FIS3_RULES: tuple[tuple[str, str, str, str], ...] = (
    # (cor, rdr, td, cail)
    ("low", "slow", "short", "very_low"),
    ("low", "slow", "moderate", "low"),
    ("low", "slow", "long", "moderate"),
    ("low", "moderate", "short", "low"),
    ("low", "moderate", "moderate", "moderate"),
    ("low", "moderate", "long", "high"),
    ("low", "rapid", "short", "moderate"),
    ("low", "rapid", "moderate", "high"),
    ("low", "rapid", "long", "high"),
    ("moderate", "slow", "short", "low"),
    ("moderate", "slow", "moderate", "moderate"),
    ("moderate", "slow", "long", "high"),
    ("moderate", "moderate", "short", "moderate"),
    ("moderate", "moderate", "moderate", "moderate"),
    ("moderate", "moderate", "long", "high"),
    ("moderate", "rapid", "short", "high"),
    ("moderate", "rapid", "moderate", "high"),
    ("moderate", "rapid", "long", "very_high"),
    ("high", "slow", "short", "moderate"),
    ("high", "slow", "moderate", "high"),
    ("high", "slow", "long", "high"),
    ("high", "moderate", "short", "high"),
    ("high", "moderate", "moderate", "high"),
    ("high", "moderate", "long", "very_high"),
    ("high", "rapid", "short", "very_high"),
    ("high", "rapid", "moderate", "very_high"),
    ("high", "rapid", "long", "very_high"),
)


# Rule-base sizes — used by validators when checking sensitivity-spec drop indices
RULE_COUNTS: dict[str, int] = {
    "fis1": len(FIS1_RULES),
    "fis2_til": len(FIS2_TIL_RULES),
    "fis2a_trd": len(FIS2A_TRD_RULES),
    "fis2b_rpd": len(FIS2B_RPD_RULES),
    "fis3": len(FIS3_RULES),
}

VALID_FIS_NAMES: frozenset[str] = frozenset(RULE_COUNTS.keys())


def _filter_rules(rules: Iterable, drop: set[int] | None) -> list:
    """Return rules with the specified 0-indexed positions removed."""
    if not drop:
        return list(rules)
    return [r for i, r in enumerate(rules) if i not in drop]


# ---------------------------------------------------------------------------
# FIS1 builder — (VS, IDL, RTR) → ULPP   (Resourcefulness)
# ---------------------------------------------------------------------------


def _build_fis1(drop_indices: set[int] | None = None) -> ctrl.ControlSystem:
    vs = ctrl.Antecedent(_U_VS, "vs_fis1")
    idl = ctrl.Antecedent(_U_IDL, "idl_fis1")
    rtr = ctrl.Antecedent(_U_RTR, "rtr_fis1")
    ulpp = ctrl.Consequent(_U_OUT, "ulpp")

    _vs_lmh(vs)
    _idl_levels(idl)
    _rtr_levels(rtr)
    _out_5level(ulpp)

    table = _filter_rules(FIS1_RULES, drop_indices)
    rules = [ctrl.Rule(vs[v] & idl[i] & rtr[r], ulpp[o]) for (v, i, r, o) in table]
    return ctrl.ControlSystem(rules)


# ---------------------------------------------------------------------------
# FIS2 (legacy) builder — (TD, RWS) → TIL   (3-obj mode only)
# ---------------------------------------------------------------------------


def _build_fis2_til(drop_indices: set[int] | None = None) -> ctrl.ControlSystem:
    td = ctrl.Antecedent(_U_TD, "td_fis2")
    rws = ctrl.Antecedent(_U_RWS, "rws_fis2")
    til = ctrl.Consequent(_U_OUT, "til")

    _td_levels(td)
    _rws_lmh(rws)
    _out_5level(til)

    table = _filter_rules(FIS2_TIL_RULES, drop_indices)
    rules = [ctrl.Rule(td[t] & rws[r], til[o]) for (t, r, o) in table]
    return ctrl.ControlSystem(rules)


# ---------------------------------------------------------------------------
# FIS2a builder — (RCS, PHS) → TRD   (4-obj mode, Robustness)
# ---------------------------------------------------------------------------


def _build_fis2a_trd(drop_indices: set[int] | None = None) -> ctrl.ControlSystem:
    rcs = ctrl.Antecedent(_U_RCS, "rcs_fis2a")
    phs = ctrl.Antecedent(_U_PHS, "phs_fis2a")
    trd = ctrl.Consequent(_U_OUT, "trd")

    _rcs_lmh(rcs)
    _phs_lmh(phs)
    _out_5level(trd)

    table = _filter_rules(FIS2A_TRD_RULES, drop_indices)
    rules = [ctrl.Rule(rcs[c] & phs[h], trd[o]) for (c, h, o) in table]
    return ctrl.ControlSystem(rules)


# ---------------------------------------------------------------------------
# FIS2b builder — (TD) → RPD   (4-obj mode, Rapidity)
# ---------------------------------------------------------------------------


def _build_fis2b_rpd(drop_indices: set[int] | None = None) -> ctrl.ControlSystem:
    td = ctrl.Antecedent(_U_TD, "td_fis2b")
    rpd = ctrl.Consequent(_U_OUT, "rpd")

    _td_levels(td)
    _out_5level(rpd)

    table = _filter_rules(FIS2B_RPD_RULES, drop_indices)
    rules = [ctrl.Rule(td[t], rpd[o]) for (t, o) in table]
    return ctrl.ControlSystem(rules)


# ---------------------------------------------------------------------------
# FIS3 builder — (COR, RDR, TD) → CAIL   (Redundancy)
# ---------------------------------------------------------------------------


def _build_fis3(drop_indices: set[int] | None = None) -> ctrl.ControlSystem:
    cor = ctrl.Antecedent(_U_COR, "cor_fis3")
    rdr = ctrl.Antecedent(_U_RDR, "rdr_fis3")
    td = ctrl.Antecedent(_U_TD, "td_fis3")
    cail = ctrl.Consequent(_U_OUT, "cail")

    _cor_levels(cor)
    _rdr_levels(rdr)
    _td_levels(td)
    _out_5level(cail)

    table = _filter_rules(FIS3_RULES, drop_indices)
    rules = [ctrl.Rule(cor[c] & rdr[r] & td[t], cail[o]) for (c, r, t, o) in table]
    return ctrl.ControlSystem(rules)


# ---------------------------------------------------------------------------
# Builder registry + module-level default systems (built once at import)
# ---------------------------------------------------------------------------

_BUILDERS = {
    "fis1": _build_fis1,
    "fis2_til": _build_fis2_til,
    "fis2a_trd": _build_fis2a_trd,
    "fis2b_rpd": _build_fis2b_rpd,
    "fis3": _build_fis3,
}

_FIS1_SYSTEM = _build_fis1()
_FIS2_TIL_SYSTEM = _build_fis2_til()
_FIS2A_TRD_SYSTEM = _build_fis2a_trd()
_FIS2B_RPD_SYSTEM = _build_fis2b_rpd()
_FIS3_SYSTEM = _build_fis3()

_DEFAULTS: dict[str, ctrl.ControlSystem] = {
    "fis1": _FIS1_SYSTEM,
    "fis2_til": _FIS2_TIL_SYSTEM,
    "fis2a_trd": _FIS2A_TRD_SYSTEM,
    "fis2b_rpd": _FIS2B_RPD_SYSTEM,
    "fis3": _FIS3_SYSTEM,
}


# ---------------------------------------------------------------------------
# Override registry
#
# NOTE: module-level mutable state. Use the `fis_overrides()` context
# manager rather than raw set/clear when possible — it guarantees the
# overrides are cleared even if the wrapped block raises.
# ---------------------------------------------------------------------------

_OVERRIDES: dict[str, ctrl.ControlSystem] = {}


def build_fis_with_drops(name: str, drop_indices: Iterable[int]) -> ctrl.ControlSystem:
    """Rebuild the named FIS with the given 0-indexed rule positions removed.

    Validates `name` against `VALID_FIS_NAMES` and each index against the
    rule-base size in `RULE_COUNTS`. Returns a fresh `ControlSystem`; the
    caller decides whether to install it via `set_fis_overrides`.

    Raises:
        ValueError: Unknown FIS name or out-of-range rule index.
    """
    if name not in _BUILDERS:
        raise ValueError(f"Unknown FIS name {name!r}; expected one of {sorted(VALID_FIS_NAMES)}")
    drops = {int(i) for i in drop_indices}
    n = RULE_COUNTS[name]
    bad = [i for i in drops if not (0 <= i < n)]
    if bad:
        raise ValueError(f"Rule indices {sorted(bad)} out of range for {name} (0..{n - 1})")
    if len(drops) >= n:
        raise ValueError(f"Cannot drop all {n} rules from {name}; at least one must remain")
    return _BUILDERS[name](drops)


def set_fis_overrides(systems: dict[str, ctrl.ControlSystem]) -> None:
    """Install a mapping of FIS name → overridden `ControlSystem`.

    Eval functions consult this mapping first; absent entries fall back to
    the module-default system. Replaces any prior overrides.
    """
    _OVERRIDES.clear()
    _OVERRIDES.update(systems)


def clear_fis_overrides() -> None:
    """Remove all FIS overrides; eval functions revert to defaults."""
    _OVERRIDES.clear()


@contextlib.contextmanager
def fis_overrides(spec: dict[str, list[int]] | dict[str, Iterable[int]]) -> Iterator[None]:
    """Context manager: temporarily apply FIS rule drops.

    `spec` maps FIS name → list of 0-indexed rule positions to remove.
    On entry, the named FIS systems are rebuilt and installed; on exit
    (including via exception), all overrides are cleared.

    Example:

        with fis_overrides({"fis1": [3, 5], "fis3": [12]}):
            front = solve(problem, config)
    """
    new_systems = {name: build_fis_with_drops(name, drops) for name, drops in spec.items()}
    previous = dict(_OVERRIDES)
    set_fis_overrides(new_systems)
    try:
        yield
    finally:
        # Restore exactly what was there before — supports nested contexts.
        _OVERRIDES.clear()
        _OVERRIDES.update(previous)


def load_fis_rules_spec(path: Path) -> dict[str, list[int]]:
    """Read a JSON FIS-rules-drop spec from disk and return it as a dict.

    Expected JSON shape:

        {
          "fis1":      [3, 5],
          "fis2a_trd": [0, 8],
          "fis3":      [12]
        }

    The structure is validated (keys must be in `VALID_FIS_NAMES`, values
    must be lists of non-negative ints). Index ranges are not validated
    here — that happens at `build_fis_with_drops` time so the JSON loader
    can be reused for partial specs in interactive flows.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On malformed JSON or schema violation.
    """
    if not path.exists():
        raise FileNotFoundError(f"FIS-rules spec not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in FIS-rules spec {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"FIS-rules spec must be a JSON object, got {type(data).__name__}")
    out: dict[str, list[int]] = {}
    for name, indices in data.items():
        if name not in VALID_FIS_NAMES:
            raise ValueError(
                f"Unknown FIS name {name!r} in spec; expected one of {sorted(VALID_FIS_NAMES)}"
            )
        if not isinstance(indices, list) or not all(isinstance(i, int) and i >= 0 for i in indices):
            raise ValueError(
                f"Drop indices for {name!r} must be a list of non-negative ints, got {indices!r}"
            )
        out[name] = list(indices)
    return out


def _active(name: str) -> ctrl.ControlSystem:
    """Return the overridden system for `name`, or the module default."""
    return _OVERRIDES.get(name, _DEFAULTS[name])


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def _run_sim(system: ctrl.ControlSystem, inputs: dict[str, float]) -> float:
    """Run a fresh simulation; return the single consequent's defuzzified value.

    Returns 50.0 (neutral mid-point of [0, 100]) on zero-firing failure.
    """
    sim = ctrl.ControlSystemSimulation(system)
    for name, value in inputs.items():
        sim.input[name] = value
    try:
        sim.compute()
        key = next(iter(system.consequents)).label
        return float(np.clip(float(sim.output[key]), 0.0, 100.0))
    except Exception:  # noqa: BLE001
        return 50.0


# ---------------------------------------------------------------------------
# Derived input helpers
# ---------------------------------------------------------------------------


def compute_vs(person: Person, weights: Weights) -> float:
    """Vulnerability Score VS_j ∈ [0, 1] per ATRes Eq. (2).

    VS_j = (WAS·AS_j + WDS·DS_j + WIL·IL_j + WLS·LS_j) /
           (WAS + WDS + WIL + WLS)
    """
    num = (
        weights.was * person.age_score
        + weights.wds * person.disability_status.score
        + weights.wil * person.injury_level.score
        + weights.wls * person.living_status.score
    )
    den = weights.was + weights.wds + weights.wil + weights.wls
    if den == 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def compute_rws(travel: TravelInfo, weights: Weights) -> float:
    """Roadworthiness Score RWS_{j,i} ∈ [0, 1] (ATRes Eq. (5), sign-corrected).

    ATRes Eq. (5) as written takes the weighted average of RCS_{j,i} and
    PHS_{j,i}, but ATRes Table 1 defines those score variables with the
    convention that *higher* values mean *worse* conditions (Blocked=1.0,
    Extreme=1.0). The same paper's Table 3 then labels RWS levels with the
    opposite convention (High RWS = "safe and efficient routes"), and its
    Table 5 FIS2 rules require *high RWS to imply low TIL*.

    Reading the formula literally produces a danger score, not a
    roadworthiness score. We invert to keep the variable consistent with
    the rule semantics and the linguistic labels:

        RWS_{j,i} = 1 - (WRC·RCS + WPH·PHS) / (WRC + WPH)

    This makes RWS=1 a perfectly safe and clear route, RWS=0 an impassable
    one — matching Table 3 and Table 5. The MDPI extended paper notes this
    correction and uses the same form throughout.
    """
    den = weights.wrc + weights.wph
    if den == 0:
        return 1.0
    danger = (
        weights.wrc * travel.road_condition.score + weights.wph * travel.possible_hazard.score
    ) / den
    return float(np.clip(1.0 - danger, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public evaluation functions — consult the override registry per call
# ---------------------------------------------------------------------------


def evaluate_fis1_ulpp(vs: float, idl: float, rtr: float) -> float:
    """ATRes Eq. (1): FIS1 — Unfairness Level in People Prioritization."""
    return _run_sim(
        _active("fis1"),
        {
            "vs_fis1": float(np.clip(vs, _EPS, 1 - _EPS)),
            "idl_fis1": float(np.clip(idl, _EPS, 100 - _EPS)),
            "rtr_fis1": float(np.clip(rtr, _EPS, 48 - _EPS)),
        },
    )


def evaluate_fis2_til(td: float, rws: float) -> float:
    """ATRes Eq. (4): FIS2 — Transportation Infeasibility Level (legacy 3-obj)."""
    return _run_sim(
        _active("fis2_til"),
        {
            "td_fis2": float(np.clip(td, _EPS, 180 - _EPS)),
            "rws_fis2": float(np.clip(rws, _EPS, 1 - _EPS)),
        },
    )


def evaluate_fis2a_trd(rcs: float, phs: float) -> float:
    """MDPI extension: FIS2a — Transport Robustness Deficit (Robustness)."""
    return _run_sim(
        _active("fis2a_trd"),
        {
            "rcs_fis2a": float(np.clip(rcs, _EPS, 1 - _EPS)),
            "phs_fis2a": float(np.clip(phs, _EPS, 1 - _EPS)),
        },
    )


def evaluate_fis2b_rpd(td: float) -> float:
    """MDPI extension: FIS2b — Rapidity Deficit (Rapidity)."""
    return _run_sim(
        _active("fis2b_rpd"),
        {
            "td_fis2b": float(np.clip(td, _EPS, 180 - _EPS)),
        },
    )


def evaluate_fis3_cail(cor: float, rdr: float, td: float) -> float:
    """ATRes Eq. (8): FIS3 — Center Allocation Imbalance Level (Redundancy)."""
    return _run_sim(
        _active("fis3"),
        {
            "cor_fis3": float(np.clip(cor, _EPS, 100 - _EPS)),
            "rdr_fis3": float(np.clip(rdr, _EPS, 100 - _EPS)),
            "td_fis3": float(np.clip(td, _EPS, 180 - _EPS)),
        },
    )
