# Changelog

All notable changes to `presidio-hardened-vol-assign` are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/), and the
project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] — unreleased

Adds a second optimisation model — humanitarian allocation of affected people to
relief centres — **side by side** with the original ED-staffing model, supporting
the in-preparation paper *"From Methodology to Practice"* (Rabiei, Arias-Aranda &
Stantchev). The ED-staffing model is unchanged and remains the default.

### Added
- **Humanitarian allocation model** (`--model humanitarian`): three new Mamdani
  FIS (Fairness in People Prioritization, Transportation Feasibility, Center
  Allocation Balance), three objectives, integer person→centre encoding with
  soft-capacity overcrowding. Inputs: `people.csv` + `centers.csv`.
- **Domain-adapter architecture**: a generic NSGA-II/NRGA engine (`engine.py`)
  driven by pluggable `Domain` adapters (`domains/`).
- **N-dimensional metrics**: MID/SM/HV generalised to any objective count; SM now
  uses the Schott nearest-neighbour definition; HV uses DEAP's n-D implementation.
- **Reproducibility (REP) metric**: bit-for-bit front signatures and a
  `verify_reproducibility` harness.
- **`pva benchmark`**: deterministic instance generation (humanitarian 5/150 and
  10/300; ed-staffing 5/75 and 10/150), Table-3-style mean±std summaries, and an
  optional REP column (`--check-repro`).
- **`pva show`**: publication-quality Pareto figures (2-D scatter or three
  pairwise projections + 3-D scatter), solvers overlaid. Requires the `viz` extra.
- **`pva allocate-people`**: convenience alias for `assign --model humanitarian`.
- **Canonical NRGA** (`--solver nrga-ranked`): rank-biased roulette-wheel survival
  (Al Jadaan et al., 2008), matching the NRGA literature; the existing `nrga`
  remains as a lightweight uniform-tie-break variant. New `--solver all` runs all
  three solvers.
- **`pva sensitivity`**: sweeps FIS-output perturbations (default ±10 %, ±20 %)
  and reports how NNS/MID/SM/HV shift, to gauge robustness to FIS rule-base
  uncertainty (`Domain.perturb`; `engine.run` accepts a pre-computed cache).
- **Greedy baseline comparator** (`--solver greedy`): a deterministic
  weighted-sum constructive heuristic, swept over the objective simplex, that
  provides a non-evolutionary baseline Pareto front for both models
  (`baselines.py`, `Domain.baseline_population`). `pva benchmark --baseline`
  adds it as a `greedy` row so the framework can be measured against an
  existing-style allocation method rather than only NSGA-II vs NRGA.
- **Wilcoxon rank-sum HV testing** (`stats.py`): `pva benchmark` now compares
  per-instance hypervolume distributions between solvers (vs. the greedy
  baseline when present) with the Wilcoxon rank-sum test, prints a significance
  table, and writes `stats_<ts>.csv` (runs automatically with ≥2 solvers and
  ≥5 instances; uses the existing `scipy` dependency — no new package).

### Changed
- **HV is now the primary reported metric**; MID is shown last and flagged
  *diagnostic*. MID rewards proximity to the ideal point, so it is not a sound
  stand-alone quality measure for a Pareto front — HV captures both convergence
  and diversity. MID is still computed for backward-compatibility with the 2023
  paper. Affects the `assign`/`metrics`/`benchmark` summary tables only; the
  `metrics_*.json` schema is unchanged.
- Pareto CSV now carries `z1..zk` objective columns (adds `z3` for the
  humanitarian model); `pva metrics` auto-detects objective dimensionality.
- `pva assign` gains `--model`, `--people`, and `--centers`; existing
  ed-staffing invocations are unchanged.
- Minimum Python is 3.10.

### Security
- The mandatory on-run dependency audit (Presidio extension #4) and
  security-event log (#5) now run on **every** command — `metrics`, `benchmark`,
  and `show` previously skipped them. Centralised in a shared CLI preamble.
- CSV formula-injection hardening: record IDs written to `assignments_*.csv` are
  quote-prefixed when they begin with a spreadsheet formula trigger
  (`= + - @`, tab, CR). Closes a latent issue that predated v0.2.0.

### Notes
- The humanitarian FIS ship with explicit, documented Mamdani rule tables and
  membership functions (`fis_humanitarian.py`), with a worked numeric example in
  `docs/fis-worked-example.md` and runnable synthetic datasets under `examples/`.

## [0.1.0]

Initial release: ED volunteer-staffing model (Rabiei et al., ESWA 2023) — three
FIS + NSGA-II + NRGA, CSV I/O, Pareto metrics (NNS/MID/SM/HV), and the Presidio
hardening profile.
