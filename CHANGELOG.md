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

### Changed
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
