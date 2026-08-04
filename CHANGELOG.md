# Changelog

All notable changes to `presidio-hardened-vol-assign` are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/), and the
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`pva build-demo`**: pre-solves a grid of slider positions and emits the demo
  GUI as a self-contained static site, so it can be hosted without any
  server-side Python. Everything downstream of the solve (trade-off slider, map,
  load table, CSV export) already ran in the browser, so only the run itself
  needed replacing — the page fetches a prebaked payload instead of posting to
  `/api/run`. Grid points are addressed by slider *index* rather than value, so
  the Python builder and the JavaScript frontend cannot disagree on number
  formatting. The compact grid is 648 runs (~46 MB on disk; ~10 kB per run over
  the wire, thanks to the generated `.htaccess` enabling `mod_deflate`).
- **`.github/workflows/deploy-demo.yml`**: manual-dispatch build-and-deploy of
  the static demo to STRATO webspace over SFTP, mirroring the pattern used by
  the Astro sites. Requires the `STRATO_SFTP_PASS` repository secret.

## [0.3.0] — 2026-08-03

Completes the v0.3.0 milestone: **sensitivity analysis** (delivered in 0.2.0)
plus the **interactive Pareto explorer**, which ships as a browser GUI rather
than the originally sketched matplotlib widget. Also releases the
evidence-carrying allocation work that landed after the 0.2.0 tag.

### Added
- **Evidence-carrying allocation** (`--emit-evidence` on `assign` /
  `allocate-people`; new `pva verify-evidence`): each run can emit a signed,
  content-addressed record (schema `presidio-hardened/allocation-evidence@1`)
  binding input snapshots (hashes + row counts, **no row contents**), the
  solver/seed/config, the Pareto front, an assignments digest, and the metrics —
  offline-verifiable, fail-closed. Canonical JSON with **bare-float rejection**
  (numbers as shortest round-trip decimal strings); SHA-256 content addressing;
  detached Ed25519 (optional `crypto` extra) or HMAC-SHA256 (stdlib) signature;
  trust-store verification. The volatile timestamp/emitter sit in the envelope so
  the `content_hash` is reproducible under a fixed seed (dovetails with the REP
  metric). Default **off**; behaviour byte-identical when unset. The humanitarian
  instantiation of evidence-carrying decisions (computational jurisprudence;
  Stantchev 2026, arXiv, ID pending). New module `evidence.py`; optional `crypto`
  extra. (Merged after the v0.2.0 tag, so it releases here rather than in 0.2.0.)
- **Interactive demo GUI** (`pva serve`, `web` extra): a browser front-end over
  the existing solver with three presets — volunteers → EDs, people → relief
  centres (soft capacity), and last-mile allocation under hard capacity limits.
  Instances are synthetic and generated from `(preset, sliders, seed)`; nothing
  is uploaded or stored. A trade-off slider walks the Pareto front with a live
  map, objective bars and per-site load table, and the chosen allocation
  downloads as CSV.
- **Demo API** on `presidio-hardened-fastapi`: `GET /api/scenarios`,
  `POST /api/run`, `GET /api/health`. Runs execute in a worker process under a
  wall-clock timeout, are capped at 300 units / 20 sites / 200 generations, and
  `/api/run` is rate-limited to 12 requests per minute per IP. Setting
  `PVA_EVIDENCE_KEY` enables signed evidence records through the same code path
  as `--emit-evidence`.
- **Dockerfile** for the demo server: multi-stage build, unprivileged user,
  healthcheck.

### Fixed
- **`pymoo` was used but never declared as a dependency.** `metrics.py` falls
  back to pymoo's hypervolume when the deap wheel omits
  `deap.tools._hypervolume` — which every deap 1.4 wheel does — so a fresh
  install failed at import on every CLI path. The fallback only appeared to work
  in environments where pymoo happened to be installed for other reasons.

## [0.2.0] — 2026-06-30

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
- **`pva ablation`** (`ablation.py`): leave-one-objective-out analysis that
  re-solves with each objective dropped in turn and reports how the dropped
  objective and the overall hypervolume degrade (`ablation_<ts>.csv`) — empirical
  evidence that each qualitative indicator is non-redundant (ATRES reviewer R2.2).
- **Exact weighted-sum baseline** (`--solver exact`): the scalarisation solved to
  optimality per weight — Hungarian assignment (`scipy.optimize.linear_sum_assignment`)
  for ed-staffing, MILP (`scipy.optimize.milp`) for humanitarian — a
  globally-optimal-per-scalarisation comparator stronger than greedy
  (`Domain.exact_baseline_population`; `pva benchmark --exact`). Uses the existing
  `scipy` dependency (ATRES reviewers R2.4 / R3.2).
- **Hard-capacity / transport-limit mode** for the humanitarian model
  (`--hard-capacity`, `--max-distance`, `--mobility-threshold`): a deterministic
  repair decoder guarantees no centre exceeds capacity and keeps low-mobility
  people within a maximum distance, complementing the default soft-capacity
  objective (ATRES reviewer R2.5). The soft model remains the default.

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
- Bumped the transitive `msgpack` pin to ≥ 1.2.1 in `uv.lock` to clear
  GHSA-6v7p-g79w-8964 (pulled in via the `pip-audit` toolchain), keeping the
  on-run and CI dependency audit green.

### Notes
- The humanitarian FIS ship with explicit, documented Mamdani rule tables and
  membership functions (`fis_humanitarian.py`), with a worked numeric example in
  `docs/fis-worked-example.md` and runnable synthetic datasets under `examples/`.

## [0.1.0]

Initial release: ED volunteer-staffing model (Rabiei et al., ESWA 2023) — three
FIS + NSGA-II + NRGA, CSV I/O, Pareto metrics (NNS/MID/SM/HV), and the Presidio
hardening profile.
