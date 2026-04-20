# PRESIDIO-REQ — presidio-hardened-vol-assign

## Project Overview

`presidio-hardened-vol-assign` is a production-ready Python CLI tool (`pva`) that implements the multi-objective volunteer assignment model introduced in:

> Rabiei, P., Arias-Aranda, D., & Stantchev, V. (2023). Introducing a novel multi-objective optimization model for volunteer assignment in the post-disaster phase: Combining fuzzy inference systems with NSGA-II and NRGA. *Expert Systems With Applications*, 226, 120142. https://doi.org/10.1016/j.eswa.2023.120142

The tool combines three Fuzzy Inference Systems (FIS) with two metaheuristic solvers (NSGA-II, NRGA) to produce Pareto-optimal volunteer assignment solutions for post-disaster Emergency Department staffing. Primary target users are researchers validating or extending the model; the output format is designed to reproduce the paper's Table 3 metrics exactly.

**Target usage:**
```bash
pva assign --volunteers data/volunteers.csv --eds data/eds.csv \
           --solver both --output results/
pva metrics --pareto results/pareto_nsga2_20240101T120000.csv
```

---

## Mandatory Presidio Security Extensions

1. **CSV input sanitization** — reject non-numeric fields where numerics are expected; no shell injection via file paths or field values
2. **Strict schema validation** — validate column headers, value ranges, and types before running any computation; fail fast with a clear error message
3. **Secure logging** — no PII from volunteer roster (names, addresses) in logs; log only IDs and aggregate statistics
4. **On-run CVE/dependency check** — run `pip-audit` (or equivalent) at startup; emit a timestamped warning if the check is stale or fails
5. **Security event logging** — emit `"presidio-hardened-vol-assign vX.Y.Z loaded; security checks passed"` to structured log at every invocation
6. **Path traversal guard** — resolve `--output` to an absolute path; reject any argument containing `..` path components
7. **Full GitHub security files** — `SECURITY.md`, `.github/dependabot.yml`, `.github/workflows/codeql.yml`, `.github/workflows/ci.yml` (pytest + ruff + pip-audit)

---

## Technical Requirements

- **Language:** Python 3.9+
- **Build system:** `pyproject.toml` with `hatchling` backend; `uv` for dependency management
- **Project layout:** `src/presidio_vol_assign/` (src-layout)
- **CLI framework:** Typer + rich (terminal tables and progress bars)
- **FIS:** scikit-fuzzy (membership functions and rule base)
- **Metaheuristics:** DEAP (NSGA-II and NRGA solvers)
- **Data handling:** numpy, pandas
- **Testing:** pytest, pytest-cov (80%+ coverage required)
- **Linting/formatting:** ruff (`ruff format` + `ruff check --fix`)
- **Documentation:** README.md with side-by-side CSV input → Pareto output example
- **License:** MIT
- **Version:** 0.1.0

---

## CLI Interface

```
pva assign   --volunteers <csv>   Path to volunteer roster CSV
             --eds <csv>          Path to Emergency Department vacancies CSV
             [--solver nsga2|nrga|both]   Solver selection (default: both)
             [--seed <int>]               Random seed for reproducibility
             [--pop-size <int>]           Population size (default: 100)
             [--generations <int>]        Number of generations (default: 200)
             [--output <dir>]             Output directory (default: ./results)

pva metrics  --pareto <csv>       Compute NNS, MID, SM, HV for a Pareto front CSV

pva version                       Print version, dependency check status
```

**Output files** written to `--output` directory:
- `pareto_<solver>_<timestamp>.csv` — one row per Pareto-optimal solution (Z1, Z2, assignment columns)
- `metrics_<solver>_<timestamp>.json` — `{NNS, MID, SM, HV, cpu_time_sec}`
- `pva.log` — structured security event log (JSON lines)

---

## Core Algorithm

### Input CSV Formats

`volunteers.csv` (one row per volunteer):
```
volunteer_id, skill_type, skill_level, distance_ed_1, ..., distance_ed_N, difficulty_tolerance
```
- `skill_type`: `triage` or `er_nurse`
- `skill_level`: float [0, 10]
- `distance_ed_i`: float [0, 100] km
- `difficulty_tolerance`: float [0, 10]

`eds.csv` (one row per vacancy):
```
ed_id, vacancy_type, num_patients, emergency_level
```
- `vacancy_type`: `triage` or `er_nurse`
- `num_patients`: int [0, 100]
- `emergency_level`: float [0, 10]

### Fuzzy Inference Systems

**FIS1** — Importance of unmet triage-nurse needs
- Inputs: `num_patients` (universe 0–100), `ed_emergency_level` (0–10), `volunteer_skill` (0–10)
- Output: `importance_unmet_triage` (0–1)

**FIS2** — Importance of unmet primary-ER-nurse needs
- Same input/output structure as FIS1

**FIS3** — Degree of unsatisfied volunteer preferences
- Inputs: `distance_to_ed` (0–100 km), `workload` (0–10), `difficulty_tolerance` (0–10)
- Output: `preference_dissatisfaction` (0–1)

Membership functions: triangular and trapezoidal, matching linguistic variables from paper Tables 1–2. Rule bases: Mamdani inference with centroid defuzzification.

### Objective Functions

- **Z1** = mean importance of unmet needs across all vacancies (minimize)
- **Z2** = mean degree of unsatisfied preferences for selected volunteers (minimize)

### Constraints
- Each vacancy is filled by exactly one volunteer
- Each selected volunteer receives at most one assignment
- Number of available volunteers ≥ number of vacancies

### Solvers
Both implemented via DEAP:
- **NSGA-II** — Non-Dominated Sorting Genetic Algorithm II
- **NRGA** — Non-dominated Ranked Genetic Algorithm
- `--solver both` runs both in sequence and outputs separate files per solver

### Metrics (paper Table 3 format)
- **NNS** — Number of Non-dominated Solutions
- **MID** — Mean Ideal Distance
- **SM** — Spacing Metric
- **HV** — Hypervolume indicator

---

## Roadmap

| Version | Status | Description |
|---------|--------|-------------|
| v0.1.0 | Planned | MVP: FIS (scikit-fuzzy) + NSGA-II + NRGA (DEAP), CSV I/O, Pareto metrics |
| v0.2.0 | Planned | Sensitivity analysis (`pva sensitivity`) + interactive Pareto explorer (`pva show`) |
| v0.3.0 | Planned | Benchmark reproducibility (`pva benchmark`) + real-world data connectors (`pva import`) |

See **Version Deliberations** below for full rationale.

---

## Version Deliberations

### v0.1.0 — MVP

**Scope decisions:**
- Implement all three FIS in `src/presidio_vol_assign/fis.py` using scikit-fuzzy. Use triangular/trapezoidal membership functions that reproduce the linguistic variables in paper Tables 1–2. Membership function parameters are hard-coded constants (not configurable) in v0.1.0.
- Both NSGA-II and NRGA via DEAP in `src/presidio_vol_assign/solvers.py`. The `--solver both` flag runs both sequentially and writes separate output files. Default hyperparameters (`pop_size=100`, `generations=200`) match the paper's experimental setup.
- `--seed` flag enables exact reproducibility. Without a seed, DEAP's default RNG is used.
- Input validation in `src/presidio_vol_assign/validation.py`: check CSV schema, value ranges, and constraint feasibility (enough volunteers) before touching the algorithm.
- Metrics module in `src/presidio_vol_assign/metrics.py` computes NNS, MID, SM, HV from a Pareto CSV.
- Test suite: unit tests for each FIS on known inputs, integration tests for small (5 EDs, 75 volunteers) and large (10 EDs, 150 volunteers) problem sizes. Fixtures in `tests/fixtures/`.

**What is NOT in v0.1.0:**
- No configurable FIS rule files (YAML/JSON) — too complex for MVP; add in v0.2.0 if needed
- No visualization — deferred to v0.2.0
- No paper benchmark instance regeneration — deferred to v0.3.0

### v0.2.0 — Sensitivity Analysis + Pareto Explorer

**Sensitivity analysis (`pva sensitivity`):**
- Vary FIS rule output weights ±10% and ±20% in a grid sweep
- Report how Z1/Z2 distributions on the Pareto front shift
- Output: CSV with (perturbation_factor, NNS, MID, SM, HV) for each sweep step
- Rationale: researchers need to understand model robustness to FIS rule specification uncertainty

**Interactive Pareto explorer (`pva show`):**
- Scatter plot of Z1 vs. Z2 Pareto front (matplotlib with `%matplotlib widget` or plotly)
- Click a point to display the full assignment table in the terminal (rich Table)
- `--solver both` shows overlaid NSGA-II and NRGA fronts with distinct colors
- Rationale: practitioners need to navigate the trade-off and select a preferred solution; researchers need to compare solvers visually

### v0.3.0 — Benchmark Reproducibility + Data Connectors

**Benchmark reproducibility (`pva benchmark`):**
- Regenerate all 30 small-size and 30 large-size problem instances using the paper's documented random seed procedure
- Run both solvers on all instances; compute aggregate metrics
- Output: summary table reproducing paper Table 3 format (mean ± std for CPU time, NNS, MID, SM, HV)
- Rationale: enables independent replication and citation; future researchers can verify their extensions against the baseline

**Real-world data connectors (`pva import`):**
- `pva import --source roster.xlsx` — import volunteer roster from Excel (openpyxl); normalize to CSV format
- `pva import --source http://...` — fetch ED vacancy list from a simple REST endpoint (JSON); validate schema before writing CSV
- Security: HTTP imports require HTTPS; certificate validation enforced; no auth tokens via CLI flags (use env vars)
- Rationale: bridges the tool from research use to real-world Emergency Operations Center (EOC) data systems

---

## Workflow Rules

1. First create or update `PRESIDIO-REQ.md` from the presidio-template ✓
2. Comment out the final delivery line below before starting implementation
3. Implement file-by-file in logical order (scaffold → models → FIS → solvers → CLI → security → tests → docs)
4. After every major section run `ruff format . && ruff check --fix . && pytest` and fix all issues automatically
5. When complete, reply exactly: "BUILD COMPLETE – ready for publish"

## Claude Code Rules

- Never commit `.claude/`, `.specify/`, or any temporary files
- Never add Co-authored-by lines
- Always commit with clean author name only

<!-- Deliver the complete working project ready for GitHub publish. -->

## SDLC

These requirements are delivered under the family-wide Presidio SDLC:
<https://github.com/presidio-v/presidio-hardened-docs/blob/main/sdlc/sdlc-report.md>.
