# presidio-hardened-vol-assign

A production-ready Python CLI tool (`pva`) implementing the multi-objective volunteer assignment model from:

> Rabiei, P., Arias-Aranda, D., & Stantchev, V. (2023). Introducing a novel multi-objective optimization model for volunteer assignment in the post-disaster phase: Combining fuzzy inference systems with NSGA-II and NRGA. *Expert Systems With Applications*, 226, 120142.

The tool combines three Fuzzy Inference Systems with NSGA-II and NRGA to produce Pareto-optimal volunteer assignment solutions for post-disaster Emergency Department staffing.

---

## Installation

```bash
pip install presidio-hardened-vol-assign
# or with uv:
uv add presidio-hardened-vol-assign
```

**Requirements:** Python 3.9+

---

## Quick start

### 1. Prepare input CSVs

**`volunteers.csv`** — one row per volunteer:

| Column | Description | Range |
|--------|-------------|-------|
| `volunteer_id` | Unique identifier | string |
| `skill_type` | `triage` or `er_nurse` | — |
| `skill_level` | Proficiency score | 0–10 |
| `distance_ed_<ED_ID>` | Distance to each ED (km) | 0–100 |
| `difficulty_tolerance` | Willingness to work under pressure | 0–10 |

```csv
volunteer_id,skill_type,skill_level,distance_ed_ED1,distance_ed_ED2,difficulty_tolerance
V1,triage,8.0,5.0,12.0,7.0
V2,triage,6.5,3.0,8.5,5.0
V3,er_nurse,9.0,15.0,4.0,8.0
```

**`eds.csv`** — one row per vacancy:

| Column | Description | Range |
|--------|-------------|-------|
| `ed_id` | Emergency Department identifier | string |
| `vacancy_type` | `triage` or `er_nurse` | — |
| `num_patients` | Current patient load | 0–100 |
| `emergency_level` | ED criticality score | 0–10 |

```csv
ed_id,vacancy_type,num_patients,emergency_level
ED1,triage,40,8.0
ED2,er_nurse,25,6.5
```

---

### 2. Run the solver

```bash
pva assign \
  --volunteers volunteers.csv \
  --eds eds.csv \
  --solver both \
  --seed 42 \
  --output results/
```

Terminal output:

```
Problem: 3 volunteers, 2 vacancies | solver: both | pop: 100  gen: 200

┌─────────────────────────────┐
│    Results — NSGA2          │
├──────────────┬──────────────┤
│ Metric       │        Value │
├──────────────┼──────────────┤
│ NNS          │            4 │
│ MID          │       0.3821 │
│ SM           │       0.0412 │
│ HV           │       0.4156 │
│ CPU time     │        1.24s │
└──────────────┴──────────────┘
  Pareto CSV  → results/pareto_nsga2_20240101T120000.csv
  Assignments → results/assignments_nsga2_20240101T120000.csv
  Metrics     → results/metrics_nsga2_20240101T120000.json

┌─────────────────────────────┐
│    Results — NRGA           │
├──────────────┬──────────────┤
│ Metric       │        Value │
├──────────────┼──────────────┤
│ NNS          │            3 │
│ MID          │       0.3904 │
│ SM           │       0.0387 │
│ HV           │       0.4021 │
│ CPU time     │        1.18s │
└──────────────┴──────────────┘
  Pareto CSV  → results/pareto_nrga_20240101T120000.csv
  Assignments → results/assignments_nrga_20240101T120000.csv
  Metrics     → results/metrics_nrga_20240101T120000.json
```

---

### 3. Output files

**`pareto_<solver>_<timestamp>.csv`** — one row per Pareto-front solution:

```csv
solver,solution_id,z1,z2
nsga2,0,0.312451,0.421300
nsga2,1,0.298760,0.445810
nsga2,2,0.341200,0.398750
```

- `z1` — mean importance of unmet nursing needs (minimise)
- `z2` — mean volunteer preference dissatisfaction (minimise)

**`assignments_<solver>_<timestamp>.csv`** — per-assignment details for each solution:

```csv
solution_id,volunteer_id,ed_id,vacancy_type,fis1_score,fis2_score,fis3_score
0,V1,ED1,triage,0.312451,0.0,0.198300
0,V3,ED2,er_nurse,0.0,0.421300,0.145200
```

**`metrics_<solver>_<timestamp>.json`** — summary metrics:

```json
{
  "solver": "nsga2",
  "nns": 4,
  "mid": 0.382100,
  "sm": 0.041200,
  "hv": 0.415600,
  "cpu_time_sec": 1.240
}
```

**`pva.log`** — structured JSON-lines security event log (no PII):

```json
{"ts": "2024-01-01T12:00:00+00:00", "level": "INFO", "version": "0.1.0", "event": "presidio-hardened-vol-assign loaded", "audit_status": "ok", "n_vulnerabilities": 0}
```

---

### 4. Re-compute metrics from a saved Pareto CSV

```bash
pva metrics --pareto results/pareto_nsga2_20240101T120000.csv
```

---

### 5. Check version and security status

```bash
pva version
```

```
presidio-hardened-vol-assign 0.1.0
Dependency audit: OK (last checked: 2024-01-01 12:00 UTC, 0 vulnerabilities)
```

---

## CLI reference

```
pva assign   --volunteers <csv>  --eds <csv>
             [--solver  nsga2|nrga|both]   (default: both)
             [--seed    <int>]              (reproducibility)
             [--pop-size <int>]             (default: 100)
             [--generations <int>]          (default: 200)
             [--output  <dir>]              (default: ./results)

pva metrics  --pareto <csv>

pva version
```

---

## Metrics explained

| Metric | Description |
|--------|-------------|
| **NNS** | Number of Non-dominated Solutions — Pareto front size |
| **MID** | Mean Ideal Distance — mean Euclidean distance from each solution to the ideal point (0, 0) |
| **SM** | Spacing Metric — standard deviation of consecutive inter-solution distances (lower = more uniform spread) |
| **HV** | Hypervolume — area of objective space dominated by the front relative to reference point (1, 1) (higher = better) |

---

## Security

`presidio-hardened-vol-assign` applies the Presidio hardening profile:

- **CSV sanitisation** — schema, types, and value ranges validated before any computation
- **Path traversal guard** — `--output` paths are resolved to absolute form; `..` traversal is rejected
- **Secure logging** — volunteer IDs only; no names, addresses, or other PII written to logs
- **Dependency audit** — `pip-audit` runs at startup and in CI; unpatched CVEs trigger a warning
- **CodeQL analysis** — automated on every push and weekly schedule

To report a vulnerability, see [SECURITY.md](SECURITY.md).

---

## Roadmap

See [PRESIDIO-REQ.md](PRESIDIO-REQ.md) for full version deliberations.

| Version | Status | Description |
|---------|--------|-------------|
| v0.1.0 | In progress | MVP: FIS + NSGA-II + NRGA, CSV I/O, Pareto metrics |
| v0.2.0 | Planned | Sensitivity analysis + interactive Pareto explorer |
| v0.3.0 | Planned | Benchmark reproducibility + real-world data connectors |

---

## License

MIT

---

## SDLC

This repository is developed under the Presidio hardened-family SDLC:
<https://github.com/presidio-v/presidio-hardened-docs/blob/main/sdlc/sdlc-report.md>.
