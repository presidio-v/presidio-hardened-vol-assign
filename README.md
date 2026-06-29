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

**Requirements:** Python 3.10+

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
{"ts": "2024-01-01T12:00:00+00:00", "level": "INFO", "version": "0.2.0", "event": "presidio-hardened-vol-assign loaded", "audit_status": "ok", "n_vulnerabilities": 0}
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
presidio-hardened-vol-assign 0.2.0
Dependency audit: OK (last checked: 2024-01-01 12:00 UTC, 0 vulnerabilities)
```

---

## CLI reference

```
pva assign   [--model  ed-staffing|humanitarian]   (default: ed-staffing)

             # ed-staffing model inputs:
             --volunteers <csv>  --eds <csv>
             # humanitarian model inputs:
             --people <csv>      --centers <csv>

             [--solver  nsga2|nrga|nrga-ranked|greedy|exact|both|all]   (default: both)
             [--seed    <int>]              (reproducibility)
             [--pop-size <int>]             (default: 100)
             [--generations <int>]          (default: 200)
             # humanitarian hard-constraint mode (optional):
             [--hard-capacity]              (enforce centre capacity via repair)
             [--max-distance <km>]          (cap distance for low-mobility people)
             [--mobility-threshold <0-10>]  (default: 3.0)
             [--output  <dir>]              (default: ./results)

pva allocate-people  --people <csv> --centers <csv>   [solver/seed/... as above]
             # convenience alias for `assign --model humanitarian`
             # also accepts --hard-capacity / --max-distance / --mobility-threshold

pva metrics  --pareto <csv>     (auto-detects 2- or 3-objective fronts)

pva show     --pareto <csv> [--pareto <csv> ...]   (overlay solvers)
             [--output <png|svg>] [--title <str>]

pva benchmark [--model humanitarian|ed-staffing]
              [--size  small|large|both]   (default: both)
              [--instances <int>]          (default: 10, per size)
              [--solver nsga2|nrga|nrga-ranked|greedy|both|all]
              [--seed <int>]               (default: 42)
              [--pop-size <int>] [--generations <int>]
              [--check-repro]              (report bit-for-bit REP)
              [--baseline]                 (add greedy comparator + Wilcoxon HV test)
              [--exact]                    (add exact weighted-sum comparator)
              [--output <dir>]

pva sensitivity [--model humanitarian|ed-staffing]
              # inputs as for `assign` (--people/--centers or --volunteers/--eds)
              [--factors <csv>]            (default: -0.2,-0.1,0,0.1,0.2)
              [--solver ...] [--seed <int>] [--pop-size <int>] [--generations <int>]
              [--output <dir>]

pva ablation  [--model humanitarian|ed-staffing]
              # inputs as for `assign` (--people/--centers or --volunteers/--eds)
              [--solver ...] [--seed <int>] [--pop-size <int>] [--generations <int>]
              [--output <dir>]

pva version
```

### Sensitivity analysis

`pva sensitivity` probes how robust the Pareto front is to FIS rule-base
specification uncertainty: it rescales the FIS output scores by each
`--factors` perturbation (e.g. ±10 %, ±20 %), re-runs the solver(s), and writes
`sensitivity_<ts>.csv` with `(factor, solver, NNS, MID, SM, HV, cpu_time_sec)`.
The unperturbed FIS scores are computed once, so the sweep is cheap and
deterministic under `--seed`.

```bash
pva sensitivity --model humanitarian \
  --people people.csv --centers centers.csv \
  --factors -0.2,-0.1,0,0.1,0.2 --solver both --seed 42 --output results/
```

### Objective ablation (indicator validation)

`pva ablation` provides empirical evidence that each qualitative indicator
contributes distinct, non-redundant information. It re-solves the problem with
each objective **dropped from the optimisation** in turn, then measures the
dropped objective — and the overall hypervolume — back in the full objective
space, writing `ablation_<ts>.csv`:

- **Δ dropped** — how much worse the dropped objective gets (mean over the front)
  when it is no longer optimised. Large = the indicator is doing real work that
  no other objective drives for free (non-redundant).
- **Δ HV** — full-space hypervolume lost by ignoring the objective.

A near-zero Δ would flag an indicator the other objectives already capture.
Deterministic under `--seed`.

```bash
pva ablation --model humanitarian \
  --people people.csv --centers centers.csv \
  --solver nsga2 --seed 42 --output results/
```

### Solvers

| `--solver` | Algorithm |
|---|---|
| `nsga2` | NSGA-II (crowding-distance elitism) |
| `nrga` | Lightweight NRGA — front-fill with uniform random tie-break |
| `nrga-ranked` | Canonical NRGA — rank-biased roulette-wheel survival (Al Jadaan et al., 2008); use this for results comparable to the NRGA literature |
| `greedy` | **Non-evolutionary baseline** — deterministic weighted-sum constructive heuristic swept over the objective simplex; a literature-style comparator the GAs are measured against (see *Baseline comparison* below) |
| `exact` | **Exact weighted-sum baseline** — the scalarisation solved *to optimality* per weight (Hungarian assignment for ed-staffing; MILP for humanitarian); a stronger, globally-optimal-per-scalarisation comparator than `greedy` |
| `both` | `nsga2` + `nrga` |
| `all` | `nsga2` + `nrga` + `nrga-ranked` |

### Benchmarking & reproducibility

`pva benchmark` generates the paper's instance sizes deterministically
(humanitarian: 5 centres/150 people and 10/300; ed-staffing: 5/75 and 10/150),
runs the solver(s) on each, and prints a Table-3-style **mean ± std** summary for
NNS, MID, SM, HV, and CPU time, written to `benchmark_<ts>.{csv,json}`. With
`--check-repro` each instance is solved twice and the fraction of **bit-for-bit
identical** results is reported as **REP** — treating reproducibility on stock
hardware as a first-class resilience criterion.

```bash
pva benchmark --model humanitarian --instances 10 --seed 42 --check-repro
```

### Baseline comparison & significance testing

Comparing only NSGA-II against NRGA shows which *algorithm* wins, not whether the
*framework* beats an existing allocation method. `--solver greedy` provides a
non-evolutionary baseline: a deterministic weighted-sum constructive heuristic
swept across the objective simplex (each weight vector yields one greedy
allocation; the non-dominated subset is the baseline front). It is reproducible
regardless of `--seed`.

For a stronger comparator, `--solver exact` (or `pva benchmark --exact`) solves
the weighted-sum scalarisation **to optimality** at each weight — an exact
bipartite assignment (Hungarian) for ed-staffing, or a MILP for the humanitarian
model (`z1`/`z2` exact, centre balance via a linear capacity-overload surrogate;
the true FIS objectives are reported on the optimal assignment). It is
globally-optimal-per-scalarisation, so it is a much harder baseline to beat than
the myopic greedy. Both comparators use the existing `scipy` dependency.

`pva benchmark --baseline` / `--exact` run those comparators on every instance and
add `greedy` / `exact` rows to the Table-3 summary. When ≥ 2 solvers and ≥ 5
instances are present, the benchmark also runs a **Wilcoxon rank-sum test** on the
per-instance hypervolume distributions (each solver vs. a reference — the greedy
baseline if present, else NSGA-II), prints a significance table, and writes
`stats_<ts>.csv`.

```bash
pva benchmark --model humanitarian --solver all --baseline --exact --instances 10 --seed 42
```

### Figures

`pva show` renders publication-quality Pareto figures from the `pareto_*.csv`
files. Two-objective fronts give a Z1–Z2 scatter; three-objective fronts give the
three pairwise projections (Z1–Z2, Z1–Z3, Z2–Z3) plus a 3-D scatter, with
solvers overlaid. Requires the `viz` extra (`pip install
'presidio-hardened-vol-assign[viz]'`).

```bash
pva show --pareto results/pareto_nsga2_*.csv --pareto results/pareto_nrga_*.csv \
         --output results/fronts.png --title "Humanitarian allocation"
```

---

## Humanitarian allocation model (v0.2.0)

A second model allocates **affected people to relief centres**, optimising three
objectives via three new Fuzzy Inference Systems. It runs through the same CLI,
**side by side** with the ED-staffing model:

```bash
pva assign --model humanitarian \
  --people people.csv --centers centers.csv \
  --solver both --seed 42 --output results/

# equivalently, via the convenience alias (and the canonical NRGA variant):
pva allocate-people \
  --people people.csv --centers centers.csv \
  --solver nrga-ranked --seed 42 --output results/
```

**`people.csv`** — one row per affected person:

| Column | Description | Range |
|--------|-------------|-------|
| `person_id` | Unique identifier | string |
| `vulnerability` | Priority/need (elderly, injured, children) | 0–10 |
| `mobility` | Personal transport access (0 = immobile) | 0–10 |
| `group_size` | People moved together (optional, default 1) | 1–20 |
| `distance_center_<center_id>` | Distance to each centre (km) | 0–100 |

**`centers.csv`** — one row per relief centre:

| Column | Description | Range |
|--------|-------------|-------|
| `center_id` | Unique centre identifier | string |
| `capacity` | Nominal capacity (people) | 1–5000 |
| `service_level` | Resource/quality level | 0–10 |
| `road_accessibility` | Route condition / access | 0–10 |

**Objectives (all minimised):**

| | Objective | FIS |
|---|-----------|-----|
| `z1` | Unfairness of people prioritisation | Fairness FIS |
| `z2` | Transportation infeasibility | Transportation Feasibility FIS |
| `z3` | Centre overcrowding / imbalance | Center Allocation Balance FIS |

The output `pareto_*.csv` carries `z1,z2,z3`; `assignments_*.csv` carries
`person_id, center_id, fairness, transport, overcrowding`.

### Hard capacity & transport limits

By default the humanitarian model treats capacity as a *soft* objective (`z3`
overcrowding) and transport as a feasibility objective (`z2`). For settings where
those are firm operational limits, `--hard-capacity` switches on a constraint
mode: a deterministic repair guarantees **no centre exceeds its capacity** (people
are placed most-constrained-first; overflow goes to the nearest centre with spare
room), and `--max-distance` caps how far a **low-mobility** person (mobility below
`--mobility-threshold`) may be sent. The soft model remains the default and is
unchanged.

```bash
pva assign --model humanitarian --people people.csv --centers centers.csv \
  --hard-capacity --max-distance 30 --mobility-threshold 3 --solver both --seed 42
```

The membership functions and the explicit Mamdani rule tables for all three FIS
are documented — with a fully worked numeric example — in
[docs/fis-worked-example.md](docs/fis-worked-example.md). Ready-to-run synthetic
datasets live under [`examples/`](examples/) (`small/` = 12 people / 3 centres,
`paper_scale/` = 150 / 5); regenerate them with `python examples/generate_examples.py`.

```bash
pva assign --model humanitarian \
  --people examples/paper_scale/people.csv \
  --centers examples/paper_scale/centers.csv \
  --solver both --seed 42 --output results/
```

---

## Metrics explained

| Metric | Description |
|--------|-------------|
| **HV** *(primary)* | Hypervolume — area/volume of objective space dominated by the front relative to reference point (1, …, 1) (higher = better). Captures both convergence and diversity, so it is the headline quality indicator |
| **NNS** | Number of Non-dominated Solutions — Pareto front size |
| **SM** | Spacing Metric — standard deviation of nearest-neighbour inter-solution distances (lower = more uniform spread) |
| **MID** *(diagnostic)* | Mean Ideal Distance — mean Euclidean distance from each solution to the ideal point (0, …, 0). Retained for backward-compatibility with the 2023 paper, but reported as a diagnostic only: it favours solutions near the ideal point, whereas every Pareto-front member is an equally valid trade-off, so it is not a sound stand-alone quality measure — prefer HV |

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
| v0.1.0 | Released | MVP: FIS + NSGA-II + NRGA, CSV I/O, Pareto metrics (ED-staffing model) |
| v0.2.0 | In progress | Humanitarian allocation model (3 new FIS, 3 objectives) **side by side** with the ED model; N-D metrics + reproducibility metric; `benchmark` + `sensitivity` + figure export. See [docs/v0.2.0-plan.md](docs/v0.2.0-plan.md) |
| v0.3.0 | Planned | Interactive Pareto explorer + real-world data connectors |

---

## License

MIT

---

## SDLC

This repository is developed under the Presidio hardened-family SDLC:
<https://github.com/presidio-v/presidio-hardened-docs/blob/main/sdlc/sdlc-report.md>.
