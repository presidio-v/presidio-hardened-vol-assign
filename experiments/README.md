# Experiments — allocation model audit

Drivers and result manifests behind the audit of the published relief-allocation model
(Paper B; companion to Rabiei, Arias-Aranda, Stantchev, *Appl. Sci.* 16(15):7581) and the
original study's runs (`run_h1_h2_h4.py`, `run_h3a.py`, `run_h3b.py`, ...).

Run every command from the repository root with the package importable, e.g.
`uv run python -m experiments.<module> ...`. All runs are seeded
(`1000 + 7919·k`); the cross-environment workflow
(`.github/workflows/repro-crossenv.yml`) shows that fronts, fuzzy caches and exact decisions
are bit-for-bit identical on macOS arm64 and Ubuntu x86-64 (Python 3.11, 3.12).

## What is committed and what is regenerated

Committed under `experiments/results/`: every summary and manifest the paper quotes
(CSV and `meta.json`). Not committed, because they are large: per-seed Pareto fronts
(`seed_floor/*/fronts.jsonl`) and generated instances (`envelope/instances/`,
`turbulence_extended/instances/`). Fronts regenerate bit for bit from the seeds. Generated
instances regenerate bit for bit since revision R1 of Paper B: the generator used to salt its
random stream with Python's `hash()`, which is randomised per process, and now uses CRC-32
(`tests/test_generate_instances.py`). The instances behind the submitted version's scaling table
(`run_scaling`) and the alternate instances `results/alt_instances/` were made with the old salt
and cannot be regenerated; the alternate instances are archived with their SHA-256 hashes, and the
revision's envelope instances are archived as well.

## Paper B: where each number comes from

| Paper element | Producer | Output |
|---|---|---|
| Seed floor, budget curve, solver / population / instance controls (RQ1, RQ2, Fig. 3, Table "RQ1") | `run_seed_floor --size {small,large} --seeds 20 --generations 150` (plus `--generations 300…9600 --seeds 5`, `--solver {nrga,nsga3}`, `--pop-size {200,400}`, `--instances results/alt_instances/seed{43,44}/<size>`) → `analyze_audit` | `results/seed_floor/`, `results/audit/audit_summary.csv` |
| Paper A runs through the dominance check (Table "paperA") | stored fronts of `run_h1_h2_h4`; see `analyze_audit.py` docstring | `results/h1_h2_h4/` (original study) |
| Fronts vs exact front (Fig. 2) | `make_audit_figures fronts` | figure PDF |
| Floor-adjusted MOEA turbulence (RQ3, text) | `run_turbulence` / `run_turbulence_full.sh` → `analyze_floor --size {small,large}` | `results/turbulence/`, `results/floor_analysis/` |
| Exact-decider regret (RQ3, Fig. 4, Table "RQ3") | `run_turbulence_exact --size {small,large} --realizations 50` | `results/turbulence_exact/` |
| Consensus of k runs (Appendix) | `analyze_consensus` | `results/audit/consensus.csv` |
| Repair, κ sweep (RQ4, Table "RQ4") | `run_repair_sweep` | `results/repair/kappa_sweep.csv` |
| Deployment envelope (RQ4, Fig. 5, Table "envelope") | `run_scaling --time-limit 60 --seeds 3`, `run_scaling --time-limit 300 --seeds 1 --min-people 1000 --manifest scaling_manifest_t300.csv` → `analyze_scaling` | `results/scaling/` |
| Exact-decider and cache timings | `profile_exact` | `results/audit/exact_timing.csv` |
| Paper A H1 on the exact 4-objective front (Discussion) | `analyze_paperA_h1_exact --steps {12,20}` | `results/audit/paperA_h1_exact*.csv` |
| Cross-environment reproducibility (Appendix) | CI workflow `repro-crossenv.yml` (`run_reproducibility`, `compare_repro`) | `results/repro/crossenv/` |
| All audit figures | `make_audit_figures [fronts budget turbulence scaling]` | `pubs/.../figures/audit/` (not in git) |

## Paper B revision (R1, November 2026)

| Paper element | Producer | Output |
|---|---|---|
| Table 2 with 20 seeds and CIs | `run_seed_floor --seeds 20` for every row (see the table) → `analyze_audit` | `results/audit/audit_summary.csv` |
| Sampled supported front vs lattice (Table "grid") | `analyze_weight_grid` | `results/audit/weight_grid.csv` |
| ULPP direction (RQ1) | `analyze_ulpp_direction` | `results/audit/ulpp_direction.csv` |
| Near-optimal multiplicity, tie-break (RQ2, Fig. "multiplicity", Table "tiebreak") | `run_multiplicity` | `results/multiplicity/` |
| Exact-decider trend tests without level 0 | `analyze_trend_exact` | `results/turbulence_exact/trend_tests.csv` |
| Turbulence across instances, conditions, bias, joint (RQ3) | `run_turbulence_extended --size {small,large} --instance {ref,101..105}` → `analyze_turbulence_extended` | `results/turbulence_extended/` |
| Repair-form sensitivity (Table "forms") | `run_repair_forms` | `results/repair/repair_forms.csv` |
| Envelope rerun (Table "envelope", Fig. 7, Appendix) | `run_envelope --time-limit {60,300} --seed {0,1,2}` (300 s from 1000 people up) → `make_envelope_tables` | `results/envelope/` |
| Second cases (Section "further models") | `run_ed_audit`, `run_knapsack_audit` | `results/ed_audit/`, `results/knapsack/` |
| Revision figures | `make_revision_figures --out <dir> [multiplicity turbulence envelope]` | figure PDFs |

Timings in the manifests are wall-clock on the machine recorded in each `meta.json` or
environment fingerprint; they are indicative, not reproducible.

## Paper A correction (Appl. Sci. 2026, 16, 7581; November 2026)

The objective of the published four-objective model averages the per-person score ULPP_j (the
unfairness of leaving person j behind) over the people **directed**; it should average it over
the people **left out**. `AllocationConfig(ulpp_objective="left_out")` (driver flag
`--ulpp-objective left_out`) implements the corrected objective; the default `"directed"` keeps
the published one, so the v0.2.0 results still reproduce. The exact deciders (`baselines`,
`exact_mip`, `multiplicity`, `local_search`) implement the published objective only and refuse
`"left_out"`.

`sh experiments/run_correction.sh <v0.2.0 results dir>` runs everything (about 1 h 45 min on
an M-series laptop):

| Step | Producer | Output under `results/correction/` |
|---|---|---|
| Reproduce the published H1/H2/H4 matrix (matches the v0.2.0 manifest exactly) | `run_h1_h2_h4` | `repro_directed/h1_h2_h4/manifest.csv` |
| H1/H2/H4, pop-200, H3a (3 sizes), H3b (3 sizes) under the corrected objective | `run_h1_h2_h4`, `run_h3a`, `run_h3b` with `--ulpp-objective left_out` | `left_out/` |
| Tables 3, 4, A7, A8 and the H3b CVs, published and corrected | `correction_tables` | `tables/{published,corrected}_*.csv` |
| H1 Spearman, null model, instance-aware HV | `analyze_h1`, `analyze_h1_null`, `analyze_hv_instance_aware` (`PVA_RESULTS`) | `left_out/h1_h2_h4/*.csv` |
| Figures 1–5 of the article (corrected HV, published CPU times) | `correction_cpu_merge`, `make_figures` (`PVA_RESULTS`, `PVA_FIGURES`, `PVA_H1_MANIFEST`) | `figures_corrected/` (not committed) |

Per-run fronts (`*/rep*/`) are not committed; they regenerate bit for bit from the seeds.
