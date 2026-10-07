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
(CSV and `meta.json`). Not committed, because they are large and regenerate bit for bit:
per-seed Pareto fronts (`seed_floor/*/fronts.jsonl`, ~15 MB) and the generated scaling
instances (`scaling/instances/`, ~26 MB). Rebuild them with the commands below before running
the analysis scripts that read them.

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

Timings in the manifests are wall-clock on the machine recorded in each `meta.json` or
environment fingerprint; they are indicative, not reproducible.
