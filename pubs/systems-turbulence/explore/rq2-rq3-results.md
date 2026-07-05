# Paper B — RQ2 / RQ3 results

Sources: `experiments/results/repro/repro_manifest.csv` (RQ2),
`experiments/results/operability/operability_manifest.csv` (RQ3). Drivers:
`experiments/run_reproducibility.py`, `experiments/profile_operability.py`
(macOS-15.7 / arm64 / py3.12, pop 100 / gen 200, 3-obj relief model).

## RQ2 — reproducibility & auditability (H-B2 supported)

- **Within-environment REP = 1.0** across all 20 runs (2 sizes × 10 seeds): two in-process
  solves of the same (size, seed) hash to the same allocation-front signature, bit-for-bit.
- **Seed-sensitive:** 10 distinct seeds → 10 distinct signatures per size (the hash reflects
  the run; it is not a constant).
- **Audit trail:** each run records an environment fingerprint (platform, Python, numpy /
  scipy / deap / pymoo versions) with its signature. Cross-environment REP = diff the
  `signature` column across OS/Python runs (CI matrix automates this).
- **Cross-environment REP = 1.0** (CI, `.github/workflows/repro-crossenv.yml`): across
  {Ubuntu, macOS} × {py3.11, 3.12} every (size, seed) produced the *identical* signature
  (5 seeds × 4 envs, distinct=1) — bit-for-bit reproducible across platforms + Python versions,
  not merely within one (small instance, reduced budget).
- Ties RQ1 → RQ2: the fragile decision is at least *accountable* — a stakeholder can verify
  which run produced which allocation.

## RQ3 — operability envelope (H-B3 supported)

| size / algorithm | latency (s) | throughput (dec./h) | peak memory (MB) |
|---|---|---|---|
| small / NSGA-II | 6.20 | 580 | 1.4 |
| small / NRGA | 4.34 | 829 | 1.4 |
| large / NSGA-II | 8.99 | 400 | 2.6 |
| large / NRGA | 6.78 | 531 | 2.6 |

Full Pareto set in 4–9 s using ≤3 MB working memory at both scales; NRGA faster than
NSGA-II. Field-deployable on a laptop without specialised infrastructure.
