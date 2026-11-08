# Revision results log — systems-4669426

Code: `~/projects/pva-experiments`, branch `feat/paperB-revision` (off `feat/paperB-experiments`, v0.4.0).
Env: Python 3.12.13, NumPy 2.4.4, SciPy 1.17.1, DEAP 1.4.x, pymoo 0.6.2, scikit-fuzzy 0.5.0, Apple M4, 16 GB.
Tests: 1164 pass at branch start; new tests added per module below.

## Defects found during the revision (disclose in the letters)

1. **Scaling instances were not reproducible.** `experiments/generate_instances.py` salted its RNG with
   `hash(spec.name)`; Python randomises str hashes per process, so every call of `run_scaling`
   regenerated different instances. Consequence: the 60 s and 300 s rows of submitted Table 6 ran on
   *different* instances (e.g. seed-0 3000x50: fallback overload 803.15 in the 60 s run, 808.07 in the
   300 s run). Reference instances (small/medium/large) are committed CSVs and are unaffected.
   Fix: CRC-32 salt; test `tests/test_generate_instances.py` (fails on old code, passes now). Envelope
   rerun on fixed stored instances (`run_envelope.py`).
2. **MOEA under the "300 s" limit did not get 300 s.** Generations were calibrated from a 20-gen timed
   run: NSGA-II ran 307.8 s (1000x25), 263.2 s (1500x40), 182.7 s (3000x50)
   (`scaling_manifest_t300.csv`). Fix: wall-clock deadline in `solve(time_budget=...)`.
3. **Shared preparation was untimed.** CAIL-per-load table: 12.7 s at 1000x25, 129 s at 3000x50 (one
   capacity variant); FIS scores 2.0 s / 11.1 s (compiled evaluator). Now reported as `fis_s`,
   `table_s`, `total_s`.

Bit-for-bit check after the solver changes: seed 1000 small reproduces `b68267fa...` (200 gen, local
manifest) and `639e4f8d...` (150 gen, all four CI environments).

## R2.5 / R3 — near-optimal multiplicity (`results/multiplicity/summary.csv`)

Max churn vs the exact optimum within epsilon (5 MGA alternatives, all proven unless noted):

| model | eps=0 | 0.1 | 1 | 5 | MOEA deficit |
|---|---|---|---|---|---|
| small published | 0.00 | 0.06 | 0.26 | 0.62 | 1.00 (17.8) |
| small soft k2 | 0.00 | 0.06 | 0.28 | 0.62 | 1.00 |
| small hard kf1.5 | 0.00 | 0.22 | 0.52 | 0.92 | 1.00 |
| large published | 0.01 | 0.12 | 0.33 | 0.73 | 1.00 (33.3) |
| large soft k2 | 0.06 | 0.17 | 0.45 | 0.95 | 1.00 |
| large hard kf1.5 | 0.22 | 0.38 | 0.73 | 1.00 | 1.00 |

Min directed-set Jaccard at the MOEA deficit: 0.08 small, 0.00 large. Reading: the seed floor (0.83 /
0.94) sits where the problem itself admits allocations that differ for every directed person; churn
near the optimum is the problem's multiplicity; exact optima are unique on small, tied on large.

Tie-break (`tiebreak.csv`, 10 presentation orders): plain solver 1/1/1 (small), 1/9/10 (large:
published/soft/hard) distinct optima; canonical (seeded PCG64 key, second lexicographic stage) 1 in
every cell.

## R3.2 — sampled supported front (`results/audit/weight_grid.csv`)

| size | steps/axis | supported pts | HV vs finest | MOEA share 150 gen | 9600 gen |
|---|---|---|---|---|---|
| small | 21 | 181 | 0.990 | 0.389 | 0.797 |
| small | 161 | 1443 | 1.000 | 0.385 | 0.789 |
| large | 21 | 197 | 0.995 | 0.180 | 0.384 |
| large | 161 | 3359 | 1.000 | 0.180 | 0.382 |

## R2.1 — second cases

ED staffing (`results/ed_audit/`): S1 not separable (one-to-one matching); S2 exact as linear
assignment. Probe: small (75 vol, 5 vac) exact sum 0.3592, NSGA-II best 0.3599, exact dominates 900/2000
points, churn 0.49 [0.39, 0.59]; large (150, 10) exact 0.3737, NSGA-II best 0.4085, 659/1920, churn 0.90.

Knapsack ZT-1999 recipe, k=2 (`results/knapsack/`): 250 items exact eq-weight sum 18744 (0.6 s), 37
supported points; NSGA-II 200 gen committed churn (Jaccard distance) 0.106 [0.066, 0.157], committed
sum 18573 +/- 46.

## S1 finding 2 — ULPP direction (`results/audit/ulpp_direction.csv`; VS decision 2026-11-03: report as S1 finding, add to Paper A correction, ask Peyman to confirm)

FIS1 gives high ULPP to high vulnerability / severe damage / short resource time; the objective minimises mean ULPP over the *directed*. Paper A prose: "Directing the most vulnerable people (low ULPP)". rho(ULPP, crisp priority) = 0.61 small, 0.64 large.

| size | decider | mean priority directed (all) | top-priority share directed | damage dir (all) | time left dir (all) | overlap w/ exact |
|---|---|---|---|---|---|---|
| small | exact (published) | 0.314 (0.393) | 0.12 | 44.3 (60.1) | 32.7 (24.6) | 1.00 |
| small | crisp priority | 0.532 | 1.00 | 67.2 | 22.0 | 0.12 |
| small | exact, documented reading (100-ULPP) | 0.459 | 0.54 | 70.9 | 14.3 | 0.08 |
| large | exact (published) | 0.282 (0.380) | 0.07 | 42.9 (56.2) | 33.8 (25.0) | 1.00 |
| large | crisp priority | 0.535 | 1.00 | 61.7 | 20.6 | 0.07 |
| large | exact, documented reading | 0.482 | 0.65 | 67.9 | 15.2 | 0.02 |

Explains 32 of the crisp rule's 40-unit gap on small (ULPP 72.09 vs 39.98).

## R2.1 — second cases, final (`results/ed_audit/floor.csv`, `results/knapsack/floor.csv`)

ED (20 seeds, pop 100, 200 gen): small exact 0.3592; NSGA-II best 0.3599, 406/2000 pts dominated, 11/20 seeds reach exact, churn 0.49 [0.39, 0.59]; NRGA 201/2000, 13/20, churn 0.42 [0.33, 0.52]. Large exact 0.3737; NSGA-II best 0.4085 (0/20 reach), 659/1920, churn 0.90 [0.87, 0.92]; NRGA 0.4224, 1195/1570, churn 0.91 [0.89, 0.94].

Knapsack (20 seeds, pop 100): 250 items exact 18744 (37 supported pts, sweep 31 s / 101 MIPs): 200 gen committed 18573 +/- 46, 29% pts dominated, churn 0.106; 2000 gen 18716 +/- 11, 4%, churn 0.062. 500 items exact 38007 (67 pts): 200 gen 37565 +/- 56, 60%, churn 0.109; 2000 gen 37901 +/- 30, 9%, churn 0.073.

## R2.7 — repair forms (`results/repair/repair_forms.csv`)
Linear reproduces Table 5 exactly (12/12 cells). Soft overload persists in 29/30 soft cells across forms (exception: convex, kappa=3, small: 0). Stretched rule base (load response x3): published optimum up to 46% worse (large), soft optimum still overloads 17.8-91.9. Hard capacity removes overload by construction in all forms.

## R2.6 — trend tests (`results/turbulence_exact/trend_tests.csv`)
Paper's rho 0.37-0.97 = levels 0.05-0.4 only (200 per cell); max Holm p 4.8e-8. With level 0: 0.44-0.98.

## R1.2 / R2.4 / R2.7 / R3 — envelope rerun (`results/envelope/`, 3 stored instances per size, all methods same instances)

- 60 s: soft MIP proven 3/3 up to 600x20, 1/3 at 1000x25, 0/3 at 1500x40, 1/3 *solved* at 3000x50 (150.7). Hard MIP proven up to 1500x40; 0/3 solved at 3000x50.
- 300 s (1000+): soft MIP proven 3/3, 1/3, 0/3; hard MIP 3/3, 3/3, proven 1/3 at 3000x50 (gap <= 0.02%), 242 s solver.
- LS soft within 0.2-0.5 of proven soft optimum up to 600x20 (<1 s); beats 60 s MIP from 1500x40 (96.5 vs 98.8; 95.7 vs 1 solved); at 300 s MIP overtakes at 1500x40 (95.7 vs 96.5) but not at 3000x50 (96.4 vs 95.7); LS solver <= 15 s.
- LS hard within 1.5% of hard MIP at all sizes, <= ~3 s solver.
- NSGA-II time-boxed: 6-43 units behind soft MIP; behind fallback at every size. Seeded: up to +44 units from 300x10; within ~1 of fallback from 600x20; inherits fallback overload.
- Prep at 3000x50: FIS 10.4 s; soft table 95 s; hard table 160 s. LS-hard total ~175 s; fallback total 10.4 s.
- Tables generated by `experiments/make_envelope_tables.py` into `revision/paperB-mdpi-source-r1/author/tables/`.

## R2.6 / R3 — extended turbulence, small done (36 conditions), large partial

Small: every noise/missing/flip cell significant (Holm) in 36/36 conditions; joint/sum-of-singles ratio 0.8-1.1 (additive); joint_all at 0.4 median 28.8 [21.1, 38.8]; bias (1 realisation/level, trend tests underpowered): RTR ~4.6-5.0 at 0.4, COR ~0.9, IDL ~2-2.6, TD ~3.4-4.5.

## R2.6 — Table 2 with 20 seeds per row (`results/audit/audit_summary.csv`, CIs added to analyze_audit)
Done 2026-11-03 16:01. Key changes vs 5 seeds: small 9600 dominated 630/2000 (32%), HV 0.81, best 77.35, churn 0.50 [0.47,0.54], Q 81.1; large 9600 1945/2000, HV 0.42, churn 0.90 [0.88,0.91]; large 150 HV 0.21 (reference point moved with more fronts). Weight grid recomputed with one reference per instance (all fronts + finest exact): shares 0.365->0.361 (small 150), 0.805->0.796, 0.160->0.159, 0.351->0.349.

Large: 36 conditions done 2026-11-03 16:09; all 8 noise/missing/flip cells significant in 36/36 conditions; joint/sum 0.78-1.08.
