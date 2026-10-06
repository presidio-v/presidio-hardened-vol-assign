# RQ1 reframe — seed-only floor and floor-adjusted results (2026-10-05)

Supersedes the "collapse at the first perturbation" reading in `rq1-results.md`.

## Why

The turbulence driver compares the decision made on perturbed inputs with the clean
decision **for the same solver seed**. Any change to the FIS cache decorrelates the GA
trajectory, so the design cannot separate input sensitivity from solver stochasticity.
Control: solve the *clean* instance under 20 seeds (`experiments/run_seed_floor.py`,
seed = 1000 + k·7919, k = 0..19; k < 8 / k < 6 are exactly the turbulence reps), commit
each front with the same rule, compare all 190 seed pairs.

Analysis: `experiments/analyze_floor.py` → `experiments/results/floor_analysis/{small,large}/`.
CIs for the floor are cluster-bootstrapped over seeds (pairs share seeds); per-level CIs
bootstrap over realisations (reps averaged). Trend test: one-sided Spearman of
`quality_loss` on level (0.05–0.4, realisation = unit), Holm within system and size.

**Consistency check (both sizes): PASS.** One manifest row (IDL noise, level 0.05,
realisation 0, rep 0) recomputed from the floor run's stored front matches the original
manifest bit-for-bit, so the floor and the turbulence data are directly comparable.

## 1. The "collapse" is the seed floor (solver non-identifiability)

| | small (5/150/50) | large (10/300/100) |
|---|---|---|
| pairwise churn, clean, different seeds | **0.834** [0.820, 0.847] | **0.939** [0.933, 0.945] |
| directed-set Jaccard | 0.31 [0.30, 0.33] | 0.25 [0.24, 0.26] |
| objective drift | 4.31 [3.30, 5.23] | 3.43 [2.87, 3.94] |
| clean quality (sum f1..f3), mean ± sd | 97.40 ± 2.05 (92.4–101.6) | 120.71 ± 1.32 (118.5–124.5) |
| fuzzy churn under turbulence − floor | −0.005 … +0.018 (all cells, all levels ≥ 0.05) | +0.002 … +0.021 |

Fuzzy turbulence churn sits **on** the floor everywhere. Exception: centre-occupancy
missingness on small sits **below** it at low levels (0.21 / 0.49 / 0.62 / 0.83 at
0.05–0.4). Mechanism verified exactly by replaying the realisations' RNG: churn is
**binary** — a realisation that blanks no centre leaves the FIS cache identical (churn 0);
one that blanks any centre jumps to the floor. Realisations with ≥1 centre blanked:
3/12, 7/12, 9/12, 12/12 → predicted churn 0.208 / 0.486 / 0.625 / 0.834 vs observed
0.210 / 0.487 / 0.622 / 0.832. So same-seed churn measures *whether the objective
landscape changed at all*, not how much — the cleanest demonstration that the "collapse"
is GA-trajectory decorrelation, not an input dose response.

Decision rule does not matter: fixed-nearest gives the same floor (small, seeds 0–4:
0.820 vs 0.818).

**Budget control (small, seeds 0–4):**

| gen | churn (eq-wt / fixed) | Jaccard | clean Q mean ± sd |
|---|---|---|---|
| 150 | 0.818 / 0.820 | 0.32 | 97.25 ± 0.58 |
| 300 | 0.826 / 0.832 | 0.31 | 95.08 ± 2.27 |
| 600 | 0.764 / 0.802 | 0.35 | 92.47 ± 2.55 |

Non-identifiability **persists to 4× budget** (floor ≈ 0.76–0.83; do *not* call it
structural until a plateau is shown — the GA is not converged: clean Q −4.8 at gen 600).
**Extended budget curve (small, seeds 0–4, equal-weight):**

| gen | churn | Jaccard | clean Q mean ± sd | s/solve (approx.) |
|---|---|---|---|---|
| 150 | 0.818 | 0.32 | 97.25 ± 0.58 | 1.3 |
| 300 | 0.826 | 0.31 | 95.08 ± 2.27 | 2.6 |
| 600 | 0.764 | 0.35 | 92.47 ± 2.55 | 4 |
| 1200 | 0.744 | 0.42 | 88.54 ± 1.94 | 8 |
| 2400 | 0.636 | 0.38 | 84.92 ± 1.45 | 16 |
| 4800 | 0.608 | 0.48 | 83.60 ± 2.01 | 35 |
| 9600 | 0.568 | 0.49 | 82.18 ± 1.36 | 69 (2 jobs in parallel) |

Quality approaches a plateau near ≈ 82 (gains −1.3 per doubling at the end); gen 150 is
≈ 15 points (≈ 7 clean-seed sd) short of it. Non-identifiability **shrinks with compute
but persists near convergence**: 57% of the allocation still differs between seeds at
gen 9600. This is a systems trade-off in its own right — latency vs. decision quality vs.
decision identifiability — and makes the operational budget a design parameter.

Consequence: the existing RQ1 matrix at gen 150 describes a badly under-converged system.
The step-2 re-run must use a near-converged operational budget (candidate gen 2400) and the
budget curve becomes a figure.

Spread/reference checks: 20-seed clean-Q sd 2.05 is genuine (range 92.4–101.6; no
degenerate fronts — 54–83 distinct fitness points per front), the n=5 sd of 0.58 was
small-sample. The turbulence reps' clean decisions are a typical draw (mean Q over reps
0–7 = 97.40 = 20-seed mean; large 120.91 vs 120.71), so the paired `quality_loss` null is
not biased by the reference.

## 2. Realised-quality harm (paired `quality_loss`, clean truth; positive = worse)

Fuzzy, level 0.4 (in units of clean-seed sd), trend p_Holm:

| cell | small | large |
|---|---|---|
| travel-duration noise | **+5.20 (2.54 sd)**, p<1e-4 | — |
| resource-time noise | +2.93 (1.43 sd), p<1e-4 | — |
| centre-occupancy noise | +2.74 (1.33 sd), p=2e-4 | +3.87 (2.94 sd), p<1e-6 |
| IDL noise | +2.00 (0.98 sd), p=1e-4 | +2.19 (1.67 sd), p<1e-7 |
| road-condition flip | +0.97 (0.47 sd), p=0.06 | +1.58 (1.20 sd), p=0.001 |
| possible-hazard flip | +0.92 (0.45 sd), p=0.08 | — |
| IDL missingness | +0.17 (0.08 sd), p=0.013 | +0.70 (0.53 sd), p=0.06 |
| centre-occ. missingness | −0.51, n.s. | — |

At level 0.05 fuzzy loss is ≈ 0 or negative in every cell. Harm appears from 0.2 and is
a real dose response for continuous inputs that feed the objectives.

Crisp baseline, level 0.4: centre-occupancy noise **+6.97 small / +10.00 large (7.6 sd)**,
road flip +3.31 / +4.72, travel +2.79, hazard +2.13 — all significant trends. Crisp is
*unhurt or improved* by IDL noise/missingness and resource-time noise (median imputation
and noise smooth its priority ranking).

## 3. Stability–quality trade-off (not "crisp wins") — partly definitional

Crisp clean quality: **114.52 small / 129.17 large** — worse than **every** one of the 20
fuzzy seeds (fuzzy max 101.6 / 124.5). Even the worst degraded fuzzy cell at 0.4
(≈ 97.4 + 5.2 = 102.6 small) stays better than the *clean* crisp decision. Crisp is
stable (churn 0.003–0.35 small, true dose response) but a much worse allocator.
Caveat (Fable): crisp is scored on the objectives the MOEA optimises, so "worse" is
partly definitional, and the comparison confounds *criteria* with *stochasticity*. Fix in
the step-2 re-run: add a deterministic **FIS-greedy** baseline (greedy on the fuzzy
cache's own scalarised cost) — same objective, zero stochasticity, so its churn under
turbulence is pure input sensitivity. Crisp-criteria greedy becomes the secondary foil.

## Reframed RQ1 story

Two separable failure modes for a digital AI allocation system under turbulence:
1. **Decision non-identifiability (solver):** 83–94% of the committed allocation changes
   between seeds on identical clean inputs — independent of turbulence, robust to budget
   and decision rule. A single run is not an accountable decision.
2. **Input-coupled quality loss (data):** above the floor, realised quality degrades with
   turbulence intensity, most for the continuous inputs the fuzzy objectives consume
   (travel, resource time, centre occupancy, IDL); the crisp rule is hurt by a different
   set (centre-side and transport categoricals).

Not yet done: per-objective decomposition of drift (needs perturbed fronts → step-2
re-run); LMM appendix (statsmodels not installed).

## 4. BLOCKER (2026-10-05): the model is separable — an exact deterministic method dominates the MOEA

Every objective is a mean over directed people of a per-person (ULPP_j) or
per-(person, centre) cache entry (TIL/CAIL_{j,i}); there is no capacity constraint and
centre occupancy is a static input (Paper A Eq. obj-cail: TCAIL_i = Σ_j d_ji·CAIL). So for
any weight vector the exact weighted-sum optimum is a greedy: best centre per person, then
the n_dir cheapest people.

| | small | large |
|---|---|---|
| exact min of raw sum f1+f2+f3 | **74.58** (39.98/15.44/19.17) | **85.18** (36.79/14.73/33.66) |
| MOEA best raw sum on any front point, gen 150 / 2400 / 9600 | 96.42 / 83.35 / 80.76 | 119.58 (gen 150) |
| front points dominated by that single greedy point (5 seeds) | 500/500, 280/500, 199/500 | 500/500 |
| exact supported front (21-step simplex weight sweep) HV | 181 pts, HV 29,840 | 197 pts, HV 19,683 |
| MOEA HV as share of exact, gen 150 / 2400 / 9600 | 38% / 70% / 80% | 14% (gen 150) |
| greedy centre loads | 38 of 50 people to one centre | 69 of 100 to one centre |

CAIL cannot penalise load concentration although Paper A's prose says it does
(paper.tex l.637–647, 753–766, Vignette 3 l.1438–1446). Step 2 is on hold pending a pivot
decision (Fable consulted).

## 5. Model fix built and tested; it does not create a regime for the MOEA (2026-10-05)

Code (all brute-force/equivalence tested, 975+ tests pass): `allocation/fast_fis.py` (compiled
FIS3 = skfuzzy to 1e-9; 43× faster scalar, batched path), `allocation/load_coupling.py`
(COR'_i(l) = min(100, COR_i + 100·l/cap_i), cap_i = ceil(κ·n_dir/n), `FISCache.cail_load`
read by the shared `evaluate_pairs`), `allocation/exact_mip.py` (HiGHS, only load levels
binary, saturated levels merged), `baselines.exact_weighted_sum_pairs` (separable model).

FIS3 occupancy response: CAIL rises only 17–20 points over COR 0→100 and saturates; 368/777
(RDR, TD) curves have small dips (worst −0.57; summed ≤ 3.15). Published rule base, not patched.

κ sweep, equal weights, MIP optimal (gap ≤ 1e-4) in 0.4–2 s at both sizes:

| | small | large |
|---|---|---|
| static-greedy decision vs coupled optimum | +0.2…+1.5% worse, overlap 0.86–0.96 | +10.9…+12.0% worse, overlap 0.14–0.21 |
| top centre loads in coupled optimum | 32–41 of 50 | ~42 / 34 / 10 of 100 |
| overload beyond free capacity (soft) | 9–40 people | 69–92 people |
| κ = ∞ | reproduces 74.58 exactly | reproduces 85.18 |

Scaling probe (κ = 2, soft): 600 people × 20 centres → caches 115 s, MIP 10.6 s, optimal.

Exact-decider turbulence (small, 30 realisations): clean dose response, churn 0.01–0.11 at
level 0.05 → 0.03–0.42 at 0.4; quality loss +0.1 (centre-occ missingness) … +8.3 (travel
noise) at 0.4. Worst degraded exact decision (≈ 82.9) still beats the MOEA's *clean* decision
at gen 150 (97.4) and matches gen 9600 (82.2).

Fable verdict: the honest paper is the **audit** (protocol: separability check → exact
dominance → seed floor → input sensitivity on the exact decider → budget curve), with the
coupled + hard-capacity model as the repaired variant solved exactly; MIP scaling figure as the
deployment envelope; MOEA guardrail re-run cancelled (one appendix paragraph from existing floor
data).

## 6. Paper A's published runs through the same dominance check (2026-10-05)

Stored fronts: `presidio-hardened-vol-asssign/experiments/results/h1_h2_h4/<size>_<k>obj_<alg>_rep*/pareto_*.csv`
(30 reps each, pop 100 / gen 200, Paper A protocol). Exact greedy computed with the fast cache
on the same instances.

| size | model | exact sum | NSGA-II dominated | NRGA dominated | NSGA-III dominated | best MOEA sum (any alg.) |
|---|---|---|---|---|---|---|
| small | 4-obj | 112.03 | 2950/3000 | 2883/2973 | 2948/3000 | 129.75 |
| small | 3-obj | 74.58 | 2976/2994 | 2838/2838 | 3000/3000 | 84.20 |
| medium | 4-obj | 105.31 | 2996/3000 | 2997/2997 | 2996/3000 | 136.71 |
| medium | 3-obj | 68.76 | 3000/3000 | 2908/2908 | 3000/3000 | 94.71 |
| large | 4-obj | 121.87 | 2997/3000 | 2994/2994 | 2993/3000 | 158.91 |
| large | 3-obj | 85.18 | 3000/3000 | 2781/2781 | 3000/3000 | 116.24 |

One deterministic greedy point dominates 97–100% of every published front point, for every
algorithm, size and model. Paper A's *relative* algorithm comparisons are unaffected in kind,
but its fronts are far from the true front, and its HV values are not near-optimal.
