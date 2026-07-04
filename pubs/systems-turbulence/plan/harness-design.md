# Paper B — experiment harness design

Design for the code that produces RQ1–RQ3 evidence (`explore/hypothesis-rq.md`). Evidence-first:
this locks the perturbation model, metrics, and module layout **before** any code. Open
decisions for review are in §12.

## 1. What is perturbed (RQ1) — and why it is distinct

Turbulence perturbs the **raw input fields** of `AllocationProblem`, not the elicitation weights
(that is Paper A's `sensitivity` module, which sweeps the 6 `Weights`). The perturbable inputs
(from `allocation/models.py`):

| Field | Type | Range | Modes |
|---|---|---|---|
| `Person.age` | continuous | years | noise, bias |
| `Person.infrastructure_damage_level` (IDL) | continuous | [0,100] | noise, missingness, bias |
| `Person.resource_time_remaining` (RTR) | continuous | hours | noise, missingness, bias |
| `ReliefCenter.center_occupancy_rate` (COR) | continuous | [0,100] | noise, missingness, bias |
| `ReliefCenter.resource_depletion_rate` (RDR) | continuous | [0,100] | noise, missingness, bias |
| `TravelInfo.travel_duration` (TD) | continuous | minutes | noise, missingness, bias |
| `disability/injury/living_status`, `road_condition`, `possible_hazard` | categorical | discrete | flip, missingness |

Derived scores (VS, RWS) are recomputed downstream, so perturbing inputs propagates realistically.

## 2. Perturbation model

Three modes, applied **one field/mode at a time (OFAT)** plus one **combined "storm"** level:

- **Noise:** additive Gaussian, σ = fraction of the field's plausible range; clip to valid range.
  Sweep σ ∈ {0, 0.05, 0.10, 0.20, 0.40}.
- **Missingness:** with rate ρ, blank the field then impute (continuous → instance median;
  categorical → mode). Sweep ρ ∈ {0, 0.05, 0.10, 0.20}. (Imputation rule is documented and fixed.)
- **Bias:** systematic directional shift δ on one field (e.g., IDL under-reported). Sweep δ over
  a small signed set. Directional (tests systematic, not just random, error).
- **Categorical flip:** relabel a discrete field to a random other level with prob ρ.

**Seeding.** One fixed base instance per size (seed `S0`). For each (mode, level), draw
`R_turb = 20` turbulence realisations (seeds `S0 + k`) to estimate variance. Deterministic and
recorded, mirroring the reproducibility discipline already in the repo.

## 3. Decision extraction (the key methodological choice)

The MOEA returns a **front**, not one decision; "decision stability" needs a single decision per
run. Proposal: extract a **canonical decision** = the front solution minimising a fixed reference
scalarisation (equal-weight sum of normalised objectives). Decode it to a concrete allocation.
Rationale: a deployed system must commit to one allocation; the equal-weight knee is a neutral,
reproducible committal rule. (Alternative: nearest-to-ideal / knee point — see §12.1.)

## 4. Decision-stability metrics (perturbed vs clean)

Decisions are **made on perturbed inputs** but **evaluated on the clean ground-truth inputs**
(the realised consequence of deciding on bad data):

- **Realised-objective drift** — Δ of the true objective vector (ULPP, TIL, CAIL for the 3-obj
  model) between the clean-input canonical decision and the perturbed-input canonical decision,
  both scored on clean inputs. Primary quality-degradation metric.
- **Allocation churn** — fraction of directed people whose assigned centre differs from the
  clean decision.
- **Centre-load rank stability** — Spearman ρ between clean and perturbed per-centre loads.
- **Front-level HV drop** (on true objectives) as a secondary, decision-agnostic view.

Report each vs turbulence level as a **degradation slope** and **variance**, fuzzy-MOEA vs crisp.

## 5. Crisp baseline (RQ1 comparator) — NEW code

The `allocation/` lineage has **no** baseline allocator (confirmed). Build a deterministic
**crisp greedy allocator** (`allocation/baselines.py`): scores each (person, centre) with a crisp
weighted sum of the same criteria (no FIS), assigns greedily under the `n_dir` cap. Same inputs,
same perturbation, same metrics → isolates the contribution of the fuzzy front-end to graceful
degradation. (Distinct from `main`'s incompatible `baselines.py`.)

## 6. Reproducibility tooling (RQ2) — NEW code

The lineage has **no** front-signature/REP for allocation fronts (top-level `repro.py` targets the
other model). Add `allocation/repro.py`: `allocation_front_signature(front)` = SHA-256 over
rounded, sorted (objective-vector + allocation) tuples; `rep_score(signatures)` = 1.0 iff all
identical. Cross-environment matrix: extend CI (already 3.10/3.11/3.12) with an OS dimension;
`experiments/run_reproducibility.py` emits the signature per (env, seed) and the REP verdict.
Audit-trail mapping (manifest, signature, `pip-audit`, SECURITY) tabulated from existing artefacts.

## 7. Operability profiler (RQ3) — NEW code

`experiments/profile_operability.py`: wrap `solve()` with wall-clock timing and peak memory
(`tracemalloc` + `resource.getrusage`), per (size, algorithm), `R=30`. Report latency, peak RSS,
throughput (decisions/hour), and scaling vs n_people/n_centers.

## 8. Module layout

- New: `allocation/turbulence.py` (perturbation model + `apply_turbulence(problem, spec, rng)`),
  `allocation/baselines.py` (crisp greedy), `allocation/repro.py` (front signature + REP).
- New drivers: `experiments/run_turbulence.py`, `experiments/run_reproducibility.py`,
  `experiments/profile_operability.py` — each writes a manifest CSV under
  `experiments/results/{turbulence,repro,operability}/`, mirroring `run_weight_sweep`'s pattern.
- Reused as-is: `solve()`, `AllocationConfig`, the sweep/manifest plumbing pattern in
  `allocation/sensitivity.py`, `pip-audit`/CI.
- CLI: optional thin wrappers (`pva alloc-turbulence`); drivers suffice for the paper. Keep minimal.

## 9. Experiment matrix

- **Sizes:** small (5 centres / 150 people) and large (10 / 300) — the 3-objective relief model.
- **Algorithms:** NSGA-II, NRGA (not NSGA-III — that headline is Paper A's).
- **Turbulence:** the §2 modes × levels, OFAT + combined; `R_turb = 20` realisations each.
- **Solver reps:** 30 seeds per cell (as in Paper A) for solver-stochastic variance.
- Pop 100, gen 200 (defaults).

## 10. Statistical analysis

- RQ1: Wilcoxon rank-sum on per-instance degradation **slopes**, fuzzy-MOEA vs crisp, per mode;
  report medians + IQR; effect direction. (Mirrors Paper A's non-parametric approach, new metric.)
- RQ2: REP ∈ {0,1} across the env×seed matrix; report any divergence source.
- RQ3: latency/memory distributions; report medians and 95th percentile.

## 11. Outputs

Manifest CSVs (per driver) + figures: (a) degradation curves fuzzy vs crisp per mode,
(b) REP table, (c) operability envelope. Figures generated by a `make_figures`-style script in
`experiments/`, kept in `pubs/systems-turbulence/figures/` (git-ignored `*.pdf`, like Paper A).

## 12. Open decisions for review (before coding)

1. **Canonical-decision rule** (§3): equal-weight scalarisation (proposed) vs knee-point vs
   nearest-to-ideal. Determines what "decision stability" means — needs your sign-off.
2. **Crisp baseline definition** (§5): confirm "crisp weighted-sum greedy on the same criteria"
   is the right foil (vs a naive nearest-centre or random baseline as a second, weaker floor).
3. **Turbulence ranges** (§2): are the σ/ρ/δ sweep ranges realistic for the relief setting, or
   should they be calibrated to a cited data-quality source?
4. **Test plan:** unit tests for `turbulence`/`baselines`/`repro` (determinism, range-clipping,
   signature stability) — assumed, following repo convention (ruff + pytest gate).
