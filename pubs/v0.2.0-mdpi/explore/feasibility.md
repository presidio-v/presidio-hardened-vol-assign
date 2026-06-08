---
provisional: true
phase: I-D
project: presidio-hardened-vol-assign
target-venue: Applied Sciences (MDPI), SI "Innovations in Supply Chain Resilience"
authors: Rabiei, Arias-Aranda, Stantchev
---

## Methodology per Hypothesis

Methodology codes: **EXPER** = controlled technical experiment; **TOOL** =
requires building or extending a tool; **MIXED** = combination.

| H-ID | Methodology | Data / Resource | Effort | Blockers | Mitigation |
|------|-------------|-----------------|--------|----------|------------|
| H1 | EXPER + analysis | `pva` 4-obj and 3-obj modes on identical instances and seeds; Spearman ρ between TRD/RPD across 4-obj front; projection-dominance counter | M | 4-obj HV computation; clean 3-obj-recovery projection that respects ATRes's RWS weighting | pymoo's WFG-based HV for ≥3D; write a `pareto_project_4_to_3.py` helper that uses ATRes's RWS weighting analytically |
| H2a (HV) | EXPER | `pva` with NSGA-III (DEAP `selNSGA3`); 30 reps × 3 sizes × 3 algorithms | M | NSGA-III with our custom chromosome encoding may misbehave; reference-point set choice must be defensible | Validate NSGA-III on DTLZ2 (4-obj) benchmark before running on our model; default to Das-Dennis p=4 (35 ref points) |
| H2b (NNS, SM) | EXPER | Same runs as H2a; t-test under normality, Mann-Whitney U otherwise | S | None beyond H2a | — |
| H2c (CPU) | EXPER | Same runs as H2a; isolated machine, no other workloads | M | CPU jitter on shared hardware | Pin to single physical core; disable turbo; warmup runs; report median + IQR |
| H3a (rule-base) | EXPER | `pva` rule-base override flag; 66 single-rule-deletion variants × 30 reps on medium-size | M-L | Compute time (1,980 runs) | Parallelize across cores; ~12-hour overnight run on workstation |
| H3b (weight LHS) | EXPER | `pva` weight override; LHS via `scipy.stats.qmc.LatinHypercube`; n=100 samples × 30 reps on medium-size | M-L | Compute time (3,000 runs) | Same; co-located with H3a runs |
| H4 (replication) | EXPER (sub-product) | Subset of H2 large-size runs comparing NSGA-II vs. NRGA | S | None | — |

## Algorithm and parameter decisions (deferring from I-A)

### FIS₂ᵦ input dimension — *single-input on TD alone*

**Decision:** FIS₂ᵦ takes Travel Duration (TD) only. Three linguistic values
(Short, Moderate, Long), three rules, output is Rapidity Deficit (RPD) on a
Very-Low → Very-High scale.

**Reason:** Rapidity in Bruneau et al.'s 4R framework is a context-free
"how fast" property of the response — it should not be modulated by who is
being directed. Adding RTR (resource time remaining) as a second input would
double-count: RTR already enters FIS₁ to set ULPP, and re-using it in FIS₂ᵦ
mixes Resourcefulness into Rapidity, undermining the 4R-objective bijection.

**Trade-off accepted:** less "context-aware" rapidity. Defensible because
the resource scarcity context is captured by ULPP's Resourcefulness axis,
and the practitioner sees both axes simultaneously on the Pareto front.

### NSGA-III reference points — *Das-Dennis with p=4*

**Decision:** Das-Dennis structured reference points with p=4 divisions,
yielding C(7,3) = 35 points uniformly distributed over the 4D simplex.

**Reason:** Standard and well-documented in Deb & Jain (2014); easy to
reproduce and easy to defend in §5; matches the small-budget humanitarian
problem scale (population sizes around 100–200 chromosomes — no need for
denser reference grids).

**Trade-off accepted:** may underweight boundary regions. If H2b shows
NSGA-III SM significantly worse than NSGA-II at any size, run a two-layer
reference-point ablation as supplementary material rather than reframing
the main results.

## Architecture decision (locked 2026-05-09)

**`pva` v0.1.0 implements Rabiei et al. ESWA 2023** (volunteer-to-vacancy
assignment, two objectives), not the ATRes 2026 people-to-relief-center
allocation problem. The MDPI extension adds the ATRes/MDPI model as a new
**`presidio_vol_assign.allocation` sub-package** alongside the unchanged
volunteer-assignment code, exposed via a `pva allocate` subcommand. Tool
name remains `presidio-hardened-vol-assign`; rename deferred to a
hypothetical v1.0.0.

## Fast path — `pva` v0.2.0 deliverables (status as of 2026-05-09)

Concrete code-level dependencies for Phase III experiments:

1. **`allocation/models.py`** — Person, ReliefCenter, TravelInfo, problem,
   config, solution, front, metrics dataclasses + status-field enums
   (DisabilityStatus, InjuryLevel, LivingStatus, RoadCondition, HazardLevel)
   with score mappings per ATRes Table 1. ✔ DONE
2. **`allocation/fis.py`** — five FIS systems: FIS₁ (ULPP, 27 rules),
   FIS₂ legacy (TIL, 9 rules) for `--objectives 3`, FIS₂ₐ (TRD, 9 rules)
   and FIS₂ᵦ (RPD, 3 rules) for `--objectives 4`, FIS₃ (CAIL, 27 rules);
   plus `compute_vs` and `compute_rws` helpers. **ATRes Eq. (5) sign
   correction applied** (see `feedback_atres_rws_correction.md`). ✔ DONE
3. **`allocation/validation.py`** — schema, range, set-membership, and
   cross-file checks for people.csv + centers.csv + travel.csv plus
   `n_dir` constraint per ATRes Eq. (15). Reuses CSV primitives from the
   existing volunteer-assignment validator. ✔ DONE
4. **`allocation/solvers.py`** — NSGA-II, NRGA, and **NSGA-III** on a
   flattened 2*n_dir chromosome (partial-permutation persons + real-valued
   centers). Custom uniform-with-repair crossover and per-gene mutation
   to preserve the encoding invariants. NSGA-III uses Das-Dennis p=4
   reference points via `tools.uniform_reference_points`. FIS pre-cache
   (numpy arrays) makes evaluation O(n_dir) lookups in the GA loop. ✔ DONE
5. **`allocation/metrics.py`** — NNS, MID, SM (ATRes Eq. 19 form),
   **HV via pymoo's WFG indicator** for any dimensionality ≥ 2.
   Reference-point default (100, …, 100) per FIS output universe;
   override-able for HV comparability across runs. ✔ DONE
6. **`allocation/writers.py`** — `pareto_alloc_<solver>_<obj>obj_<ts>.csv`
   + `allocations_alloc_*.csv` + `metrics_alloc_*.json`; reader infers
   3- or 4-obj formulation from CSV header. ✔ DONE
7. **`pva allocate` and `pva alloc-metrics` CLI commands** — full Typer
   surface mirroring the AllocationConfig parameter set, including
   per-weight overrides for sensitivity runs. ✔ DONE
8. **Tests** — 79 new tests covering models, FIS sign-correctness,
   validation paths, solver determinism, HV correctness, writer
   round-trips, and CLI integration. Coverage on allocation modules:
   90–99%. ✔ DONE

**Week 3 deliverables (complete as of 2026-05-09):**

9.  **Rule-base override runtime** for H3a. Rule tables exposed as module
    constants (`FIS1_RULES`, `FIS2_TIL_RULES`, `FIS2A_TRD_RULES`,
    `FIS2B_RPD_RULES`, `FIS3_RULES`); `build_fis_with_drops(name, indices)`
    rebuilds a named FIS with selected rules removed; `fis_overrides({...})`
    context manager swaps in overridden systems for the duration of an
    enclosed `solve()`. CLI: `--fis-rules <path.json>` on `pva allocate`,
    JSON spec validated via `load_fis_rules_spec`. ✔ DONE
10. **Weight-LHS sensitivity sweeper** for H3b. New module
    `allocation/sensitivity.py` with `lhs_weight_samples` (Latin-Hypercube
    over the six VS/RWS weights via `scipy.stats.qmc.LatinHypercube`,
    scaled to baseline·(1±bound) and clipped to [0, 1]) and
    `run_weight_sweep` (one solver run per sample, per-sample subdirs,
    top-level `weight_sweep_manifest.csv` aggregating six weights, run
    metadata, and per-objective min/mean/HV). New CLI:
    `pva allocate-weight-sweep --n-samples N --bound B --lhs-seed S`. ✔ DONE
11. **4-obj → 3-obj Pareto projection helper** for H1. New module
    `allocation/projection.py` with `project_pareto_4_to_3(front, problem,
    weights)` (recomputes TIL per allocation via the ATRes FIS2_TIL
    pathway; tags solutions whose 3-obj projection is strictly dominated
    by another), `spearman_trd_rpd(front)` (Spearman ρ between TRD and
    RPD across the 4-obj front), and `summarise_h1(...)` returning an
    `H1Summary` with the dominance fraction, ρ, and the joint
    confirmation verdict per the operationalisation in
    `hypothesis-rq.md`. ✔ DONE
12. **Tests** — 42 new tests covering rule constants, override registry,
    JSON spec loader, LHS sample bounds and determinism, sweep manifest
    schema (3-obj and 4-obj), Spearman correlation edge cases (singleton,
    constant axis, anti-correlation), projection round-trip, dominance
    flags, and CLI integration for `--fis-rules` and
    `allocate-weight-sweep`. Total suite now 260 tests, 95.83% coverage. ✔ DONE

All Week 2 + Week 3 deliverables are complete; experiment runs (Weeks 4–5)
unblocked.

## Compute budget

Per-run wall time (laptop, current `pva` benchmarks): roughly 5–15 s for
small, 30–60 s for medium, 90–180 s for large.

| Experiment | Runs | Wall time (sequential) | With 8-way parallelism |
|---|---|---|---|
| H1 (3-obj vs. 4-obj × 3 sizes × 3 algos × 30 reps) | 540 | ~10–20 h | ~1.5–2.5 h |
| H2 (NSGA-III + comparators) | already in H1 envelope if combined | — | — |
| H3a (66 rule deletions × 30 reps × medium) | 1,980 | ~16–33 h | ~2–4 h |
| H3b (100 LHS × 30 reps × medium) | 3,000 | ~25–50 h | ~3–6 h |
| **Total** | ~5,500 | ~50–100 h | ~7–13 h |

Manageable on a single workstation across two overnight runs. No cloud or
HPC required.

## Open dependencies (must clear before Phase III)

1. **Co-author alignment** (user-owned). Rabiei and Arias-Aranda must
   confirm: (a) extended MDPI version with ATRes citation on title page,
   (b) the 4-objective split and resilience reframing, (c) author order,
   (d) Zenodo DOI release of `pva` v0.2.0 as the citable artifact.
   *Critical path — blocks Phase II planning sign-off.*
2. **ATRes copyright clearance.** Confirm with the ATRes / Elsevier author
   agreement (assuming Elsevier-style transfer) that an extended journal
   version with >50% new content is permitted. Standard Elsevier policy
   permits this; verify the specific clause in the ATRes acceptance letter.
3. **MDPI Special Issue scope confirmation.** Confirm with guest editor
   Leonardo Agnusdei that the resilience reframing (4R + humanitarian
   last-mile) fits the SI scope. The invitation language ("Innovations in
   Supply Chain Resilience") strongly suggests yes; one short email closes
   the question.
4. **Zenodo deposit pre-flight.** Confirm `presidio-v` GitHub org → Zenodo
   integration is enabled so the v0.2.0 tag automatically mints a DOI.

## Fast-path "first cut" candidates (Week 2)

If schedule pressure forces triage, these subsets of work still produce a
publishable paper, in priority order:

- **A — Critical:** 4-obj split, NSGA-III, three problem sizes, H1+H2+H4
  on those (no sensitivity). Reframed §1–2 + new §3.5 + new §5. Submittable
  but weaker. Drops H3 entirely — replace §7 with a "limitations and future
  work" item naming sensitivity analysis as outstanding.
- **B — Strong:** A + H3b (weight sensitivity only, no rule-base
  perturbation). Adds half of §7. Most defensible compromise.
- **C — Full:** A + B + H3a (rule-base sensitivity). The originally scoped
  paper.

Recommended path: **C** (full); fall back to **B** at end of Week 4 if H1
or H2 runs over budget.

## Quality check (per design pattern)

- ☑ Every H has a named methodology
- ☑ Blockers are explicit (compute, copyright, co-author alignment)
- ☑ Fast-path identified (three triage tiers)
- ☑ Effort estimates present (M / S / L; total compute budget)
