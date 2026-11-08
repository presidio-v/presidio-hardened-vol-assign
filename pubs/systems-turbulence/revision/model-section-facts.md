# Model section: facts and sources

Draft: `model-section-draft.tex`. Code: `/Users/vstantch/projects/pva-experiments` at HEAD `ecbd49a`
(working tree has one uncommitted change to `experiments/generate_instances.py`, see F-SALT).
Paths below are relative to that repo unless marked `paper:` (= `.../paperB-mdpi-source-r1/author/paper.tex`,
line numbers as of 2026-11-03; the file was being edited during this work, so line numbers may drift).

## A. Discrepancies and problems (read first)

| ID | Issue | Evidence | Action |
|---|---|---|---|
| **F-ULPP** | ULPP_j increases with VS, with IDL and with short RTR (e.g. high/severe/short -> very_high; low/minor/long -> very_low). f_1 is the *mean ULPP over directed people*, minimised. The model therefore prefers to direct the people with the **lowest** ULPP, i.e. the least vulnerable/least urgent. Checked on the small instance (in-memory run, no files written): Spearman(ULPP, crisp priority) = 0.61; exact equal-weight optimum directs people with mean crisp priority 0.31 vs 0.39 for all people and 0.53 for the crisp rule; exact objectives (39.98, 15.44, 19.17) = 74.58, crisp (72.09, 16.25, 26.17) = 114.52, so 32.1 of the 39.9-unit crisp gap is f_1. | `src/presidio_vol_assign/allocation/fis.py:154-180`, `solvers.py:215`, `baselines.py:33-49,79-83` | Author decision. The draft states the rule direction (two example rules) and leaves the consequence in a `% CHECK` comment. Reviewers reading the new section can infer it. It also explains most of the "crisp is 40-44 units worse" result (paper:628, paper:659-661). |
| **F-NRGA** | The paper cites NRGA as Al Jadaan et al. (paper:180, `\cite{aljadaan2008nrga}`). The `nrga` solver used in every run is `sel_nrga`: NSGA-II-style front filling with **uniform random truncation of the last front, no crowding, no ranked roulette**. The canonical ranked-roulette NRGA exists as `sel_nrga_ranked` but is not reachable from the allocation solver. | `src/presidio_vol_assign/engine.py:59-75` (used), `engine.py:78-110` (not used), `src/presidio_vol_assign/allocation/solvers.py:69,559-560` | Draft describes it as implemented ("NRGA, as implemented, ..."). Consider a sentence that it is a simplified NRGA. |
| **F-SALT** | The committed reference instances were generated with salt `abs(hash(name)) % (2**32-1)`; Python salts `str` hashes per process, so they **cannot be regenerated**. The working tree (uncommitted) switches to CRC-32; regenerating with seed 42 now gives different CSVs for all three sizes (verified: all 9 files differ). Distributions are unchanged (diff touches only the salt). `experiments/instances/small/INSTANCE.md` still says `salted with hash('small')`. `run_envelope.py:5-9` confirms the r1 envelope (run_scaling) 60 s and 300 s rows were different instances. README:16 says scaling instances "regenerate bit for bit". | `git diff HEAD -- experiments/generate_instances.py` (lines 132-136 new), `experiments/instances/small/INSTANCE.md`, `experiments/run_envelope.py:5-9`, `experiments/README.md:16` | Draft says the reference instances are fixed archived files identified by SHA-256 and does **not** claim they regenerate from the seed. Check paper:812 ("every run can be repeated bit for bit") and the data statement: true for runs on the archived files, not for instance generation. |
| F-FLOOR | Paper S4 (paper:428-430): for the stochastic system "we report it above that floor". In code only **churn** has the floor subtracted (`churn_minus_floor`). The quality harm is *not* floor-subtracted: it is paired by seed, tested against zero, and scaled by the clean-seed s.d. (`loss_in_clean_sd`). The analyze_floor docstring claims a 95% CI for "churn minus the floor"; the code computes the CI of the churn mean only, not of the difference. | `experiments/analyze_floor.py:12-17,164-165,194,215-231` | Draft gives both formulas exactly (eq:above-floor). Reword paper:429 if it implies a subtracted quality floor. |
| F-BIAS | Paper:421 "in one of four modes" lists bias. Bias exists in the harness but **no experiment cell uses it**; cells are noise (4), missingness (2), flip (2). | `allocation/turbulence.py:42-45,106-107`; `experiments/run_turbulence_full.sh` CELLS; `experiments/run_turbulence_exact.py:39-48` | Say "three of the harness's four modes are used", or drop bias. |
| F-RQ3 | Paper:666 lists significant MOEA harm trends for "travel duration, resource time, damage level and centre-occupancy noise". Small, fuzzy, Holm: also **IDL missingness p=0.013**; large: also **road-condition flip p=0.0014**. Large fuzzy turbulence was run for only 4 of 8 cells (COR noise, IDL noise, IDL missingness, road flip): TD and RTR noise have no large-instance MOEA result. | `experiments/results/floor_analysis/{small,large}/quality_trend.csv`; `ls experiments/results/turbulence/large/` | Outside the model section; fix the RQ3 sentence. |
| F-ALGS | Paper:305 (Sec. 3.1) "NSGA-II or NRGA search the allocations", but NSGA-III is used in RQ1/RQ2 controls (paper:486-487, 511-512, 573). | — | Add NSGA-III to Sec. 3.1. |
| F-RTR | Turbulence clips RTR to a "plausible range" [0, 72] h, the crisp rule normalises by 72 h, but the model's valid RTR range is [0, 48] (validation, FIS universe). Paper:422 says values are "clipped to its valid range". | `turbulence.py:54`, `baselines.py:29,40-41`, `validation.py:208`, `fis.py:57,599` | Draft states the ranges and that RTR > 48 enters FIS_1 at its top. |
| F-RWS | RWS in code is `1 - (wRC*RCS + wPH*PHS)/(wRC+wPH)`, sign-inverted relative to the prior paper's Eq. (5) as printed (code docstring says the literal formula is a danger score). | `fis.py:558-584` | Draft gives the code form. Make sure nobody "corrects" it back to the prior paper. |
| F-GEN-DOC | Generator docstring says travel durations are "linked to a synthetic distance grid" and converted "at 60 km/h". Code draws TD per pair i.i.d. as `clip(100*Beta(1.5,4), 2, 180)`; no grid, no spatial structure; the 180 cap never binds (max 100; observed max 81.28 small, 88.73 large). | `experiments/generate_instances.py:7-8,18-22,110-112`; committed CSVs | Draft states the i.i.d. form. |
| F-EVALS | "240,000 evaluations" = mu x generations (200x1200, 400x600, 100x2400). The loop re-evaluates only modified offspring (expected about 1-0.3*0.8 = 0.76 of lambda per generation), plus the initial mu, so actual objective evaluations are fewer. | `solvers.py:387-419`; `experiments/results/seed_floor/small_evals240k_pop{200,400}/meta.json` | Draft defines "evaluations" as mu x generations in the table caption. |
| F-ENV | `solve()` supports a wall-clock deadline and seeded initial populations, but the r1 envelope (`run_scaling.py`) uses calibrated generations: `max(20, int(T / per_gen))`. The revision driver `run_envelope.py` uses a deadline (generation bound 1e9) and an NSGA-II seeded with the fallback + 21 weight-sweep optima. | `run_scaling.py:46-49,138-144`; `run_envelope.py:10-11,52-57,178-186` | Draft table row describes run_scaling; a `% CHECK` comment gives the replacement row if the revision adopts run_envelope. |
| F-MUT | Mutation quirk worth knowing: the displaced person is appended to the pool and popped by the next firing gene, so a mutation event changes the directed set by at most one person; further firing genes only re-pair already-directed people with other centre genes. | `solvers.py:325-336` | Stated in the draft as coded. |
| F-XO | Repair compacts the person list after dropping duplicates and appends random unused persons at the end, so persons after a duplicate shift position and get a different centre gene. | `solvers.py:280-292` | Stated in the draft ("the list closes up"). |

## B. Unverified items

- **Alternate instances (seed 43/44)**: the run metadata point at `experiments/results/alt_instances/seed{43,44}/<size>`, which does not exist in the tree; I could not confirm they came from `generate_instances.py --seed 43/44` or which salt was used. Draft only says "two further instances per size". Sources: `experiments/results/seed_floor/small_gen150_inst43/meta.json` (`instances` field), `experiments/README.md:25`.
- **Zero-firing fallback** (returns 50): coded, but I did not check whether it ever triggers. With full-coverage MFs and complete grids it should not. `fis.py:519-532`.
- **Large-instance MOEA turbulence counts** (R=8, S=6) read from one manifest (`results/turbulence/large/center_occupancy_rate_noise/turbulence_manifest.csv`); the other three large cells not individually checked. Small (R=12, S=8) checked on `travel_duration_noise` and matches `run_turbulence_full.sh:12-13`.
- Budget-curve seed counts above 150 generations (5) read from `meta.json` of each `seed_floor/*_gen{300..9600}`; 20 seeds at 150 from `small_gen150`/`large_gen150`.

## C. Facts in the draft and their sources

### Decision and inputs
| Claim | Source |
|---|---|
| 0 < n_dir < m enforced | `src/presidio_vol_assign/allocation/validation.py:116-121`; `models.py:198-201` |
| Exactly n_dir distinct directed people, one centre each (chromosome guarantees it) | `solvers.py:7-16,173-191,256-260` |
| No capacity constraint in the published model | `baselines.py:144-155` (docstring: "no capacity coupling"); `load_coupling.py:3-7` |
| DS scores none/minor/severe = 0/0.5/1 | `models.py:22-31` |
| IL scores 0/0.25/0.5/0.75/1 | `models.py:34-51` |
| LS scores with_support/alone = 0/1 | `models.py:54-62` |
| RCS scores clear/partial/blocked = 0.1/0.5/1.0 | `models.py:65-74` |
| PHS scores 0.05/0.25/0.5/0.75/1.0 | `models.py:77-94` |
| Age score piecewise (25, 60, 90) | `models.py:141-151` |
| COR_i: % of capacity in use; RDR_i: % of resources consumed per hour; TD minutes; RTR hours; IDL [0,100] | `models.py:127-130,160-163,182` |
| Domains: age [0,120], IDL [0,100], RTR [0,48], COR/RDR [0,100], TD [0,180] | `validation.py:168-169,199,207-208,259` |
| VS = equal-weight mean of AS, DS, IL, LS | `fis.py:540-555`; weights all 1.0 `models.py:237-242` |
| RWS = 1 - mean(RCS, PHS) | `fis.py:558-584`; `models.py:241-242` |
| Generator distributions (age U(5,90) 1 dp; DS .70/.20/.10; IL .40/.25/.20/.10/.05; LS .65/.35; IDL 100·Beta(2.5,2); RTR U(1,48); COR U(20,90); RDR U(10,80); TD clip(100·Beta(1.5,4),2,180); RCS .45/.40/.15; PHS .30/.30/.25/.10/.05) | `experiments/generate_instances.py:60-73,76-87,90-97,100-124` |
| Rounding (age 1 dp, others 2 dp) | `generate_instances.py:80,84-85,94-95,120` |

### Fuzzy systems
| Claim | Source |
|---|---|
| Universes and sampling (101 points; TD 181) | `fis.py:55-64` |
| Input MF breakpoints: VS, RWS (0,.25,.5,.75,1); IDL, COR, RDR (0,25,50,75,100); RTR (0,12,24,36,48); TD (0,30,60,90,180); RCS (0,.2,.5,.8,1); PHS low (0,0,.2,.4), mod (.2,.5,.8), high (.6,.8,1,1) | `fis.py:74-135` |
| Output five terms (0,0,10,30) ... (70,90,100,100) | `fis.py:138-144` |
| Rule counts 27 / 9 / 9 / 3 / 27; complete grids | `fis.py:152-249,253-259` |
| Example rules FIS1 (high,severe,short->very_high; low,minor,long->very_low) | `fis.py:156,178` |
| Example rules FIS3 (low,slow,short->very_low; high,rapid,*->very_high) | `fis.py:222,246-248` |
| FIS1 inputs (VS, IDL, RTR); FIS2 (TD, RWS); FIS2a (RCS, PHS); FIS2b (TD); FIS3 (COR, RDR, TD) | `fis.py:276-365` |
| AND = min, OR = max (skfuzzy defaults); accumulation = max; defuzzification = centroid | `.venv/.../skfuzzy/control/term.py:122`, `rule.py:53`, `antecedent_consequent.py:84,90`; replay in `allocation/fast_fis.py:1-18` |
| Inputs clipped to [eps, max-eps], eps = 1e-4; output clipped to [0,100]; zero-firing returns 50 | `fis.py:66,519-532,592-645` |
| Compiled evaluator pinned to skfuzzy by tests (used by exact/turbulence-exact/envelope) | `fast_fis.py:16-18`; `run_turbulence_exact.py:71`; paper:Software section |

### Objectives
| Claim | Source |
|---|---|
| ULPP_j = FIS1(VS_j, IDL_j, RTR_j) | `solvers.py:139-143` |
| TIL_{j,i} = FIS2(TD_{j,i}, RWS_{j,i}) (3-obj) | `solvers.py:156-158` |
| CAIL_{j,i} = FIS3(COR_i, RDR_i, TD_{j,i}), static input occupancy | `solvers.py:151-155,103-104` |
| Objectives are plain means over directed people/pairs | `solvers.py:211-224` |
| `_cail_terms`: plain per-pair cache when `cail_load is None`; with load coupling reads `cail_load[j,i,l_i]`, l_i = bincount of assigned centres | `solvers.py:227-232` |
| COR'_i(l) = min(100, COR_i + 100 l / cap_i), cap_i = ceil(kappa n_dir / n), kappa = inf -> separable | `load_coupling.py:9-19,33-40,64-81` |
| Quality = sum of the three objectives on clean cache | `run_seed_floor.py:151-153`; `decisions.py:116`; `analyze_floor.py:186` |
| 4-objective variant: (ULPP, TRD, RPD, CAIL), TRD=FIS2a(RCS,PHS), RPD=FIS2b(TD) | `solvers.py:159-163,218-221` |
| Cache once per instance; O(n_dir) look-ups | `solvers.py:90-96,119-125` |

### Encoding and operators
| Claim | Source |
|---|---|
| 2 n_dir genes; person slice partial permutation; centre reals in [0,1) | `solvers.py:7-16` |
| Decoding centre = min(floor(r n), n-1) (0-based; draft writes 1 + that) | `solvers.py:186-190` |
| Initialisation: random.sample + U[0,1) | `solvers.py:256-260` |
| (mu+lambda), lambda = mu, offspring = clones of the population, no mating selection | `solvers.py:399-401,418` |
| Pairs (i-1, i) crossed with prob 0.7 | `solvers.py:352,404-406` |
| Person slice uniform 0.5, dedupe (first kept), random refill at end | `solvers.py:280-292` |
| Blend alpha = 0.5, gamma = (1+2a)U - a, children (1-g)a+gb and ga+(1-g)b, clip [0, 1-1e-9] | `solvers.py:297-307` |
| Mutation prob 0.2 per individual; per gene 0.05; person swap-in; centre N(0, 0.1) clipped | `solvers.py:318-345,353,409-411` |
| NSGA-II = DEAP selNSGA2; NRGA = sel_nrga; NSGA-III = selNSGA3 with uniform_reference_points(nobj, p=4) | `solvers.py:555-570`; `models.py:268` |
| 15 reference points (3 obj, p=4), 35 (4 obj) | computed: `tools.uniform_reference_points(3,4)` -> 15, `(4,4)` -> 35 |
| Termination: `config.generations` (deadline optional, see F-ENV) | `solvers.py:395-398` |
| Output: first non-dominated front of final population, sorted lexicographically | `solvers.py:437-444` |

### Parameter table
| Claim | Source |
|---|---|
| Defaults pop 100, generations 200, nsga3_divisions 4 | `models.py:265-268` |
| Paper B config: NSGA-II, 3 objectives, equal weights | `experiments/run_turbulence.py:42,57-65` |
| Seeds 1000 + 7919 k | `experiments/run_h1_h2_h4.py:55-56`; `run_seed_floor.py:125`; `run_turbulence.py:97,129` |
| Prior paper: pop 100, 200 generations, 30 reps | `run_h1_h2_h4.py:57-59` |
| Seed floor 150 gen, 20 seeds (small, large) | `results/seed_floor/{small,large}_gen150/meta.json`; `run_seed_floor.py:72-74` |
| Budget curve 150/300/600/1200/2400/4800/9600, 5 seeds above 150 | `results/seed_floor/*_gen*/meta.json` |
| NRGA/NSGA-III: 150 gen 10 seeds, 2400 gen 5 seeds (small only) | `results/seed_floor/small_gen{150,2400}_{nrga,nsga3}/meta.json` |
| 240k controls: pop 200 x 1200, pop 400 x 600, 5 seeds | `results/seed_floor/small_evals240k_pop{200,400}/meta.json` |
| Alternate instances: 150 gen, 10 seeds | `results/seed_floor/*_gen150_inst4{3,4}/meta.json` |
| Turbulence MOEA: 150 gen, pop 100, small 8 reps x 12 realisations; large 6 x 8 | `experiments/run_turbulence_full.sh:12-16`; manifests (see B) |
| Envelope: NSGA-II pop 100, seed 1000, kappa = 2, gens = max(20, int(T/per_gen)) | `experiments/run_scaling.py:46-49,97,138-144` |

### Reference deciders
| Claim | Source |
|---|---|
| Exact: cost = w1 ULPP + min_i (w2 TIL + w3 CAIL), n_dir lowest, lexsort tie to lower index, equal weights default | `baselines.py:138-171` |
| Crisp priority = mean of AS, DS, IL, LS, IDL/100, 1 - min(max(RTR,0),72)/72 | `baselines.py:29,33-49` |
| Rank descending, tie to lower index, top n_dir | `baselines.py:79-83` |
| Crisp cost = mean of TD/180, COR/100, RDR/100, RCS, PHS; strict < keeps lowest index | `baselines.py:30,52-60,88-95` |
| No load awareness; deterministic; scored via `evaluate_pairs` | `baselines.py:1-10,99-110` |

### Committal rule and stability
| Claim | Source |
|---|---|
| Equal-weight min-max normalised sum; constant objective contributes 0 (span set to 1); argmin first index | `decisions.py:24-39` |
| Alternative rule: min raw L2 norm | `decisions.py:42-57` |
| Churn numerator: clean-directed people re-routed or no longer directed; denominator len(clean_map) = n_dir; persons directed only in the perturbed decision not counted | `decisions.py:109-112` |
| Symmetry in value when both sets have size n_dir | derived from `decisions.py:109-112` and `solvers.py:389-390` / decoder (both decisions always have n_dir distinct persons) |
| S3 orientation: `decision_stability(decisions[a], decisions[b])` with a < b | `run_seed_floor.py:157-160` |
| Jaccard of directed sets (ignores centres) | `run_seed_floor.py:59-66` |
| Quality loss = sum(perturbed obj) - sum(clean obj), both on clean cache | `decisions.py:106-107,116` |
| Floor = mean over 190 pairs; CI cluster bootstrap over seeds, 4000 resamples, self-pairs dropped | `analyze_floor.py:54,63-80,171-175` |
| Realisation r: `default_rng(1000 + r)`, same across levels | `run_turbulence.py:116-117`; `run_turbulence_exact.py:86-87` |
| Churn above floor: mean over realisations of mean over reps, minus floor mean | `analyze_floor.py:203-227` |
| Harm L_r: mean over reps of quality_loss; effect size mean/sd_clean, sd ddof=1 over clean seeds | `analyze_floor.py:164-165,212,231` |
| Bootstrap over realisations, 4000 | `analyze_floor.py:83-86,215-216` |
| Trend: Spearman of (level, L_r) over levels 0.05-0.4, one-sided, Holm within system (per size run) | `analyze_floor.py:55,209-214,234-247` |
| Exact decider regret: R = 50, no reps | `run_turbulence_exact.py:56,74,95-100` |

### Instances and perturbations
| Claim | Source |
|---|---|
| Sizes small 5/150/50, medium 8/225/75, large 10/300/100 | `generate_instances.py:53-57` |
| One generator per size, base seed 42, draw order people, centres, travel | `generate_instances.py:135-140,159` (salt: see F-SALT) |
| All fields independent; TD per pair | `generate_instances.py:76-124` |
| Observed TD max 81.28 (small), 88.73 (large); COR, RDR, IDL, RTR within generator ranges | committed `experiments/instances/{small,large}/*.csv` (read-only check) |
| Envelope sizes (150,5) ... (3000,50); n_dir = m // 3; instance seed 7 + s | `run_scaling.py:46,88,92` |
| Turbulence ranges: age [0,120], IDL [0,100], RTR [0,72], COR/RDR [0,100], TD [0,180]; noise sigma = level x range; clipped | `turbulence.py:51-58,104-105,112` |
| Flip to a uniformly chosen other level; missingness imputes median / mode | `turbulence.py:108-111,123-134` |
| RTR > 48 clipped at FIS1 input | `fis.py:599` |
