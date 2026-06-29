# ATRES Rejection — Reviewer Comments (stored) & Assessment

**Status:** archived for action planning
**Stored:** 2026-06-29 · forwarded by D. Arias-Aranda (the EiC email had landed in spam and was recovered)

| Field | Value |
|---|---|
| Manuscript number | **ATRES-D-26-00014** |
| Title | *A Novel Multi-Objective Optimization Model for Directing Affected People to Relief Centers in Post-Disaster Scenarios: Combining Fuzzy Inference Systems with NSGA-II and NRGA* |
| Journal | Autonomous Transportation Research (Elsevier) |
| Decision | **Reject** (reviewers recommend against publication) |
| Editor-in-Chief | Xinping Yan |
| Corresponding author | Prof. D. Arias-Aranda |
| Reviewers | 3 |

This manuscript is the paper behind the **humanitarian allocation model** in this
repository (the v0.2.0 "affected people → relief centres" model; see
[`docs/v0.2.0-plan.md`](v0.2.0-plan.md)). Several reviewer points map directly
onto code, data, and metrics that already live here — those mappings are made
explicit in the assessment so the manuscript and the implementation can be
brought into agreement for a resubmission.

---

## 1. Verbatim reviewer comments

### Editor

> I regret to inform you that the reviewers recommend against publishing your
> manuscript, and I must therefore reject it.
>
> Note (Reviewer 1): "Basically, the topic is out of the scope of this journal."

The strongest single reason for rejection is **scope**: an autonomous-transportation
journal is a poor venue for a humanitarian relief-allocation paper. This is not a
quality defect and should weigh heavily toward **resubmission to a better-matched
journal** (e.g. the venue the original 2023 ED-staffing paper appeared in, or an
OR / disaster-management / decision-support journal).

### Reviewer 1

> The paper proposed a MOO framework for the humanitarian issues. Basically, the
> topic is out of the scope of this journal. Thus, I believe that it is not
> suitable for publication. Technically, the following comments may be helpful to
> enhance the quality if the authors would like to submit the paper to somewhere
> else.
>
> 1. The paper seems like a modelling problem rather than framework. The
>    contribution focuses on the assumption and modelling, no visible improvement
>    for the MOO methods.
> 2. The writing is AI taste. I am not sure if the authors use AI to polish the
>    writing or not but the writing is very dry which make the contents difficult
>    to follow, please add an acknowledgement if the AI tools have been used
>    during the writing.
> 3. The assumptions need the evidence from the social science and humanity
>    research, more literature is needed to support the statement. In my opinion,
>    the topic and description is very general and it is difficult to overview the
>    whole topic using one single paper.
> 4. The comparison study is not sufficient, why these MOO methods have been used
>    here not others, while the data should be attached to support the
>    reproductive to the proposed results.

### Reviewer 2

> The manuscript addresses an interesting and practically relevant problem in
> post-disaster victim allocation by integrating fuzzy inference systems with
> multiobjective evolutionary optimization. However, several issues should be
> addressed before publication.
>
> 1. The novelty of the work is limited, as the main contribution appears to be
>    the design of three qualitative indicators, while the integration of FIS with
>    evolutionary algorithms is not new.
> 2. The proposed indicators (FLPP, TFL, and CABL) lack sufficient theoretical or
>    empirical justification. Their effectiveness and relevance should be further
>    validated.
> 3. The fuzzy rule base is entirely expert-driven, but no sensitivity analysis or
>    validation of the rules and membership functions is provided.
> 4. The experimental study mainly compares NSGA-II and NRGA rather than
>    demonstrating the superiority of the proposed framework over existing
>    allocation or decision-making approaches.
> 5. The optimization model omits several practical constraints, such as explicit
>    shelter capacity and transportation limitations, which may affect its
>    applicability.
>
> Overall, the manuscript has potential, but substantial revisions are required to
> strengthen its novelty, validation, and experimental evaluation.

### Reviewer 3

> The following comments are written in markdown, along with LaTeX formulas
> surrounded in dollar signs.
>
> #### Main problems with this manuscript
> I think this topic is worthy of investigation, and I think your methods have a
> lot of potential. However:
> 1. Its objectives are too vague
> 2. The study doesn't measure its methods against any known bench mark or existing
>    solution
> 3. It's poorly written, and reads like a first draft. Assumes reader has the time
>    and patience to dig through 14 tables of raw data without the support of the
>    manuscript or figures.
>
> #### Global comments
> - Your literature review doesn't sufficiently justify the purpose of your study
> - You do not appropriately introduce fuzzy systems, their utility, and why
>   they're critical to your study. You should not assume that the reader is an
>   expert in the field
> - You over use "qualitative," as some of the measures you're using aren't
>   qualitative (distance)
> - Whole paper comes off as a rough draft.
> - In section 3 you offer no intuitive description of the objectives. Just formulas
>   that the reader is expected to parse through, dig through charts, and figure out
>   for themselves.
> - Section 3 makes much more sense after section 4. The paper should be
>   restructured to flow better and be easier to read
> - 14 pages of tables! And your methodology expects the reader to parse through the
>   tables to understand the methods. This is too much.
> - Only two figures, and no figures of results. Compile your data into well
>   summarized figures.
> - Paper lacks a clear objective and hypothesis. The closest thing I see is
>   "Therefore, the motivation of this work is to build a robust multi-criteria
>   optimization model that effectively directs affected people to relief centers in
>   the response phase of a disaster by considering proper objective functions.
>   Also, we want to encapsulate experts' expertise, knowledge, and insight away
>   from mathematical complexities in the decision-making process to deal with the
>   constantly changing environment effectively."
> - This doesn't really have a clear hypothesis
> - Assuming your implicit hypothesis is "we can include fuzzy expert knowledge to
>   multiobjective disaster relief", you'd want to compare your method against an
>   existing method in the literature as a bench mark. Instead, you just compare to
>   algorithms to each other (NSGA-II and NRGA). For all we know, you're just
>   comparing two algorithms which could have poor performance compared to the rest
>   of the literature.
>
> #### Line by line comments
> - Page 4, lines 11-13: Is this globally? for a certain country or region?
> - Page 5, lines 9-10: Sentence is poorly structured, as well as the paragraph as
>   a whole
> - Page 6, lines 36-40 "... and transportation hubs, gett**ing**... relief
>   services... and plan**ning**..."
> - Page 7, line 31: Make the period a comma, or capitalize the "s" in "such"
> - Page 7, lines 41-56: Consider combining these two paragraphs into one
> - Page 7: Please describe why these objective functions can be in conflict with
>   each other
> - Page 8, lines 28-30: feasibility and balance have not been introduced yet.
>   Provide the appropriate context for these concepts in your literature review. If
>   they are new concepts, provide a bit more context for these exact literature
>   gaps.
> - Page 8: Why use qualitative criteria instead of quantitative? I don't think you
>   appropriately introduce this
> - Page 8: lines 46-50: Your literature review has neither an introduction of fuzzy
>   systems nor a justification for its use in this study
> - Page 9, line 24: "...exceeds much further..." should be rewritten
> - Page 9, line 59: I'd recommend changing "...highlighting of inequalities..." to
>   "...exacerbation of inequalities...". "Highlighting" implies that inequalities
>   are just becoming more visible, as opposed to actually becoming better or worse
> - Page 10, lines 4-9: this sentence should be rewritten. It is hard to follow
> - Page 10, lines 19-21: This sentence is confusing
> - Page 10, lines 22-36: Are these qualitative indices? It feels like a mix of
>   qualitative and quantitative indices
> - Page 10, lines 37-41: Are these subindices equally weighted? I imagine you
>   mention this later on, but it'd be helpful mention this earlier on
> - Page 11, lines 11-19: Again, this is a mix of qualitative and quantitative
>   metrics. Travel duration is a quantitative index.
> - Page 11, lines 47: Again, these don't feel entirely qualitative
> - Page 12, section 3.3: A section cannot be a list and a table. This should be a
>   manuscript, not an outline
> - Page 13, Table 1: I like tables that catalog terms, but it is unreasonable to
>   expect readers to comb through variables to try to sus out their meaning.
>   Introduce the concepts as prose, and possibly move the table to the appendix. It
>   could also be organized more effectively, as there is so much information within
>   it
> - Page 16, lines 19 and 20: I do not understand this notation. Is this notation
>   for fuzzy logic?
> - Page 16, Equations 1-3: At a first pass, I'm quite lost with what is going on
>   here. So what is the objective function? As I dig into the manuscript (off in
>   page 18), I eventually learn that $Mn\_ULPP$ is the objective, but neither the
>   equation nor the caption gives any hint of this on page 16.
> - Page 16, lines 45-46: Change "Travel" to "travel".
> - Page 16-17, Equations 4-7: Again, I cannot follow this. No support to building
>   an intuitive understanding of the concept.
> - Page 18, end of section 3: Overall, I'm very unclear how the objectives are
>   formed. The paper assumes that the reader has the time to dig through these
>   equations, look up each term in a huge table, and slowly piece together what
>   these equations are trying to say. This puts too much onus on the reader, and
>   assumes that they are comfortable with fuzzy logic, which is unreasonable.
>   Regardless of the merit of these approaches, the design of the methods should be
>   clear to anyone with a general background in STEM.
> - Page 19, lines 37-40: What are membership functions?
> - Page 19, lines 41-43: What are triangular and trapezoidal numbers?
> - Page 20: You're missing a lot of justification to the use of these models. So
>   why GA over classical methods (e.g., integer programming, dynamic programming,
>   brute force)? Why did you choose NSGA-II and NRGA over other multiobjective
>   algorithms?
> - Page 20: You're not the first authors to combine GAs with fuzzy logic. Back up
>   the claim or be more nuanced.
> - Page 20: I think a brief paragraph on how GAs would be useful
> - Page 21, Page 21: what do you mean by "are outlined?"
> - Page 21, expression 17: This description is very vague. So, $C_r$ is a real
>   number that maps to a relief center? So like if we have five relief centers,
>   $C_1=0$ , $C_2=0.25$, $C_3=0.5$, $C_4=0.75$, and $C_5=1$? Then $P_n$ is an ID of
>   an individual? Why not just have an $n\times1$ chromosome where each cell
>   represent an individual to serve, and the value of each cell is the center
>   they're directed to?
> - Page 21, Tables 2-5: these tables help the understanding of equations 1-7
>   tremendously. They should be introduced far earlier on to support the
>   understanding of the objective functions
> - Page 29, equation 18: MID is not a good measure of performance for MOO, as it
>   will favor solutions nearest to the ideal point. However, every member of a
>   Pareto front is equally optimal, each representing a particular trade-off between
>   the conflicting objectives. I'd recommend removing this. HV is sufficient, as it
>   measures both convergence and diversity.
> - Page 30, lines 4-7: Pages 9 through 14: You're missing some critical
>   optimization parameters in here: population size, number generations/stopping
>   criteria. Also, are you doing repeated runs with different seeds?
> - Page 36-37: These tables belong in the appendix. Way too much detail. Only table
>   14 is necessary. Also, just a thought: most multiobjective studies just use
>   Wilcoxon rank sum, so you probably don't need the whole statistical analysis

---

## 2. Assessment

### 2.1 Headline read

The decision is **reject**, but two of three reviewers explicitly say the topic
is worthy and the methods have potential (R2: "interesting and practically
relevant", R3: "worthy of investigation… a lot of potential"). The dominant
killer is **journal scope**, not fatal scientific error — R1 leads with
"out of the scope of this journal." The actionable conclusion is **revise and
resubmit to a better-matched venue**, not abandon.

Three themes recur across all reviewers and should drive the revision:

1. **Benchmarking gap (most serious technical objection).** R1.4, R2.4, R3.2 and
   R3's global comments all say the same thing: comparing NSGA-II vs NRGA only
   shows two algorithms against each other, not the *framework* against an
   existing allocation/decision method. This is the one comment that recurs in
   every review and is the highest-priority technical fix.
2. **Presentation / writing.** R1.2 ("AI taste", dry, hard to follow), R3.3 and
   most of R3's line-by-line: rough-draft prose, no intuition before equations,
   14 pages of tables, only two figures and none of results, Section 3 before
   Section 4. This is a large but mechanical body of work.
3. **Justification / validation of the indicators and the FIS.** R2.1, R2.2,
   R2.3, R3's "vague objectives" and "introduce fuzzy systems": novelty rests on
   three indicators that need theoretical/empirical grounding, and the
   expert-driven rule base needs a sensitivity/validation study.

### 2.2 What the code in this repo already answers

A striking number of reviewer points are *already implemented* in
`presidio-hardened-vol-assign` and simply need to be **surfaced in the
manuscript**. These are low-cost wins:

| Reviewer point | Status in this repo | Where |
|---|---|---|
| **R3 — chromosome design** ("why not an $n\times1$ chromosome, each cell = the centre that person is directed to?") | **Already exactly this.** The humanitarian domain uses integer encoding: a list of length `n_people`, gene *i* ∈ [0, n_centers−1] = the centre person *i* is allocated to; every chromosome is feasible. The manuscript's vague real-valued `$C_r$/$P_n$` description (expression 17) is *out of date* relative to the implementation — align the paper to the code. | `src/presidio_vol_assign/domains/humanitarian.py` (encoding docstring) |
| **R2.3 / R2.5 — sensitivity analysis of rules & MFs** | **Implemented.** `pva sensitivity` perturbs FIS output scores by ±10/±20 % and re-runs the solvers, emitting `(factor, solver, NNS, MID, SM, HV, cpu_time)`. Directly answers "no sensitivity analysis of the rules and membership functions." | `sensitivity.py`, `test_sensitivity.py`, README "Sensitivity analysis" |
| **R1.4 / R3 — "data should be attached for reproducibility" / repeated runs / seeds** | **Implemented.** Deterministic `--seed`, REP bit-for-bit reproducibility metric (SHA-256 of canonical Pareto CSV), `pva benchmark --check-repro`, and ready-to-run synthetic datasets under `examples/` (regenerable via `generate_examples.py`). | `repro.py`, `benchmark.py`, `examples/` |
| **R3 — missing optimisation parameters** (population size, generations, stopping criteria, repeated runs/seeds) | **Implemented & exposed** as CLI flags (`--pop-size`, `--generations`, `--seed`) with documented defaults (pop 100, gen 200). The values exist; they must be stated in the manuscript's experimental-setup section. | `cli.py`, README CLI reference |
| **R2.5 — "omits shelter capacity and transportation limitations"** | **Partially answered.** Capacity *is* modelled (Z3 centre overcrowding via FIS-C utilisation = Σ group_size / capacity) and transportation feasibility *is* an objective (Z2 / FIS-B: distance × mobility × road accessibility). Caveat: capacity is a **soft** objective, not a hard constraint (v0.2.0-plan §8.3). Reviewer likely read the manuscript's older constraint set — surface FIS-B/FIS-C explicitly, and consider adding a hard-capacity option. | `domains/humanitarian.py`, `fis_humanitarian.py` |
| **R3 — only two figures, none of results** | **Tooling exists.** `pva show` renders publication-quality Pareto figures (pairwise projections + 3-D scatter, solvers overlaid; `examples/paper_scale_fronts.png` already generated). Use it to produce the missing results figures. | `viz.py`, README "Figures" |

> Implication: a meaningful fraction of R2 and R3's technical objections are
> **manuscript-presentation gaps, not capability gaps.** The code and data are
> ahead of the paper; the revision largely needs to *describe what already
> exists* and align notation (especially the chromosome encoding) with the
> implementation.

### 2.3 Genuine gaps the code did **not** yet answer

> **Status update (2026-06-29):** all four code-addressable gaps below have now
> been implemented on this branch — see the ✅ rows. R2.2 is supported by a new
> objective-ablation mode (`pva ablation`); the remaining manuscript-side
> validation (case study / expert elicitation) stays with the authors.

| Reviewer point | Gap | Action / status |
|---|---|---|
| **R1.4 / R2.4 / R3.2 — benchmark vs an existing method** (the recurring #1 technical objection) | The repo only compared NSGA-II / NRGA / nrga-ranked **to each other**; no baseline allocation method. | ✅ **Done.** Added a deterministic weighted-sum greedy baseline comparator (`SolverType.GREEDY`, `baselines.py`, `Domain.baseline_population`) for both models; `--solver greedy` and `pva benchmark --baseline` report it as a `greedy` row so the framework is measured against an existing-style heuristic. |
| **R3 — MID is a poor MOO indicator; drop it, HV suffices** | `metrics.py` reported MID prominently; MID rewards proximity to the ideal point, penalising legitimate trade-offs. | ✅ **Done.** HV is now the primary reported metric; MID is shown last and flagged *diagnostic* (still computed for 2023-paper backward-compat). Docstrings/README/CHANGELOG updated. Manuscript: drop or footnote eq. 18. |
| **R3 — statistical testing (Wilcoxon rank-sum)** | No statistical test; benchmark reported mean ± std only. | ✅ **Done.** `stats.py` runs a Wilcoxon rank-sum test on per-instance HV distributions (each solver vs. the greedy baseline), auto-invoked by `pva benchmark` with ≥2 solvers and ≥5 instances; prints a significance table and writes `stats_<ts>.csv`. Uses the existing `scipy` dependency. |
| **R2.2 — empirical justification of FLPP/TFL/CABL indicators** | The indicators are defined and computed, but there is no empirical validation that they capture distinct, relevant information. | ✅ **Code-supported.** Added `pva ablation` (`ablation.py`): leave-one-objective-out re-solve, reporting how the dropped objective and full-space HV degrade — quantitative evidence each indicator is non-redundant. The complementary manuscript-side validation (case study / expert elicitation) remains the authors' study-design task. |

### 2.4 Manuscript-only items (no code impact)

These are writing/structure tasks for the authors; tracked here for completeness
and to feed a revision checklist. They cannot be resolved in this repository.

- **Scope / venue (R1, editor):** choose a better-matched journal before resubmitting.
- **Writing quality & "AI taste" (R1.2, R3.3):** substantial copy-edit; per R1.2,
  add an AI-tool-use acknowledgement if applicable (also an integrity requirement).
- **Clear objective + hypothesis (R3):** state an explicit hypothesis ("fuzzy
  expert knowledge improves multi-objective disaster relief vs. a baseline") —
  which dovetails with the benchmarking gap in §2.3.
- **Introduce fuzzy systems / membership functions / triangular-trapezoidal
  numbers (R3 global + p19):** add a primer; don't assume an expert reader.
  Note: the math is fully worked in [`docs/fis-worked-example.md`](fis-worked-example.md),
  which can seed this section.
- **Restructure: Section 3 after Section 4; tables 2–5 earlier; move raw-data
  tables (pp. 36–37, Table 1) to an appendix; add intuitive prose before every
  equation (R3).**
- **"Qualitative" overused — distance/travel-duration are quantitative (R3,
  multiple lines):** relabel indices precisely; the README/specs already call
  some of these quantitative inputs (distance km, group_size), so align terminology.
- **Literature support for social-science assumptions (R1.3) and for the
  feasibility/balance concepts (R3 p8).**
- **All line-by-line copy edits (R3 pp. 4–21):** grammar, sentence rewrites,
  "Travel"→"travel", "highlighting"→"exacerbation", combine paragraphs, etc.

### 2.5 Priority recommendation

1. **Re-home the paper** to an in-scope journal (addresses the dominant rejection reason).
2. **Close the benchmarking gap** — the only objection raised by all three
   reviewers. ✅ Code-ready: greedy baseline comparator + `pva benchmark
   --baseline`; report the framework against it in the manuscript.
3. **Surface existing capability in the manuscript:** sensitivity analysis,
   seeds/parameters/repeated runs, REP reproducibility, results figures (`pva show`),
   and **fix the chromosome-encoding description to match the integer encoding the
   code already uses** (directly answers R3's encoding question).
4. **Metrics:** lead with HV, de-emphasise MID; add Wilcoxon rank-sum.
   ✅ Code-ready (HV primary in all summaries; Wilcoxon HV test in `pva benchmark`).
5. **Major rewrite pass** for clarity/structure/figures and the fuzzy-systems primer.
6. **Add the AI-use acknowledgement** if AI tools were used (R1.2 + integrity).

> None of the reviewers identified a correctness error in the model or the code.
> The objections are venue, framing, benchmarking, and presentation — all
> addressable. Recommend **revise & resubmit elsewhere**, not abandonment.

---

## 3. Manuscript revision plan (two submissions)

Planning context: **one submission in the next few days**, and **a second in
2–3 months**. The code-addressable gaps are now closed in this repo (§2.3), so
what remains is overwhelmingly *manuscript* work — the authors' to write — plus
two deeper modelling extensions now also implemented (see §4). Checklist below;
"✅ code-ready" means the evidence/feature exists and only needs to be written up.

### 3.1 Submission 1 (next few days) — writing & framing only, no new experiments

**Positioning**
- [ ] State an explicit **hypothesis** and sharpen the **contribution** — the
  contribution is the *integration + the three indicators + a reproducible tool*,
  not a new MOO operator (answers R1.1, R2.1, R3 "vague objectives / no hypothesis").
- [ ] **Re-home the journal** to an in-scope OR / disaster-management /
  decision-support venue (the dominant rejection reason — R1 "out of scope").
- [ ] Add an **AI-use acknowledgement** if AI tools were used (R1.2; integrity).

**Paste in evidence the code now generates** (the reviewers' #1 technical gap)
- [ ] ✅ **Baseline comparison** + **Wilcoxon HV** significance — `pva benchmark
  --baseline` (R1.4 / R2.4 / R3.2).
- [ ] ✅ **HV leads; MID dropped/footnoted** (R3, eq. 18).
- [ ] ✅ **Experimental params** — population, generations, stopping criterion,
  seeds, repeated runs (R3, p30).
- [ ] ✅ **Sensitivity** (`pva sensitivity`, R2.3) and **ablation** (`pva
  ablation`, R2.2) tables as robustness / indicator-non-redundancy evidence.
- [ ] ✅ **Fix the chromosome description** to match the integer n×1 encoding the
  code already uses (R3, p21 / expr 17).
- [ ] ✅ **Results figures** via `pva show` (R3 — "only two figures, none of results").

**Structure & clarity** (the bulk of R3's line-by-line)
- [ ] Reorder so **Section 3 follows Section 4**; introduce **Tables 2–5 earlier**;
  move **Table 1 and the pp. 36–37 raw tables to an appendix**; add **intuitive
  prose before every equation** (R3).
- [ ] Add a **fuzzy-systems primer** (MFs, triangular/trapezoidal numbers, why
  fuzzy) — seed from [`docs/fis-worked-example.md`](fis-worked-example.md) (R3 p8/p19).
- [ ] Fix **"qualitative" overuse**: distance/travel-duration are quantitative
  inputs; only the fuzzy-aggregated outputs are qualitative. State sub-index
  weighting up front (R3 p10/p11).
- [ ] Explain **why the objectives conflict** (R3 p7).
- [ ] **Copy-edit pass** for the rough-draft prose and every specific line fix
  (p4 "global/regional?", p6 gerunds, p7 punctuation, p9 "exceeds much further" /
  "highlighting→exacerbation", p16 "Travel→travel", etc.).

### 3.2 Submission 2 (2–3 months) — deeper experiments & validation

- [ ] ✅ **Stronger / exact baseline** — `--solver exact` (exact weighted-sum:
  Hungarian assignment for ed-staffing, MILP for humanitarian) gives a
  globally-optimal-per-scalarization comparator beyond the myopic greedy
  (R2.4 / R3.2). See §4.1.
- [ ] ✅ **Hard capacity + transport limits** — `--hard-capacity` /
  `--max-distance` repair mode, complementing the soft-capacity default (R2.5).
  See §4.2.
- [ ] **Empirical indicator validation** — a real-data **case study** and/or
  **expert-elicitation / inter-rater agreement** study for the rule bases and MFs,
  beyond the statistical non-redundancy the ablation already shows (R2.2 / R2.3).
- [ ] **Social-science grounding** — literature support for the vulnerability /
  fairness / prioritisation assumptions from humanitarian and social-science
  research (R1.3).

---

## 4. Modelling extensions delivered for Submission 2

### 4.1 Exact weighted-sum baseline (`--solver exact`)

A non-myopic comparator stronger than the greedy heuristic: the weighted-sum
scalarisation solved **to optimality** at each weight on the objective simplex
(`Domain.exact_baseline_population`).

- **ed-staffing** — exact min-cost type-feasible bipartite assignment per weight
  via `scipy.optimize.linear_sum_assignment` (Hungarian), encoded as the
  permutation the greedy decoder reproduces.
- **humanitarian** — exact weighted-sum MILP per weight via
  `scipy.optimize.milp`: `z1`/`z2` are linear in the assignment and modelled
  exactly; centre balance uses a **linear capacity-overload surrogate** (so the
  programme stays an exact MILP). The true FIS objectives (`z1`, `z2`, `z3`) are
  then reported on the optimal assignment. For weights with no balance term the
  result is the exact `z1+z2` optimum.

Uses the existing `scipy` dependency (no new package). Run standalone
(`--solver exact`) or as a benchmark comparator.

### 4.2 Hard-capacity & transport-limit mode (humanitarian)

The default humanitarian model treats capacity as a *soft* objective (`Z3`) and
transport as a feasibility objective (`Z2`). The new constraint mode adds, on top:

- **Hard capacity** — a deterministic greedy **repair decoder** guarantees no
  centre exceeds its capacity (overflow people, lowest-priority first, are moved
  to the nearest centre with spare room).
- **Transport limit** — low-mobility people (`mobility` below a threshold) are not
  placed beyond a maximum distance during repair, with a documented nearest-
  feasible fallback.

Enabled with `--hard-capacity` (and `--max-distance` / `--mobility-threshold`);
the soft-capacity default is unchanged.
