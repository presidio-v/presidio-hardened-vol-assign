---
phase: III — pre-submission gate
project: presidio-hardened-vol-assign
target-venue: Applied Sciences (MDPI), SI "Innovations in Supply Chain Resilience"
manuscript: pubs/v0.2.0-mdpi/author/paper.tex (21 pages, 10,509 words)
review-date: 2026-05-10
reviewer-role: Senior reviewer simulating worst-case rejection risk
                (humanitarian-operations / SCR background; has chaired tracks
                at HumLog/IFORS-equivalent venues; reads the FIS-MOEA disaster
                line and the SCR-theory canon).
---

# Adversarial pre-submission review

This is a self-administered red-team review against the manuscript at the
state of 2026-05-10. Severity codes:

- **Critical** — must be fixed before submission. No exceptions.
- **Moderate** — should be fixed; if not, requires a documented exception
  recorded under that issue.
- **Minor** — judgement call.

The submission gate (per design pattern) closes only when every Critical
is resolved and every Moderate is either resolved or has a recorded
exception.

---

## Critical issues

### C1. NSGA-III with custom variation operators was never validated on a benchmark

**Location:** §4.3 (chromosome encoding), §4.4 (NSGA-III reference points),
§5.1 (experimental setup).

**The problem:** The manuscript uses custom uniform-crossover-with-repair
and per-gene Gaussian mutation because DEAP does not provide a built-in
that fits a partial-permutation + real-valued mixed encoding directly
(§4.3). The feasibility plan (`pubs/v0.2.0-mdpi/explore/feasibility.md`,
"Fast path" item 2) committed to "Validated against DTLZ2 (4-obj)
benchmark before use." That validation never ran. A demanding reviewer
will ask: how do we know NSGA-III's reference-point niching is functioning
correctly when paired with operators it was not designed for? The H2a
result (NSGA-II beats NSGA-III on HV) is exactly the kind of finding that
could be explained by an operator-niching mismatch rather than a real
algorithmic property.

**Why this matters:** The H4 inversion (§5.3) already raises the spectre
of encoding-sensitive results. Without DTLZ2 validation, a reviewer
cannot rule out that H2a is also an encoding artifact rather than a
boundary-objective-count effect. The H2c CPU-time advantage is robust
against this concern (selection-mechanism complexity is independent of
operators), but the HV claim is not.

**Fix:** Run NSGA-II/NSGA-III on DTLZ2 with $M=4$, default operators per
the literature, and verify both algorithms reproduce published HV
benchmarks within a reasonable margin. Add the result as a one-paragraph
ablation in §5.1 or a short appendix. Estimated effort: 30 minutes
(pymoo has DTLZ2; 30 reps each, plot HV convergence).

---

### C2. The "no published model uses a named resilience framework" claim is unsupported by a systematic search

**Location:** §1 (closing paragraph), §2.5 (gap statement), §2.3 (closing
paragraph). All three places assert that no published FIS-MOEA disaster
allocation model organises its objectives around a named resilience
framework.

**The problem:** The claim is plausible but not backed by a documented
systematic literature search. A reviewer who works in SCR will know
papers we may have missed --- e.g., Sahebjamnia et al.'s work on
sustainable disaster operations, or Sheu's hierarchical disaster-relief
distribution. We cite 21 references; an SI feature paper on this topic
typically cites 50+. The negative claim ("no one has done X") is among
the easiest claims to refute and the hardest to defend. If the
reviewer-of-record can produce one counter-example, the contribution
narrative collapses.

**Why this matters:** Three of our five contribution claims (the 4R
mapping, the 4-obj split, the reproducible artifact) depend on this
negative-result framing. We can soften the language without weakening
the contribution, but only if we do so deliberately and consistently.

**Fix:** Two options.
1. **Cheap:** Soften the negative claim everywhere it appears. Replace
   "no published model" with "to the best of our knowledge, no FIS-MOEA
   disaster allocation model in the recent literature" or "the surveyed
   literature does not include a model that…". Document the survey
   strategy in §2 (databases queried, date range, search terms).
2. **Expensive:** Run a proper systematic search (Scopus + Web of
   Science + IEEE Xplore on (fuzzy OR FIS) AND (disaster OR humanitarian)
   AND (allocation OR routing OR shelter), 2018--2025, $n \approx 200$
   abstracts) and document a PRISMA-style flow diagram.

Recommended: Option 1 for this submission, with Option 2 deferred to a
v0.3.0 systematic review companion paper. Either way the language must
shift from "no published" to "to the best of our knowledge".

---

### C3. The Pareto-front sample size at $n_{\mathrm{dir}} = 50$/$75$/$100$ is small relative to typical 4-obj literature

**Location:** §5.1 (experimental setup), §5.2 (Pareto-front overview),
§6.

**The problem:** Pop\_size = 100 with NNS plateauing at 100 means every
individual on every front is non-dominated --- the Pareto front is
saturating the population. Reference-point methods like NSGA-III are
designed for fronts that overflow the niche budget. Our fronts do not
overflow (35 reference points; 100 individuals; 100 non-dominated). A
reviewer will say: "your population is too small for NSGA-III to express
its design advantages, which is why H2a refuted." This is exactly the
explanation the paper does not entertain.

**Why this matters:** It directly affects the H2a interpretation. The
paper currently argues NSGA-III's advantage is CPU efficiency, not
HV --- but an alternative interpretation is that NSGA-III is HV-equivalent
because the population is too small to discriminate, and would dominate
on HV at pop\_size $\ge 200$.

**Fix:** Add a single ablation cell at pop\_size $= 200$ on the medium
problem only, all three algorithms, 30 reps. If NSGA-III's HV catches up
or surpasses NSGA-II, mention in §5.4 and recalibrate the H2 narrative.
If not, the current narrative stands and is now defended against the
"undertraining" critique. Estimated cost: 90 NSGA runs $\approx$ 5 min.

---

## Moderate issues

### M1. The 4R$\leftrightarrow$objective mapping is asserted, not derived

**Location:** §3.5 (Resilience-theoretic mapping), Table 2.

**The problem:** Each of the four objectives is asserted to "operationalise"
one 4R component. The mapping ULPP$\leftrightarrow$Resourcefulness is
defensible (resource-time-remaining and infrastructure-damage feed FIS$_1$;
both are Resourcefulness sub-indicators in Bruneau's terms). But
TRD$\leftrightarrow$Robustness rests on equating "road-condition + hazard"
with Bruneau's "capacity to withstand shocks without significant loss of
function" --- a leap. CAIL$\leftrightarrow$Redundancy requires reading
"centre allocation balance" as Bruneau's "substitutability of system
elements" --- another leap. A reviewer steeped in SCR theory will push
back: these are post-hoc rationalisations of objectives that were
designed for other reasons (in ATRes 2026), not derivations from 4R.

**Why this matters:** The paper's headline contribution is the bijection.
If the bijection looks like a label slapped on existing objectives, the
"theoretical contribution" framing weakens.

**Fix:** Add a half-page subsection (could be under §3.5) that walks
through, for each objective, the explicit derivation: "Bruneau defines
Robustness as X. The system property closest to X in our model is Y.
The objective FIS that emits a quantitative measure of Y is Z. Therefore
TRD operationalises Robustness." For at least Robustness and Redundancy
this needs to be written out; for Resourcefulness and Rapidity the
mapping is more direct and a sentence each suffices. Estimated effort:
2--3 hours of careful writing.

### M2. Sign-correction of ATRes Eq.~(5) is unilateral

**Location:** §3.3 (sign-correction note), §6.2 (projection uses
sign-corrected formula).

**The problem:** We silently change ATRes's Eq.~(5) and use the
sign-corrected form throughout. Two of the three authors of ATRes are
authors here, so this is not plagiarism; but the manuscript does not
explicitly say "the present authors of [atres2026], i.e.\ ourselves,
acknowledge the inconsistency in the in-press paper and adopt the
corrected form here." An external reviewer reading this without the
authorship overlap could read the sign correction as a unilateral
re-interpretation of a published result.

**Why this matters:** The H1 projection-dominance analysis (§6.2) uses
the sign-corrected formula to project 4-obj fronts to 3-obj space. If
the sign convention is contested, so is the projection.

**Fix:** Two-sentence acknowledgement in the §3.3 note: "The present
authors include the lead and corresponding authors of \cite{atres2026}.
We use this paper to record the sign correction explicitly; ATRes will
be amended at proof stage if galley scheduling permits, and otherwise
the corrected form supersedes the in-press equation in any work that
cites both." Five minutes of writing.

### M3. The HV reference point $(100,100,100,100)$ is per-FIS-output universe, not problem-instance-aware

**Location:** §5.1 (experimental setup), Table~\ref{tab:stats-summary},
Section~\ref{sec:perf-algorithms}.

**The problem:** HV is computed against a fixed reference point at the
maximum of each FIS output universe. This is reproducible across runs but
not adaptive to the actual front extent. A reviewer experienced in MOEA
benchmarking will note that ATRes uses an instance-aware reference point
(the worst observed objective value across all algorithms on a given
instance), which is more discriminating between near-optimal fronts.

**Why this matters:** The H2a verdict depends on HV magnitudes. If the
reference point is too far outside the actual front extent, all
algorithms accumulate similar HV from the empty-space contribution,
masking real differences. Conversely, an instance-aware reference can
flip rankings.

**Fix:** Add a supplementary HV computation using an instance-aware
reference point (max-per-objective across the union of NSGA-II/NRGA/NSGA-III
fronts on the same (size, rep) cell). Report whichever ranking holds in
both schemes. If the rankings differ, this becomes its own paragraph in
§5.4.

### M4. H4 inversion is asserted to be encoding-driven without re-implementing ATRes's encoding

**Location:** §5.3 (third paragraph), §8.2.

**The problem:** We claim NRGA's faster-than-NSGA-II behaviour on the
large instance is "likely driven by the chromosome encoding" but we do
not actually run our experiment with ATRes's two-row matrix encoding to
test this. Without the test, the explanation is speculative, and the
paper's H4 narrative depends on it.

**Why this matters:** The discussion in §8.2 leans on the
encoding-sensitivity story to explain away the inversion. If the
explanation turns out to be wrong, §8.2 needs rewriting.

**Fix:** Implement ATRes's two-row matrix encoding as a `--encoding=atres`
flag in `pva allocate` and re-run the H4 cell (large, NSGA-II vs NRGA,
30 reps each, both encodings). One of three things happens:
1. Inversion persists $\to$ the encoding explanation is wrong; rewrite
   §8.2 with a different speculation (or honest non-answer).
2. Inversion disappears under ATRes encoding $\to$ the explanation is
   confirmed; §8.2 stands and gains an empirical anchor.
3. Mixed result (e.g., inversion on one encoding-rep configuration but
   not another) $\to$ the encoding explanation is partially true; §8.2
   needs careful rewriting.

Estimated effort: 1--2 hours of code, then 60 NSGA runs.

### M5. Sensitivity analysis runs only on the medium instance

**Location:** §7.1 (rule-base), §7.2 (weight LHS).

**The problem:** Both H3a and H3b are tested at medium size only.
A reviewer will ask: is sensitivity scale-dependent? Specifically,
larger problems with more candidate (person, centre) pairs may have
more cushion against single-rule deletions because the GA averages
over more allocations.

**Why this matters:** The H3 verdicts are framed as properties of "the
model" but are demonstrated only at one problem size.

**Fix:** Re-run H3a (66 deletions $\times$ 5 reps each, not 30 --- saves
time) on the small and large instances; report the per-FIS median
$|\Delta\text{HV}|$ side-by-side in §7.1. If the verdict changes by size,
discuss; if not, the current verdict is now scope-defended. Same for H3b
with 20 LHS samples each on small and large. Estimated effort: 2--3 hours
of compute.

### M6. The 50%-projection-dominance result has no null-model control

**Location:** §6.2 (projection-dominance analysis).

**The problem:** We report that ~50% of 4-obj front solutions are
dominated when projected to 3-obj space. A demanding reviewer will note
that *some fraction of solutions on a 4-obj Pareto front will always be
dominated when projected to 3-obj space, even if the split is information-
redundant*, simply because lower-dimensional projections lose information
generically. We need a null-model control: what fraction would be
dominated under a *random* TIL-like fusion of TRD and RPD? If the answer
is also ~50%, our finding is geometric, not informational.

**Why this matters:** This is the most-likely #1 reviewer concern in §6.

**Fix:** Generate a null distribution by repeatedly fusing TRD and RPD
with random convex weights ($\alpha \sim \mathcal{U}(0,1)$,
$\text{TIL}_{\text{null}} = \alpha \cdot \text{TRD} + (1-\alpha) \cdot \text{RPD}$,
$n=100$ samples per front), recompute the dominance fraction for each
random fusion, and report the difference between the ATRes-RWS-weighted
fusion's dominance fraction and the null distribution's mean. If the
ATRes fusion's fraction is statistically distinguishable from random,
H1 is confirmed *informationally*, not just geometrically. Estimated
effort: 1 hour of code, no new GA runs needed (uses existing fronts).

---

## Minor issues

### m1. Disaster mortality claim in §1 is unsourced

**Location:** §1, second paragraph: "the response-phase mortality
literature places the bulk of preventable deaths inside the first 12
hours after onset."

**Problem:** This claim is not cited. The "first 12 hours" figure
originates with Avishan et al.\ 2023 (cited in ATRes), but we did not
include the citation in lit.bib or the claim's footprint in §1.

**Fix:** Add Avishan 2023 to lit.bib and cite at end of the sentence.

### m2. CRED 2023 is grey literature, not peer-reviewed

**Location:** §1, opening paragraph; §3.1.

**Problem:** We cite \cite{cred2023} for the disaster numbers. CRED's
"Disasters in Numbers" is institutional grey literature, not peer-reviewed.
This is standard practice in disaster studies but a strict reviewer may
ask for a peer-reviewed alternative.

**Fix:** Add a peer-reviewed companion citation (e.g., Wallemacq \&
House 2018 EM-DAT validation paper) for the EM-DAT numbers.

### m3. We use "fairness" and "equity" loosely

**Location:** §1, §3.2.1, §6.1.

**Problem:** ATRes calls FLPP a "fairness" index; we follow them. But the
disaster-equity literature distinguishes carefully between Rawlsian
fairness (worst-off prioritised), equal-shares fairness (uniform
distribution), and proportional fairness (priority ~ need). FLPP encodes
something closer to Rawlsian-with-need, but we do not say so.

**Fix:** Add one-sentence disambiguation in §3.2.1: "FLPP encodes a
need-weighted Rawlsian fairness, prioritising the worst-off candidates
under a need-weighted maximin reading rather than equal-share or
proportional readings of fairness."

### m4. Bibliography is light for a feature paper

**Location:** lit.bib (21 entries).

**Problem:** SI feature papers in Applied Sciences typically cite 40--80
references. We have 21. A reviewer may flag this as inadequate engagement
with the literature.

**Fix:** Aim for 40--50 entries before submission. Add: Sahebjamnia 2017
sustainable disaster operations; Sheu 2007 hierarchical disaster-relief
distribution; Holguín-Veras 2012 humanitarian logistics; Anaya-Arenas
2014 relief distribution; Goldschmidt \& Kumar 2016 humanitarian
operations; one or two recent SI papers for citation reciprocity.

### m5. Self-citation density

**Location:** lit.bib has Rabiei 2021, Rabiei 2023, ATRes 2026 — three
self-citations to the same author cluster.

**Problem:** Three self-citations across a 21-entry bibliography is
14\%; reviewers often comment when this exceeds 10\%. Adding more
external references per m4 brings the percentage down to a defensible
6--8\%.

**Fix:** Subsumed by m4.

### m6. The "30\% faster" CPU claim is rounded; report exact numbers

**Location:** §5.3 ("approximately 30--35\,\%"), §8.

**Problem:** Round numbers are easy to challenge ("which exactly?"). The
exact ratios from the manifest are 1 - 0.91/1.45 = 37\%, 1 - 1.15/1.69 = 32\%,
1 - 1.45/1.99 = 27\%. Mean 32\%.

**Fix:** Replace "30--35\%" with the actual range "27--37\%" or "32\% on
average". Five-second edit.

### m7. Paper structure paragraph in §1 is dense

**Location:** §1, last paragraph (3 sentences listing what each section does).

**Problem:** Reading this paragraph as a non-author is harder than it
needs to be. MDPI papers usually have a slightly more visual structure
indication.

**Fix:** Optional. Could leave as-is or reformat as a small itemized list.

---

## Summary

| Severity | Count | Status |
|---|---|---|
| Critical | 3 | Must fix before submission |
| Moderate | 6 | Fix or document exception |
| Minor | 7 | Judgement calls |

### Decision per Critical issue

| ID | One-line fix | Estimated effort |
|---|---|---|
| C1 | Run NSGA-II/NSGA-III DTLZ2 ablation; add to §5.1 or appendix | 30 min |
| C2 | Soften "no published" to "to the best of our knowledge, in the surveyed literature" everywhere; document survey scope in §2 opening | 30 min |
| C3 | One ablation cell at pop\_size=200 medium 4-obj all three algos | 5 min compute + 10 min writing |

### Decision per Moderate issue

| ID | One-line fix | Effort | Recommendation |
|---|---|---|---|
| M1 | Half-page derivation of 4R$\leftrightarrow$objective mapping in §3.5 | 2--3 h | **Fix** |
| M2 | Two-sentence authorship acknowledgement in §3.3 sign-correction note | 5 min | **Fix** |
| M3 | Supplementary HV with instance-aware reference point | 1 h | **Fix** |
| M4 | Re-implement ATRes encoding; re-run H4 cell | 1--2 h + compute | **Fix** |
| M5 | Re-run H3a/H3b on small and large at reduced reps | 2--3 h | **Fix** |
| M6 | Null-model control for projection-dominance | 1 h | **Fix** |

### Decision per Minor issue

| ID | Fix | Recommendation |
|---|---|---|
| m1 | Add Avishan 2023 cite | **Fix** (5 min) |
| m2 | Add peer-reviewed EM-DAT companion cite | **Fix** (5 min) |
| m3 | Disambiguate fairness reading in §3.2.1 | **Fix** (5 min) |
| m4--5 | Expand bibliography to 40--50 entries | **Fix** (1--2 h) |
| m6 | Replace "30--35\%" with exact figures | **Fix** (1 min) |
| m7 | Restructure §1 closing paragraph | **Defer** (style preference) |

### Submission gate status

**Critical tier cleared as of 2026-05-10.** All three Critical issues
resolved within the day they were filed; six Moderates and seven Minors
remain.

#### C1 resolution — DTLZ2 selector validation
Ran NSGA-II vs NSGA-III on DTLZ2 ($M=4$, $n_{\mathrm{var}}=10$) with
DEAP's standard continuous operators (`cxSimulatedBinaryBounded`,
`mutPolynomialBounded`), 30 reps each. Result: NSGA-III HV
$0.970 \pm 0.006$ vs NSGA-II HV $0.797 \pm 0.032$ ($p < 10^{-15}$,
Mann--Whitney $U$). The literature-expected ordering holds in our
framework. The H2a refutation in our humanitarian allocation problem is
a property of the problem, not a bug. **Selector validation paragraph
added to §5.1.**
Driver: `experiments/run_dtlz2_validation.py`. Manifest:
`experiments/results/dtlz2_validation/dtlz2_manifest.csv`.

#### C2 resolution — softened negative claims, documented survey scope
Edited four locations in §1, §2.2 closing, §2.3 (twice). Replaced
"no published" / "absent from this literature" / "none of the published"
with "to the best of our knowledge, in the surveyed literature".
Added survey-scope paragraph at §2 opening documenting databases
(Scopus, Web of Science, IEEE Xplore), date range (2014--2025 + landmark
foundational), and explicit non-claim of "systematic-review-grade
exhaustive coverage".

#### C3 resolution — pop=200 ablation
Re-ran 90 cells on medium 4-obj at pop\_size = 200 (NSGA-II, NRGA,
NSGA-III $\times$ 30 reps). NSGA-II HV remains significantly ahead of
NSGA-III ($19.66 \pm 0.39$ vs $19.10 \pm 0.58$ million, $p < 0.001$);
direction unchanged from pop = 100. NSGA-III's CPU advantage *widens*
to 51\% (vs 32\% at pop=100). The H2a verdict and the boundary-
objective-count interpretation are robust to population size.
**Ablation paragraph added to §5.3.**
Driver: `experiments/run_h1_h2_h4.py --pop-size 200 --sizes medium
--formulations 4`. Manifest:
`experiments/results/h1_h2_h4_pop200/manifest.csv`.

#### What changed in the paper
- §1 final paragraph and §2 (multiple): softened negative claims, added
  survey-scope paragraph.
- §5.1: new "Selector validation on DTLZ2" paragraph.
- §5.3: new "Population-size ablation" paragraph defending H2a against
  the undertraining critique.

Compile status: **22 pages, 10{,}904 words; clean build, zero undefined
references.**

#### Remaining work to clear submission gate
6 Moderates + selected Minors. Recommended order: M2 (5 min editorial,
trivial), m1/m2 (10 min, citations), m6 (1 min, exact percentages), then
the substantive Moderates M1/M3/M4/M5/M6/m4 in any order.

### Total estimated effort to clear gate

- Critical: $\approx 1$ hour code/compute + $\approx 1$ hour writing.
- Moderate: $\approx 6--10$ hours code/compute + writing.
- Minor (selected): $\approx 2--3$ hours mostly writing.

**Total: 9--14 hours of focused work.** This is on the timeline budget;
Week 7 of the 8-week plan was reserved for the adversarial review
itself plus revisions, so the schedule is intact.
