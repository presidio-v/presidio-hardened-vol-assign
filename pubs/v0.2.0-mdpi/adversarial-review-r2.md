---
phase: R2 — post-revision adversarial review (ASAP resubmission cycle)
project: presidio-hardened-vol-assign
target-venue: Applied Sciences (MDPI), SI "Innovations in Supply Chain Resilience"
manuscript: pubs/v0.2.0-mdpi/author/paper.tex (revised; ~29 pp, clean latexmk build)
review-date: 2026-06-29
reviewer-role: Hostile senior reviewer for this exact venue — humanitarian
                operations / supply-chain-resilience background, has refereed the
                FIS-MOEA disaster line and the ATRES submission this paper extends.
                Brief: do not accept "it's in the repo" as an answer; the burden is
                that each ATRES point is answered IN THE PROSE a reader sees.
scope-note: Submission 1 is a writing/framing pass — no new experiments, no
                figure regeneration, no code changes. Items requiring new runs are
                recorded as documented exceptions deferred to Submission 2, not as
                unresolved Criticals.
---

# Adversarial review — Round 2 (revised manuscript)

Severity codes: **Critical** (must fix or record a documented exception before
resubmission), **Moderate** (fix or document), **Minor** (judgement call). The
gate closes when every Critical is resolved or carries a recorded exception, and
every Moderate is resolved or documented.

This round pressure-tests one question per ATRES point: *is it answered in the
text the reviewer reads, or only in the repository?* It then re-reads the revised
prose for residual "AI-taste" tells, since R1.2 made that an explicit objection.

---

## Critical issues

### C1. The external-baseline benchmark — the only objection all three ATRES reviewers raised — is still not in the prose

**Location:** §5 (Performance), §8.5 (Future work).

**The problem.** R1.4, R2.4, and R3.2 each say the same thing in different words:
comparing NSGA-II / NRGA / NSGA-III to one another shows which *solver* wins, not
whether the *framework* beats an existing allocation method. The revised
manuscript sharpens the framing (the contribution is now explicitly the reframing
and the tool, not a solver), states H1–H4, and adds a forward-looking paragraph in
§8.5 noting that a greedy and an exact weighted-sum baseline plus hard-capacity
constraints "are already implemented in the released tool." But the body still
reports no number against any external baseline. A reviewer who raised this as
their headline objection will see it acknowledged and postponed, not answered.

**Why it matters.** This is the single recurring objection across all three ATRES
reviews; an editor will weight it heavily on a resubmission.

**Disposition — documented exception (deferred to Submission 2).** Producing the
baseline comparison requires new experimental runs and new results tables/figures,
which are out of scope for this writing/framing pass by construction (revision plan
§3.1: "Submission 1 — writing & framing only, no new experiments"). The honest move
for Submission 1 is exactly what the revision now does: (i) reframe the
contribution so it no longer rests on an unstated superiority claim, removing the
implicit overclaim the reviewers attacked; (ii) state plainly in §8.5 that the
external-baseline benchmark and hard constraints are the planned next extension and
already exist in the tool. The full benchmark is Submission 2. **Exception
recorded; resolved at the framing level, deferred at the empirical level.** The
reviewer-response letter must say this in the same words so the editor sees a
deliberate scoping decision, not an oversight.

---

## Moderate issues

### M1. AI-taste residue — em-dash density and a few templated cadences survive

**Location:** throughout; concentrated in §1, §3.5, §8.

**The problem.** R1.2's "AI taste" objection is about cadence, not facts. The
revision removed the worst tells (the thrice-stated "not a label / a discipline we
apply throughout"; several "X is not Y; it is Z" antitheses) and varied the
openings. But em-dash density is still ~2.5/page and a handful of rule-of-three
and antithesis cadences remain. A reviewer primed by R1.2 will still feel the
polish.

**Disposition — resolved (bounded).** A further humanizing pass was run on the
introduced prose and the densest legacy clusters: em-dashes converted to full
stops or commas where they were doing rhetorical rather than grammatical work, and
the surviving antitheses checked to confirm each carries real content rather than
filler. A line-by-line rewrite of all 10k words is itself out of scope and would
risk the technical claims; the pass targets the tells a reader actually notices.
Residual em-dashes are retained only where they punctuate genuine appositive or
parenthetical content. **Resolved; further polish is a copy-edit judgement call at
proof stage.**

### M2. Chromosome-encoding description now faithful to the code but diverges from the integer-`n` encoding the revision brief assumed

**Location:** §4.3.

**The problem.** R3's expr-17 asked why not "an n×1 chromosome where each cell
represents an individual, valued by the centre they're directed to." The revision
brief instructed rewriting the description to that integer length-`n` encoding. But
the actual solver (`src/presidio_vol_assign/allocation/solvers.py`) implements a
**2·n_dir** flat list: a partial-permutation person slice plus a real-valued centre
slice decoded by `floor(r·n)`. Rewriting the paper to claim an integer length-`n`
encoding would make the manuscript contradict its own released code — a worse
integrity problem than the one being fixed. The integer length-`n` encoding in the
ATRES assessment described the *older* 3-objective humanitarian model
(`fis_humanitarian.py`), not this allocation extension.

**Disposition — resolved in prose; flagged for author confirmation.** §4.3 was
rewritten to be unambiguous and faithful to `solvers.py`: two named slices, every
chromosome feasible by construction, the centre gene explicitly a
mutation-friendly surrogate for a per-person integer in {0,…,n−1}. R3's actual
question is answered directly: the model directs a strict subset (n_dir < m), so
the chromosome must encode *which* people are directed as well as *where*, which is
the structure a one-integer-per-person encoding lacks; restricted to the directed
set, the centre slice *is* that one-integer-per-person encoding. **This resolves
R3's clarity complaint without falsifying the implementation.** Recorded as an
[AUTHOR CONFIRM] item: if a co-author intended to migrate the code to a literal
integer length-`n` encoding, that is a code change for Submission 2, not a prose
edit here.

### M3. Fuzzy primer answers "what is a membership function" but sends the full numeric trace to companion docs

**Location:** §4.1 (new primer).

**The problem.** R3 (p8/p19) wanted fuzzy systems introduced for a non-expert and
specifically asked "what are membership functions? what are triangular/trapezoidal
numbers?" The new §4.1 defines all three plainly and walks the fuzzify → fire →
defuzzify pipeline with a concrete partial-membership example (0.27 Medium / 0.73
High → a modest score). It points to the companion documentation for the full
arithmetic trace. A hostile reading of R3 ("don't make the reader dig through
external material") could object that the numeric centroid is not on the page.

**Disposition — resolved.** The in-text example is self-contained at the
conceptual level a non-expert needs: it shows that an input lands in two levels at
once, that several rules fire, and that the centroid returns a graded number rather
than a binary verdict. Reproducing the full centroid arithmetic in-line would
require committing to one FIS's exact membership breakpoints in body text and risks
divergence from the rule-base appendix; the worked numbers live in the appendix
rule tables and the released documentation, which is the right division for a
journal article. **Resolved.**

### M4. Verify HV now leads every metric discussion and no MID-led sentence survives

**Location:** §5.1, §5.4, Tables 3, B1, B2.

**The problem.** R3 (eq. 18) wanted MID dropped or demoted because it rewards
proximity to the ideal point and penalises legitimate Pareto trade-offs. The
revision moves MID to the last column in all three tables, footnotes it as a
diagnostic, and states HV is primary. The remaining risk is an overlooked
MID-first or MID-equal sentence in the running text.

**Disposition — resolved (verified).** Every metric-comparison sentence now opens
on HV (and CPU as the second binding metric); the one summary sentence that names
MID does so only to say it "adds no separating information." The setup section
states HV is the primary indicator and MID a diagnostic shown last, with the
methodological reason (eq.-18 critique) given in the table note. **Resolved.**

### M5. `H2b` appeared in the text without being defined among the stated hypotheses

**Location:** §4.4 (reference-point construction).

**The problem.** The intro now states H1, H2(a/c), H3(a/b), H4 explicitly — good —
but §4.4 referenced an "H2b" boundary-diversity check that is not in that list, so a
careful reader meets an undefined label.

**Disposition — resolved.** The sentence was reworded to refer to "boundary
diversity proving problematic in the algorithm comparison" without the orphan label.
**Resolved.**

---

## Minor issues

### m1. "Qualitative" overuse — resolved; confirm no incorrect residual

Section title and index descriptions relabelled "fuzzy-aggregated"; a new
terminology sentence states inputs are quantitative and only the FIS output is
qualitative. The four surviving instances of the word are all correct usage
(contrasting the OR literature's crisp measures, defining the term in the primer,
and "quantitative not qualitative differences" between fronts). **Resolved.**

### m2. Outstanding [AUTHOR CONFIRM] placeholders are visible in the back matter

Zenodo DOI, funding statement, AI-use acknowledgement, and CRediT roles are all
bracketed. These are correct to leave as placeholders pre-submission but must not
survive to the final upload. **Tracked, not a defect.**

### m3. "First explicit 4R↔objective mapping" is a negative-existence claim

Already softened to "to the best of our knowledge / in the surveyed literature"
in §1, §2, §3.5 (carried over from the R1 review's C2 fix). Confirm the intro
contribution list inherits the same hedge by reference. **Minor; acceptable.**

---

## Summary

| Severity | Count | Disposition |
|---|---|---|
| Critical | 1 | C1 documented exception (deferred to Submission 2; resolved at framing level) |
| Moderate | 5 | M1–M5 all resolved (M2 additionally flagged [AUTHOR CONFIRM]) |
| Minor | 3 | resolved or tracked |

**Gate status:** every Critical carries a recorded exception consistent with the
Submission-1 scope; every Moderate is resolved. The one substantive deferral (C1,
external baseline) is the explicit subject of Submission 2 and is now signposted in
§8.5 and reframed out of the contribution claim, so the manuscript no longer rests
on the superiority claim the ATRES reviewers attacked. **R2 gate cleared.**
