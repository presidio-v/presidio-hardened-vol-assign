---
phase: R3 — final adversarial pass (pre-resubmission)
project: presidio-hardened-vol-assign
target-venue: Applied Sciences (MDPI), SI "Innovations in Supply Chain Resilience"
manuscript: pubs/v0.2.0-mdpi/author/paper.tex (revised through R2; clean latexmk build)
review-date: 2026-06-29
reviewer-role: Same hostile senior reviewer, final read. Focus per brief:
                residual clarity, internal numerical/notational consistency,
                Appendices A and B, and MDPI house-style / back-matter completeness.
scope-note: Writing/framing pass only. No new experiments. Numbers are checked for
                internal consistency against each other and the released manifests
                as quoted; none are recomputed.
---

# Adversarial review — Round 3 (final pass)

Severity codes as in R2. This pass assumes the substantive ATRES points were
handled in R1/R2; it hunts for the residual things that get a paper desk-returned
by an MDPI editor or flagged by a careful referee: inconsistent numbers, undefined
notation, appendix/body mismatches, and missing house-style sections.

---

## Critical issues

None. The R2 gate decision stands: the only open Critical (external-baseline
benchmark) is a documented Submission-2 exception, reframed out of the contribution
claim and signposted in §8.5.

---

## Moderate issues

### M1. Rounded CPU-speedup figures were internally inconsistent (30% vs 32%)

**Location:** §8.2, §9 (Conclusions) vs Abstract, §5.3.

**The problem.** The abstract and §5.3 report the speedup precisely (37/32/27% by
size, 32% mean). The discussion and conclusion still said "30%." Round numbers
that disagree with the precise figure elsewhere are exactly what a referee circles.

**Disposition — resolved.** Both "30%" instances were replaced with "32% on
average (27--37% across sizes)" in §8.2 and "32% mean CPU advantage" in §9, so every
mention now traces to the same manifest figure. **Resolved.**

### M2. MDPI back-matter was missing the Institutional Review Board and Informed Consent statements

**Location:** back matter.

**The problem.** MDPI's template requires an Institutional Review Board Statement
and an Informed Consent Statement in every research article, even when the answer is
"Not applicable." Their absence triggers an editorial pre-check return.

**Disposition — resolved.** Both statements were added with "Not applicable" and a
one-line justification (synthetic instances; no human or animal subjects). The back
matter now runs Data Availability → Author Contributions → Funding → IRB → Informed
Consent → Acknowledgments (with the AI-use disclosure) → Conflicts of Interest,
matching MDPI order. **Resolved.**

---

## Minor issues

### m1. Appendix A rule tables vs body counts — consistent (verified)

The body states rule-base sizes 27 / 9 / 3 / 27 for FIS$_1$ / FIS$_{2a}$ /
FIS$_{2b}$ / FIS$_3$ and 9 for the baseline FIS$_2$. Appendix A enumerates 27 (FIS$_1$,
verified by row count), 9 (FIS$_{2a}$), 3 (FIS$_{2b}$), 27 (FIS$_3$), and 9 (baseline
FIS$_2$). The H3a sweep text ("66 single-rule deletions across the four FIS rule
bases: 27+9+3+27") sums correctly to 66. **Consistent.**

### m2. Run-count arithmetic — consistent (verified)

"540 runs" = 3 sizes × 3 algorithms × 2 formulations × 30 reps. The four-objective
"270-run cell" = 3 × 3 × 30. The Data Availability statement's "540 H1+H2+H4 runs,
100 H3b samples, 66 H3a deletions" matches the experiment descriptions. **Consistent.**

### m3. Wilcoxon equivalence now stated in both setup and Appendix B

R3 (p36–37) noted most MOEA studies just use the Wilcoxon rank-sum test. The
equivalence "Mann–Whitney U = Wilcoxon rank-sum" is now stated in §5.1 and again in
the Appendix B preamble where the per-pair tables live, pre-empting the objection at
the point the reader meets the tables. **Resolved.**

### m4. "Five FIS" vs "four FIS" — not a contradiction

The four-objective model uses four FIS; the shared output universe and the H1
comparator add the baseline FIS$_2$, so "all five FIS" (cross-objective scale) and
"four control systems" (the 4-obj model) are both correct in context. A pedant might
still pause; left as-is because spelling it out each time would clutter. **Judgement
call, not a defect.**

### m5. Residual [AUTHOR CONFIRM] placeholders

Zenodo DOI (Abstract contribution 5, Data Availability, Conclusions), Funding,
AI-use acknowledgement, and CRediT roles remain bracketed by design. These must be
filled before upload; none can be resolved by the assistant. **Tracked in the final
summary and reviewer-response.**

---

## Summary

| Severity | Count | Disposition |
|---|---|---|
| Critical | 0 | — |
| Moderate | 2 | M1, M2 resolved |
| Minor | 5 | resolved or tracked (consistency checks passed) |

**Gate status.** No open Critical. Both Moderates resolved. Internal numerical and
notational consistency checks (rule counts, run counts, percentage figures, FIS
counts) pass. MDPI back-matter is complete except for the bracketed author-supplied
values. The manuscript builds clean under `latexmk` with zero warnings and zero
undefined references. **R3 gate cleared; manuscript ready for author review and
resubmission once the [AUTHOR CONFIRM] fields are filled.**
