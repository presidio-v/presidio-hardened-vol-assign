# Response to Reviewer 2, second round — systems-4669426

**Manuscript:** Is the Optimiser Needed? Auditing a Digital Decision-Support System for Necessity, Decision Stability and Input Fragility under Turbulence

Dear Reviewer,

Thank you for a second careful report. You were right on every point. Point 1 asked for new evidence, and we ran it; it changes one of our conclusions, which we now report. Point 3 led to a diagnosis that differs from the mechanism you proposed, and we give it below. All changes are highlighted in the revised manuscript. Section, table and equation numbers refer to the revised version.

---

**Comment 1 (RQ2–RQ4 not repeated under the corrected ULPP; status of the correction to [10]).**

We reran RQ3 and RQ4 under the corrected objective, as you suggested, and report the results in a new Section 5.5, "RQ3 and RQ4 under the corrected ULPP objective".

- **Implementation.** The code and all reruns are archived as version 0.6.1 (doi:10.5281/zenodo.24250168). The corrected objective is the mean ULPP over the people left out. With $n_{\mathrm{dir}}$ fixed, it is still a per-person term, so the model stays separable and the exact greedy and the MIP remain exact. Both are verified against brute-force enumeration under the corrected objective. As a control, rerunning the published objective with the same code reproduces every quality, load and overload value of Table 11.
- **RQ3, reference instances (50 realisations per level, as before).** Regret still grows with the level in every single-field cell at both sizes (one-sided Spearman, ρ = 0.37–0.97, Holm-adjusted p < 10⁻⁷). **The ranking of the inputs changes, however.** Under the published objective, the person-side inputs (resource time and damage level) were among the costliest. Under the corrected one, their regret at level 0.4 falls by 59–74%: from 7.2 to 2.8 (resource time) and 5.9 to 1.5 (damage) on the small instance, and from 7.3 to 3.0 and 5.6 to 2.0 on the large one. Travel time, centre occupancy and road condition now lead.
- **Why the ranking changes.** The corrected ULPP is a mean over the $m - n_{\mathrm{dir}}$ people left out. Per person moved, it therefore changes by $n_{\mathrm{dir}}/(m-n_{\mathrm{dir}}) = 1/2$ as much as the published mean over the directed. The person-side inputs act on the objective only through ULPP, so their leverage halves.
- **RQ4 (Table 11 grid).** The grid gives the same three results. Every solve is proven optimal in at most 3.3 s.
  - The soft repair penalises the separable optimum: by 1.1–3.5% on the small instance and 10.5–11.0% on the large one.
  - Overload persists under soft capacity at every κ: 9.7–35.2 people on the small instance and 70.6–89.9 on the large one.
  - Hard capacity removes the overload, and the separable optimum violates it by 21–67 people.
- **Who is directed.** The corrected optimum directs people with more damage (67.8 against an average of 60.1, small) and less resource time left (17.0 against 24.6 h). It shares 22% (small) and 14% (large) of its directed people with the published optimum and 54–55% with the crisp priority rule. The crisp rule's gap to the optimum shrinks from 40 and 44 units to 12.8 and 12.5.
- **36-condition study.** We also repeated the generalisation study of Table 10 (six instances per size, three demand and two occupancy conditions) under the corrected objective. It is reported in Section 5.5 and Supplementary Table S11.
  - Person-side regret at level 0.4 falls by more than half: resource-time noise from 7.4 to 3.0 (small) and 7.5 to 3.2 (large), damage noise from 5.1 to 1.7 and 5.7 to 2.3. Occupancy and travel-time noise lead.
  - Every noise, missing-value and flip cell still grows with the level in all 36 conditions on the large instances. On the small ones this holds for six of the eight cells, and for occupancy noise and occupancy missing values in 33 and 28 of 36 conditions. The paper states this.
  - Joint perturbations still add up: all eight cells at once cost 0.75 (small) and 0.81 (large) of their single-field parts.
- **What changes in the paper.**
  - Section 6.3 now asks auditors to recompute the per-input ranking whenever the objective is corrected.
  - Sections 5.1 and 5.5 and the Limitations say what was rerun and what was not. Not rerun: the evolutionary runs of RQ1–RQ2, the repair-form variants of Table 12, the near-optimal enumeration and the envelope, whose local search and alternative-generation code implement the published form only.
  - Section 5.1 also corrects one sentence. For ULPP alone, the left-out mean and the mean of 100 − ULPP over the directed give the same optimiser. In the weighted sum they differ in scale by the factor above. We use the left-out form, which our correction request for [10] proposes.
- **Status of the correction to [10].** On 8 November 2026 all three authors submitted a correction request to *Applied Sciences*. It is under editorial review, and we will inform the editor as soon as it is decided. It corrects the ULPP objective and the description of CAIL, and it recomputes every result of [10] that depends on the objective (CPU times are kept). The findings on two hypotheses change in detail and the other verdicts hold. Section 6.5 now says this and cites the archive of the correction's code and results (doi:10.5281/zenodo.24061353). Once the correction is published, we will ask that the proof of this article cite it, so that the two papers agree in the published record.

**Comment 2 (equal end-to-end latency vs. equal solver time).**

You are right. The limit binds the solver only. The Introduction now describes the envelope as comparing methods "under equal solver-time limits", with total decision latency, preparation included, reported separately. The positioning table and Section 4.5 say the same.

Section 5.4 now also states what would change if total latency were the binding limit:

- The hard-capacity local search would meet a one-minute total budget up to 1000 × 25 (28 s) and would be at the limit at 1500 × 40 (60–65 s).
- At 3000 × 50, no method that sees the load would meet it, because building the load tables alone takes about 90–165 s.
- Only the fallback, which ignores capacity, would remain (about 10 s).

Under such a limit, the binding cost is the preparation, not the search.

**Comment 3 (seeded NSGA-II worse with 300 s than with 60 s).**

The decline is real, but its cause is different from the one you suggest, and we checked it directly.

- **What the column reports.** The NSGA-II columns of Tables 13 and S10 (formerly A15) report the best point of the final front by equal-weight sum, not the decision committed by Eq. (6). The min–max normalisation of Eq. (6) therefore does not enter these numbers. The captions now say so.
- **The trace.** We re-ran the seeded NSGA-II on the 1500 × 40 instance with seed 0 and recorded, in every generation, the best equal-weight sum in the merged population and among the survivors.
  - The run reproduces the archived final value (100.119).
  - The best point, 99.359, is one of the exact seeds, and the search never improves on it.
  - Within the first generations, the whole population of 100 becomes non-dominated. At generation 1847, just after the 60 s run stopped (generation 1738), crowding-distance truncation discards this seed in favour of a new non-dominated offspring.
  - The best sum then stays at 100.119 to the end.
- **The other affected instances.** On the other two 1500 × 40 instances, the 60 s best point is likewise the best exact seed, unchanged. After 300 s it is the second-best seed on one of them and worse than both on the other. On the affected 3000 × 50 instance, the 60 s best point is the best seed (99.911) and the 300 s best point is the second-best seed (100.434). In all four cases, the search never improved on its seeds; it lost the best one.
- **What this means.** NSGA-II's survivor selection keeps non-domination and spread, not quality in the equal-weight sum. A longer run can therefore lose a good point it was given.
- **Why it matters for the argument.** This bears on decision stability in the way you suspected, though not through Eq. (6): more compute does not make the evolutionary system's output monotone in quality. Section 5.4 now reports this.

**Comment 4 (soft capacity is not operationally feasible; base the envelope conclusions on hard capacity).**

We agree. Section 5.4 now states, before the envelope results, that every method overloads the centres under soft capacity (753 of 1000 directed beyond free capacity for the local search at 3000 × 50). We draw the conclusions from the hard-capacity results and keep the soft ones as a diagnostic.

On that basis, the Abstract, the Introduction, the answer to RQ4, Section 6.2 and the Conclusions now base the envelope on hard capacity:

- With hard capacity, the local search stays on average within 0.8–1.4% of the MIP at every size (at most 2.1% on any instance) on under 4 s of solver time.
- At 3000 × 50 it returns a feasible allocation where the MIP finds none within a minute of solver time (Sections 5.4 and 6.2).
- The audited evolutionary system, which runs on the soft model, trails at every size.

We also qualify "under one minute" as solver time, in line with Comment 2: total latency at 3000 × 50 is about 3 min.

**Comment 5 (the "search needed" branch; Table 14 measures; "decomposition"; MGA extremes).**

- **The "search needed" branch.** The Limitations and the end of Section 5.6 (formerly 5.5) now state that none of the three models reaches the branch in which a stochastic search is needed, because each has an exact method that is practical at the sizes tested. That branch is not demonstrated.
- **Table 15 (formerly Table 14).**
  - The caption now states that the measures differ by row, so the Gap and Seed floor columns compare budgets and solvers within a model, not models with each other.
  - Footnote marks give each row's gap definition (a–c) and stability measure (d–f).
  - The caption also notes that the small ED instance's churn moves in steps of 0.2.
- **"Decomposition".** It is now "qualitative separation" in the Introduction, Section 2.7, Section 4.3, Section 5.2 and the Conclusions. Section 4.3 also states that the three sources interact and are not assigned additive shares.
- **MGA.** Section 4.3 and the Limitations now say that the near-optimal enumeration finds the most different allocations within each tolerance. It is an upper bound on how far two good allocations can differ, not a typical pair.

**Comment 6 (versions, timing wording, numbering, rule tables, length).**

- **Versions.** Both statements were correct but referred to different environments. Section 4.6 gives the local machine on which all reported runs were made (NumPy 2.4.4, DEAP 1.4.3, as recorded in every manifest). Appendix D (formerly K) gives the continuous-integration matrix, whose fresh environments resolved later patch releases (NumPy 2.4.6, DEAP 1.4.4). Section 4.6 now says this explicitly.
- **scikit-fuzzy.** Reference [44] now cites version 0.5.0 (doi:10.5281/zenodo.13372212), the version used.
- **"In milliseconds".** Now "in well under a millisecond", in line with the 0.01–0.02 ms of Section 5.1.
- **Appendix and table numbering.** This was our error. The source lacked the template's `\appendixstart`, so the appendix counters were not reset. The appendices now run from A to D, and their tables from A1. The "Table A2" of our first response is the timing table; with the per-instance table moved to the Supplementary Materials, it is now Table A1.
- **Rule tables.** The full rule bases of all five fuzzy systems are now Supplementary Tables S1–S9. They are generated from the rule constants of the implementation, so they are the rules every result was computed with.
- **Length.** The per-instance envelope table (formerly in Appendix I) has moved to Supplementary Table S10. We have kept Table 10 in the main text, because it is the evidence for the generalisation of RQ3 that you and Reviewer 3 asked for in the first round. The new results for Comment 1 added about two pages, which the moves to the Supplementary Materials do not fully offset.

Kind regards,
Vladimir Stantchev, on behalf of the authors
