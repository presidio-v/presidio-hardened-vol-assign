# Response to Reviewer 1 — systems-4669426

**Manuscript:** Is the Optimiser Needed? Auditing a Digital Decision-Support System for Necessity, Decision Stability and Input Fragility under Turbulence (previously: Is the AI Needed? Auditing a Digital AI Decision System for Necessity, Identifiability and Input Fragility under Turbulence)

Dear Reviewer,

Thank you for the report. Each of your four points led to a change in the manuscript, and the revision answers them below in order. All changes are highlighted in the revised manuscript. Section, table and figure numbers refer to the revised version.

The other reviewers' reports also led to changes that touch your points, and we summarise them here because they change what the paper claims:

- The paper now runs steps S1–S3 of the protocol on two further models besides the audited system. One is the ED volunteer-staffing model from which the system grew, where the choices are coupled by a one-to-one matching. The other is a multi-objective knapsack benchmark that is not ours and whose weighted sums are NP-hard (new Section 5.5). The protocol reaches a different verdict on each, including one where the evolutionary search is close to optimal and stable.
- "Identifiability" is now "decision stability". The instability between seeds is decomposed into solver non-convergence, the multiplicity of near-optimal allocations in the model, and the absence of a selection rule. An exact decider with a demonstrated tie-breaking rule commits to one allocation.
- While answering the reviews we found and corrected a defect in our own artefact. The generator of the scaling instances was not reproducible across processes, so the 60 s and 300 s rows of the submitted Table 6 ran on different instances. The deployment envelope has been rerun on fixed, archived instances, with every method on the same instances (details under your comment 1.2).
- The structural check found a second mismatch between the published model's equations and its documentation. As written, the prioritisation objective (ULPP) rewards directing the people of lowest priority, while its documentation says the opposite (Section 5.1). We report it, and a correction to our earlier article is in preparation.

---

**Comment 1.1.** *The mapping between disaster-induced shocks and economic turbulence remains conceptual analogy without empirical evidence from real-world market data. Quantitative audit results cannot be directly interpreted as real-world performance of commercial AI systems under economic turbulence. The manuscript should tone down cross-context quantitative extrapolation and retain only mechanistic analogy.*

We agree, and the revision keeps the analogy at the level of mechanism throughout.

- The title no longer presents the paper as a study of "digital AI systems" under economic turbulence. It now asks whether the optimiser of a digital decision-support system is needed, which is what the experiments test.
- The abstract, the Introduction and Section 3.3 (retitled "Relief allocation and economic turbulence: an analogy of mechanism") state that no market data enter the audit, that the instances are synthetic, and that none of the reported numbers is a statement about commercial AI systems.
- The former "Economic reading" (Section 6.4) has been rewritten to keep only the four questions an organisation would ask. The quantitative statements have been removed.
- The Conclusions now separate what is established analytically, what the experiments support on synthetic instances, and what is proposed for other domains.

**Comment 1.2.** *Scalability boundaries of the repaired MIP formulation are insufficiently discussed … Clear boundaries on when evolutionary meta-heuristics become truly necessary should be supplemented.*

The new Section 6.2, "Where exact optimisation stops", answers this. It gives the size of the formulation (assignment and linking variables, and the CAIL tables of m × n × ($n_{\mathrm{dir}}$ + 1) entries, 1.2 GB at 3000 × 50). It reports where the soft and hard MIP stop proving optimality within one and five minutes, and how long the preparation of the tables takes, which at 3000 × 50 exceeds the solve itself. It then names three conditions under which a metaheuristic becomes necessary:

1. the coupling cannot be tabulated for an exact model;
2. the model outgrows memory at the target size;
3. the decision must be recomputed faster than the exact method can find a good incumbent, under constraints such as hard capacity that a simple fallback cannot respect.

Additional real-world constraints would move all three boundaries towards smaller sizes; we say so and state that we have not tested them.

The envelope behind these statements has been rerun (Table 13, new Appendix B). Every method now runs on the same three stored instances per size and under the same wall-clock limit, at 60 s and, from 1000 people up, at 300 s. The comparison adds two references that know about capacity: a capacity-aware local search on the repaired model and an evolutionary search seeded with exact solutions. Solver time and total decision latency, preparation included, are reported for every method (new Table A2). With five minutes, the hard-capacity MIP finds a solution on all three 3000 × 50 instances but proves optimality on one, and the soft MIP proves none. The local search is within 0.5 units of the MIP on average (0.6 on any instance) wherever the MIP is proven optimal, beats the one-minute MIP from 1500 × 40 on, and stays within 1.5% of it on average under hard capacity, with under 17 s of solver time. With five minutes the MIP overtakes it at 1500 × 40 but not at 3000 × 50. The local search is the first method that "claims the regime" beyond the exact boundary, and the audited evolutionary system, seeded or not, falls behind it.

**Comment 1.3.** *Engagement with existing literature on algorithmic auditing and high-risk-AI regulation is weak … fails to clarify novel checkpoints and incremental methodological contributions compared with prior work.*

Section 2 now places the protocol against these frameworks explicitly:

- A new Table 1 maps each of the five steps against SMACTR (Raji et al.), the ML Test Score (Breck et al.), the EU AI Act (Articles 9, 10, 12 and 15), ISO/IEC 42001 and 23894, and optimisation benchmarking guidelines (Hooker; Bartz-Beielstein et al.; Beiranvand et al.). NIST AI RMF 1.0 is added to the discussion.
- The new checkpoints are named: the structural check (S1), the dominance test of the committed system against an exact reference (S2), the decomposition of decision instability into solver, model and selection policy (S3), and regret measured on a noise-free decider (S4).
- A new subsection relates decision stability to the literature on predictive multiplicity and the Rashomon effect (Breiman; Marx et al.; Semenova et al.; Black et al.) and to modelling to generate alternatives in operational research (Brill et al.).
- The Introduction states what is new beyond the correction of our earlier study.

**Comment 1.4.** *Practical implications remain generic and lack actionable recommendations for humanitarian practitioners and AI procurement stakeholders.*

The new Section 6.3, "Recommendations", gives each group concrete actions.

For procurement and approval, it lists five requests, each answerable before a system is in use:

1. the objectives and constraints written as functions of the decision variables;
2. an exact or bounded reference at the target size, with the share of committed outputs it dominates;
3. the seed floor, the multiplicity of the model at the committed quality, and the stated selection rule;
4. regret per input on a noise-free decider;
5. the decision latency including preparation, and the constraint violations of every method compared.

For humanitarian practitioners, it gives three practices:

1. re-decide on every data update with the deterministic decider, which takes well under a second at operational sizes;
2. check whom the optimum directs, since both mismatches we found are visible in a single exact solution;
3. treat capacity as a constraint wherever overcrowding is a hazard, since no form of the soft objective we tested removed overload.

Kind regards,
Vladimir Stantchev, on behalf of the authors
