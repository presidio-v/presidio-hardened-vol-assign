# Response to Reviewer 3 — systems-4669426

**Manuscript:** Is the Optimiser Needed? Auditing a Digital Decision-Support System for Necessity, Decision Stability and Input Fragility under Turbulence (previously: Is the AI Needed? Auditing a Digital AI Decision System for Necessity, Identifiability and Input Fragility under Turbulence)

Dear Reviewer,

Thank you for a careful and constructive report. We have followed every recommendation, and three of them changed the paper's argument as well as its wording: the distinction between the optimiser and the decision support, the correction of the reference front, and the separation of decision stability from uniqueness and accountability. The answers below follow the order of your report. All changes are highlighted in the revised manuscript, and section, table and figure numbers refer to the revised version.

Changes made in answer to the other reviewers that bear on your points:

- Steps S1–S3 of the protocol now also run on two further models (new Section 5.5): the ED volunteer-staffing model from which the system grew, where the choices are coupled by a one-to-one matching, and a multi-objective knapsack benchmark that is not ours and whose weighted sums are NP-hard. On the knapsack, the evolutionary search turns out to be close to optimal and stable, and the protocol says so.
- The structural check found a second mismatch between the published model's equations and its documentation: as written, the prioritisation objective (ULPP) rewards directing the people of lowest priority (Section 5.1). A correction to our earlier article is in preparation.
- The deployment envelope has been rerun on fixed instances after we found that the generator of the scaling instances was not reproducible across processes (details under "Clarify the deployment comparison").

---

**Clarify what the study establishes about the necessity of AI.**

You are right that the experiments establish that the stochastic optimiser is unnecessary, not that the decision support is. The exact decider consumes the same fuzzy scores. The revision makes the distinction in the title ("Is the Optimiser Needed? Auditing a Digital Decision-Support System …"), the abstract, the Introduction, the Discussion (Section 6.1) and the Conclusions. The Discussion adds that removing the search does not show the rule bases to be right, and the ULPP finding shows that one of them points the wrong way. The positioning against optimisation benchmarking and algorithmic auditing is now explicit, and the turbulence study is framed as a sensitivity grid (Section 4.4): a new Table 1 maps each step against SMACTR, the ML Test Score, the EU AI Act, ISO/IEC 42001 and 23894, and benchmarking guidelines, and the text names the checkpoints that are new.

**Correct the description of the "exact supported front."**

Agreed. Throughout the paper, the front is now a "sampled supported reference front". Section 4.2 states that every point is exactly optimal for its weights, but that a finite lattice need not recover every supported non-dominated point, an omission separate from the unsupported points. We ran the finer-grid sensitivity analysis you suggested (new Table 8). Refining the lattice from 21 to 161 steps per axis raises the number of supported points from 181 to 1443 (small) and from 197 to 3359 (large), so the coarse lattice found only a fraction of them. Yet the front's hypervolume changes by at most 1.1%, and the evolutionary fronts' hypervolume share by at most 0.01. The dominance findings and the hypervolume comparisons are stable.

**Separate decision stability from uniqueness and accountability.**

We adopted "decision stability" in place of "identifiability" throughout, and the revised Section 4.3 and Section 5.2 separate the three sources of instability you name.

- *Multiple optimal or near-optimal solutions.* We measure the multiplicity of the model directly by modelling to generate alternatives. Within ε of the exact optimum, we find the allocations that share the fewest pairs or directed people with the optimum and with each other (new Eq. (10), new Figure 4). Within one objective unit of the optimum, about 1% of its value, allocations differ for 26–33% of the directed people. At the evolutionary system's own quality deficit, they can differ for all of them.
- *Solver non-convergence.* The seed floor of 0.83 and 0.94 therefore sits where the model allows almost any allocation. The solver's distance from the optimum is what exposes the multiplicity.
- *Selection policy.* The exact optimum is unique on the small instance but not on the large one, where alternative optima differ for 1%, 6% and 22% of the people (published model, soft and hard capacity). This is the mechanism behind your observation on Appendix C of the submitted version (now Appendix D).

The tie-breaking rule is now demonstrated, not only recommended (new Table 9). Presented with the same large instance in ten random orders, the plain hard-capacity MIP returned ten different optimal allocations. With the stated rule, a fixed seeded key per (person, centre) pair minimised in a second lexicographic stage, it returned one in every case.

Section 5.2 also takes up your point that variation alone does not make an output illegitimate. A documented randomised policy can be fair and accountable, and a deterministic rule can select arbitrarily. What an audit trail needs is a stated selection policy, and a seeded search does not supply one.

**Provide the essential methodological details within the manuscript.**

A new Section 3.2, "The model", defines everything the results depend on:

- the inputs, their encoding and the instance generator's distributions, with the dependencies between inputs (Table 2);
- the fuzzy systems (inputs, terms, breakpoints, rule counts, inference and defuzzification);
- the objectives as equations;
- the encoding and variation operators exactly as implemented;
- a table of all solver settings for every run.

It also gives the items you list:

- **The crisp priority baseline.** Its ranking score and assignment rule, as equations.
- **Allocation churn.** The formula and denominator. The denominator is the reference decision's directed set, so a person directed only in the other allocation does not enter the numerator. Because both allocations direct exactly $n_{\mathrm{dir}}$ people, the measure is nonetheless symmetric in value.
- **The evolutionary representation, operators and parameters.** As above, with the settings table.
- **The instance generation.** Its distributions and dependencies.
- **The effect "above" the seed baseline.** The exact calculation: churn above the floor, and a quality harm paired by seed and scaled by the clean-seed standard deviation.

**Systematic bias** results are now reported. Bias in both directions on the four continuous fields enters the extended turbulence study (Section 5.3). Bias of a given size costs about two thirds as much as noise of the same size on resource time, between half and the full noise effect on travel time, and half or less on damage and occupancy, in either direction. With one realisation and four levels per condition its trend tests have little power, so the paper reports its magnitude.

**Strengthen the basis for generalisation.**

The exact-decider turbulence study has been repeated on five further generated instances per size, under three demand conditions ($n_{\mathrm{dir}}$ = m/6, m/3, m/2) and two occupancy conditions (as generated, and every centre's occupancy raised by 25 points). It now also includes three joint perturbations, applied together at the same level. Across 36 conditions per size, the regret of every noise, missing-value and flip cell grows with the level in every condition (all tests significant after Holm correction), and the ranking of the inputs is unchanged. Joint perturbations add up rather than amplify each other: perturbing all eight cells at once at level 0.4 costs 29 (small) and 32 (large) units, 0.78–0.79 of the sum of its single-field parts (new Table 10 and Figure 5). We removed the claim that levels 0.05–0.1 are "most relevant in practice" and the description of the fragility as "manageable". Section 4.4 now calls the levels a sensitivity grid rather than estimates of real error rates, and cites the empirical evidence we found on data-quality errors in supply chains and humanitarian operations, which establishes that records are often wrong but not by how much per field. The conclusions are stated explicitly as findings about synthetic instances and model-based losses.

**Clarify the deployment comparison and treatment of unsuccessful solves.**

We reran the envelope, and in doing so found a defect we should report. The generator of the scaling instances salted its random stream with a hash that Python randomises per process. Each run of the scaling script therefore regenerated different instances, and the 60 s and 300 s rows of the submitted Table 6 ran on different instances. The generator now uses a stable salt, and its reproducibility is tested. The instances are stored and archived with their hashes, and every method and time limit runs on the same three instances per size, including the 300 s limit, which was a single instance before.

- **Unsuccessful solves.** An instance on which the MIP finds no incumbent is reported as unsolved and never averaged. Table 13 shows solved and proven counts, Figure 6 marks unsolved instances, and Appendix B gives every instance separately.
- **The single-instance boundary.** The five-minute statement in the abstract now reflects the three-instance result.
- **The fallback in a combined strategy.** HiGHS through SciPy cannot take a MIP start, so we tested the combination through a capacity-aware local search that starts from the fallback. Started from the fallback, the local search comes within 0.2–0.5 units of the proven soft optimum on average up to 600 × 20, in under a second. From 1500 × 40 on it beats the one-minute MIP, and under hard capacity it stays within 1.5% of the MIP on average with under 4 s of solver time. This is the combined strategy you suggested, and Section 6.2 recommends it.
- **Latency.** Optimisation time and preparation are now reported separately for every method: the fuzzy scores, and the CAIL-per-load tables, which dominate at 3000 × 50. Total decision latency is reported alongside, for every method (Table A2).

**Qualify the operational interpretation of the results.**

Agreed on both points. Section 6.1 now states that regret is a loss in the model's own objectives rather than in humanitarian outcomes, and that lower objective values show better optimisation of the specified model; they do not show better real-world decisions. It also states that making occupancy endogenous did not reduce the maximum load or the overload in every case, and that only a hard capacity constraint removed overload. The repair-form analysis requested by another reviewer supports this: under soft capacity, overload persists in 29 of 30 cells across five forms of the repair, including one with a tripled response to load. The paper now distinguishes correcting the model's structure from demonstrating improved operational safety.

**Smaller remarks.**

- **Figures 4 and 5.** Now Figures 5 and 6, both redrawn with larger labels; Figure 6 marks unsolved instances.
- **Conclusions.** They now distinguish findings established analytically, findings supported by the experiments, and implications proposed for other domains.

Kind regards,
Vladimir Stantchev, on behalf of the authors
