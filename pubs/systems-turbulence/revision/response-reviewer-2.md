# Response to Reviewer 2 — systems-4669426

**Manuscript:** Is the Optimiser Needed? Auditing a Digital Decision-Support System for Necessity, Decision Stability and Input Fragility under Turbulence (previously: Is the AI Needed? Auditing a Digital AI Decision System for Necessity, Identifiability and Input Fragility under Turbulence)

Dear Reviewer,

Thank you for a thorough report. Your seven points asked for new evidence, not only new wording, and most of the revision's new experiments answer them. They are: a second and third model; near-optimal multiplicity; stronger heuristic baselines with full latency accounting; joint and bias perturbations across instances and conditions; seed counts and confidence intervals; a rerun of the envelope on matched instances; and an analysis of the repair's form. While doing this work we also found a defect in our own artefact and a second mismatch in the published model, and we report both openly below. All changes are highlighted in the revised manuscript, and section, table and figure numbers refer to the revised version.

---

**Comment 2.1 (novelty; one system, our own; S1 settles necessity at once).**

We agree that the separability itself is elementary. What is new does not follow from it, and the revised Introduction states it explicitly:

1. a protocol whose steps each end in a checkable number and branch differently on models of different structure;
2. a decomposition of decision instability into solver variance, model multiplicity and selection policy, with a demonstrated tie-breaking rule;
3. input fragility measured on a noise-free decider, which separates it from the solver's own variance;
4. a deployment envelope that compares exact and heuristic methods at equal end-to-end latency.

The correction of [10] is a consequence of the protocol, not its contribution.

As you asked, steps S1–S3 of the protocol now run on two further models where S1 does not settle necessity (new Section 5.5, Table 14):

- **ED volunteer staffing.** The model from which the relief tool grew is our own but structured differently: a one-to-one matching couples the choices, so it is not separable. S1 names the problem class, a linear assignment problem, and S2 solves it exactly in polynomial time. Averaged over seeds, the evolutionary system's best point comes within 0.2% of the optimum on the small instance (11 of 20 seeds reach it exactly) but not on the large one (9% worse, seed floor 0.90).
- **Multi-objective 0/1 knapsack.** This is the benchmark of Zitzler and Thiele, which we did not design. Every item is coupled by the capacity constraints and every weighted sum is NP-hard. HiGHS solves each weighted sum exactly in about 0.3 s at 250 and 500 items, and NSGA-II with the standard repair operator does well: at 2000 generations its committed decision is within 0.3% of the exact optimum, and two seeds differ by a Jaccard distance of only 0.06–0.07.

The three models take three different paths through the protocol, and on the third the protocol does not condemn the evolutionary search. We could not obtain an open third-party optimisation-based decision system in operational use within the revision window. The knapsack is a benchmark, not such a system, and we name this as the first limitation (Section 6.6).

**Comment 2.2 (title and scope; "deployed"; fit to the Special Issue).**

- **Title.** Toned down to what is tested: "Is the Optimiser Needed? Auditing a Digital Decision-Support System for Necessity, Decision Stability and Input Fragility under Turbulence".
- **Abstract and keywords.** They now present the case as post-disaster relief on synthetic instances, with the economic setting engaged only through an analogy of mechanism ("supply chain" has been dropped from the keywords).
- **"Deployed".** Removed. The Introduction now says that the tool has been released and tested on synthetic instances, that we know of no operational use, and that we do not claim one. "Deployment envelope" remains as the name of S5, which concerns the conditions under which a method could be deployed.
- **Fit to the scope.** A new paragraph at the end of the Introduction explains it: the audit treats a digital decision-support system as a whole, from data to the committed allocation, at the point where an organisation decides whether to adopt it for a turbulent environment.

**Comment 2.3 (elements defined only in [10]; symbols undefined; algorithm parameters).**

A new Section 3.2, "The model", makes the paper self-contained:

- the inputs with their encoding and generator distributions (Table 2);
- all three fuzzy systems with inputs, terms, breakpoints, rule counts, inference and defuzzification (Table 3);
- the definitions of $\mathrm{FIS}_3$, $\mathrm{COR}_i$, $\mathrm{RDR}_i$ and $\mathrm{TD}_{j,i}$;
- the objectives as equations;
- the encoding, including how exactly $n_{\mathrm{dir}}$ directed people are selected, and the variation operators exactly as implemented;
- the crisp priority rule as equations;
- a table of every solver setting in every run (Table 4).

Writing this section from the code also brought out two points we now state. The NRGA variant of the artefact truncates the last front at random instead of using ranked roulette selection, and the instance generator draws all fields independently.

**Comment 2.4 (fairness of the baselines; budget; timing accounting; a stronger heuristic).**

- **Budget.** The budget of 150 generations is the artefact's default, close to the 200 generations of the published experiments (population 100); the budget curve to 9600 generations makes the choice immaterial for the conclusions. The budget curve already ran to 9600 generations and the population controls to 240,000 evaluations, and both rows now have 20 seeds (Table 6).
- **Stronger references.** The envelope (Section 5.4, Table 13) adds two, as you suggested:
  - a capacity-aware local search on the repaired model, with re-route and swap moves evaluated exactly from the CAIL-per-load tables, started from the fallback or from a capacity-respecting greedy allocation;
  - NSGA-II with its initial population seeded with the deterministic fallback and the exact optima of the published model at 21 weight vectors.
- **Timing.** The accounting is now equal and complete. Each MOEA is stopped by a wall-clock deadline rather than a calibrated generation count. Here we must correct the submitted version: under the "300 s" limit, the calibrated NSGA-II ran 307.8 s at 1000 × 25, 263.2 s at 1500 × 40 and only 182.7 s at 3000 × 50. All methods used the compiled fuzzy evaluator. The fuzzy scores and the CAIL-per-load tables are now timed separately, and solver time and total decision latency are reported for every method (Table A2). The paper now states the correction of the submitted timing in Section 4.5.
- **The softened statement.** Up to 600 × 20 both MIPs are proven optimal within a minute, and the local search comes within 0.2–0.5 units of the soft optimum in under a second. Beyond the exact boundary, the local search returns better allocations than the one-minute MIP (96.5 against 98.8 at 1500 × 40; at 3000 × 50, all three instances against one solved by the MIP), and under hard capacity it stays within 1.5% of the MIP on average with under 4 s of solver time, while the hard MIP at 3000 × 50 needs 115 s on the one instance it proves and reaches the limit on the other two. With five minutes the soft MIP overtakes the local search at 1500 × 40 but not at 3000 × 50. The seeded NSGA-II improves on the plain one by up to 44 units but ends near its seeds and inherits the fallback's overload. The plain NSGA-II stays behind the fallback at every size; it overloads the centres least, because it spreads people almost at random. The statement you quote ("only a capacity-aware heuristic could claim that regime") has been replaced by the measured result.

**Comment 2.5 (churn: non-convergence vs. a flat landscape; "by construction"; tie-breaking; Rashomon literature).**

We quantified the allocations within ε of the exact optimum as you proposed, by modelling to generate alternatives (Section 4.3, Eq. (10), Figure 4).

- **Flatness near the optimum.** Within one objective unit of it (about 1%), allocations differ from the optimum for 26% of the directed people (small) and 33% (large). Within five units, for 62% and 73%.
- **At the evolutionary system's own quality deficit.** The model admits allocations that share no (person, centre) pair with the optimum, and directed sets with a Jaccard index of 0.08 and 0.
- **What this means for the seed floor.** The floor of 0.83 and 0.94 therefore sits in a region where the model itself allows almost any allocation. Non-convergence and a flat landscape are not alternatives here: the solver's distance from the optimum is what exposes the flatness. Near the optimum, the remaining multiplicity is the model's own.
- **"By construction".** You are right that it holds only under a fixed tie-breaking rule. The exact optimum is unique on the small instance, but on the large one alternative optima differ for 1%, 6% and 22% of the directed people (published model, soft and hard capacity). The paragraph is now in the main text (Section 5.2), and the tie-breaking rule is demonstrated (Table 9): over ten random presentations of the same instance, the plain hard-capacity MIP returned ten different optimal allocations, and the rule returned one.
- **Related literature.** Decision stability is now related to predictive multiplicity and the Rashomon effect (Breiman; Marx et al.; Semenova et al.; Fisher et al.; Black et al.) and to modelling to generate alternatives (Brill et al.; Danna et al.) in a new subsection of Section 2.

**Comment 2.6 (one field at a time; level justification; level 0 in the trend tests; few realisations; mixed seed counts in Table 2).**

- **Trend tests.** They use levels 0.05–0.4 only. Level 0 never entered them, and the paper now says so (Section 4.4). Recomputed from the archived manifests, the reported ρ = 0.37–0.97 and Holm-adjusted p < 10⁻⁶ (largest 4.8 × 10⁻⁸) are the values without level 0. With level 0 they would be 0.44–0.98. The script that computes them is now in the artefact.
- **Joint perturbations.** Three are added: all four continuous fields with noise; the transport triple of travel-duration noise with road-condition and hazard flips; and all eight single-field cells at once. Systematic bias in both directions is added as well. Across 36 conditions per size, the regret of every noise, missing-value and flip cell grows with the level in every condition (all tests significant after Holm correction), and the ranking of the inputs is unchanged. Joint perturbations add up rather than amplify each other: perturbing all eight cells at once at level 0.4 costs 29 (small) and 32 (large) units, 0.78–0.79 of the sum of its single-field parts (new Table 10 and Figure 5).
- **Level justification.** We found no empirical source for per-field error magnitudes. The paper now calls the levels a sensitivity grid, cites the evidence that does exist (inventory-record inaccuracy in retail; information gaps after the Haiti earthquake), and drops the claim that 0.05–0.1 is "most relevant in practice".
- **Generalisation.** The exact-decider study now covers six instances per size, three demand conditions and two occupancy conditions.
- **Seed counts.** Every row of Table 6 (formerly Table 2) now has 20 seeds and a cluster-bootstrap confidence interval. The conclusions are unchanged. At 9600 generations the exact point still dominates 32% (small) and 97% (large) of the front points, and the seed floor falls to 0.50 [0.47, 0.54] on the small instance but stays at 0.90 [0.88, 0.91] on the large one. Two shares moved for a reason we now state in the caption: the hypervolume reference point is 1.1 times the maximum over all fronts of an instance, so adding seeds moves it, and the large instance's share at 150 generations reads 0.21 instead of 0.18. The evolutionary turbulence analysis keeps its 12/8 realisations and 8/6 seeds, which we state as a limitation, because its role in the revised argument is only to show that the evolutionary system's turbulence response is buried in its seed floor.

**Comment 2.7 (single-instance 300 s rows; mixed instance sets; fallback overload; repair form; abstract).**

- **The envelope rerun.** It now uses three instances per size at both limits, and every method runs on the same instances. While doing this we found that the old 300 s and 60 s rows were not on the same instances at all. The instance generator salted its random stream with Python's per-process string hash, so each run of the scaling script generated new instances. The generator is fixed and tested, and the instances are stored with their SHA-256 hashes.
- **Unsolved instances.** These are counted, never averaged (Table 13, Figure 6), and Appendix B reports every instance. Overload is given for every method, including NSGA-II, the seeded NSGA-II and the fallback (Table 13 and Table A2).
- **The repair's form.** The linear form is the definition of an occupancy rate, since $\ell_i/\mathrm{cap}_i$ is the share of the centre a decision fills. We have nonetheless solved the RQ4 grid under four alternatives: load counted double; convex and concave forms $(\ell/\mathrm{cap})^\gamma$ with γ = 2 and 0.5; and the rule base's response to load tripled (Section 5.4, Table 12). The linear form reproduces the submitted Table 5 (now Table 11) exactly. Under soft capacity, overload persists in 29 of 30 cells across the forms, including the tripled response. The paper's explanation is therefore refined: the binding limit is that CAIL saturates once a centre is full, not only its small range, and hard capacity remains what makes the optimum feasible.
- **The abstract.** It now reflects the three-instance result. The submitted sentence on proven optimality at 3000 people in five minutes rested on a single instance. On three instances the hard-capacity MIP proves optimality on one, with gaps of about 0.02% on the other two, and the abstract no longer makes the claim.

---

**Two further corrections found during the revision.** Both are stated in the paper, because they bear on how far our earlier work, and this audit, can be trusted:

1. The reproducibility defect in the instance generator described under 2.7. The reference instances were always archived files and are unaffected.
2. The structural check found that the published model's prioritisation objective, as written, rewards directing the people of lowest priority. The rule base gives the highest ULPP to the most vulnerable, and the objective minimises mean ULPP over those directed, while the documentation says the opposite. The exact optimum directs only 12% (small) and 7% (large) of the people the crisp rule ranks highest (Section 5.1). A correction to [10] covering this and the CAIL description is in preparation.

Kind regards,
Vladimir Stantchev, on behalf of the authors
