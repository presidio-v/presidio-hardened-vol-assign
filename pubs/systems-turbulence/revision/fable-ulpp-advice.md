# ULPP: history, the grouping hypothesis, and the intended reading

Advice for Vladimir Stantchev, 2026-11-03. Read-only review of the sources listed at the end; nothing in any repository was changed.

## 1. Short answer

1. ULPP is not in the older Rabiei papers. It first appears in the 2026 ATRES manuscript (both versions identical on this point) and carries unchanged into *Applied Sciences* 2026, 16, 7581: per-person FIS output, objective = mean over the **directed** people, **minimised**. The direction never changed between versions. The closest precedent, Z1 of the ESWA 2023 volunteer paper, is also "mean over the selected pairs, minimised", but its FIS output depends on the decision (the volunteer's skill), so there it is sound. The relief model kept the template and dropped the decision-dependent input; that is where the sign went wrong.
2. The grouping or equity reading has no support in any equation or line of code. ULPP_j is a function of person j alone, and the objective is an arithmetic mean over the directed set with the centre index summed out. The value cannot depend on who is grouped with whom or on which centre anyone goes to. The words "equitable", "inequity", "misallocation" appear only as linguistic labels and in prose.
3. Most likely intended reading: **(a)**, ULPP_j is the unfairness incurred if person j is *not* prioritised, so the sum should run over the undirected people (equivalently, maximise the mean over the directed, or minimise the mean of 100 − ULPP over the directed; with n_dir fixed these are the same optimiser). Confidence about 80 %. The decisive evidence is the *Applied Sciences* text itself, which defines the index conditionally on the decision ("if they are not [directed first], FLPP rises"), and the ESWA template, in which the FIS output is zero exactly when the assignment is right.

## 2. ULPP history, source by source

### 2.1 WPOM 2021 (Rabiei & Arias-Aranda, 12(1), doi:10.4995/wpom.14699) — read in full (gold OA PDF)

Personnel assignment in a chocolate company; no disaster context, no fairness index. The fitness of a solution is a weighted sum of candidate competencies, maximised, with a linguistic weight per position:

> "As positions are of different importance, the fitness value of a solution is not simply the summation of each individual competency. [...] the competencies of candidates who could occupy several positions are awarded higher fitness values, which results in their placement in the positions of higher priority." (p. 10, Step 6)

Relevance: "importance" here is a *weight on the position*, and the optimiser places strong candidates in important positions. No sign issue; nothing resembling ULPP.

### 2.2 ICTIS 2021 (Rabiei & Arias-Aranda, doi:10.1109/ICTIS54573.2021.9798556) — not readable

IEEE Xplore returned an empty page to the fetch tool, Crossref and Semantic Scholar hold no abstract, the Granada research portal lists only bibliographic data (ICTIS 2021, pp. 1226–1243), and there is no local copy. The title ("vehicle routing and relief supply distribution in post-disaster phase: combining fuzzy inference systems with NSGA-II and NRGA") is about supply distribution, not people prioritisation. I could not confirm or exclude a fairness objective in it.

### 2.3 ESWA 2023 (Rabiei, Arias-Aranda & Stantchev, 226:120142) — read in full (local copy `/Users/vstantch/projects/presidio-hardened-vol-asssign/1-s2.0-S0957417423006449-main.pdf`)

No ULPP, FLPP, "fairness" or "unfairness" anywhere. The first objective is:

> "Z1 — Mean importance of unmet needs by volunteers for all vacancies in all centers" (Table 1, p. 4)
> "x′_ji — Importance of unmet triage needs by volunteer j in center C_i, x′_ji ∈ [0 100]" (Table 1)
> "fis1 : (P_i, E_i, T_j) → x′_ji" (Eq. 1, p. 5); "Min Z1 = Σ_i Σ_k x_ik / Σ_i s_i" (Eq. 8, p. 5)
> "This index is evaluated based on the number of covered patients in the center, the emergency level of the center, and the level of triage ability (or nursing ability) of the volunteer assigned to the vacancy." (§3.3.1, p. 5)

So Z1 is: per *pair* (volunteer, vacancy), minimised, averaged over all vacancies (all of which are filled, constraints 5 and 12), i.e. over the *assigned* set. The rule base (Appendix A, p. 17) shows the output is a **misfit** score, not a priority score:

| Rule | P | E | T | x′ |
|---|---|---|---|---|
| 1 | Low | Low | Low | Very Low |
| 3 | Low | Low | High | Very High |
| 25 | High | High | Low | Very High |
| 27 | High | High | High | Very Low |

A quiet ED with an over-qualified volunteer (rule 3) and a critical ED with an under-qualified one (rule 25) both score Very High; a critical ED with a highly skilled volunteer (rule 27) scores Very Low. "Importance of unmet needs *by volunteer j*" is the importance of what volunteer j leaves unmet. Minimising its mean over the assigned pairs is therefore correct: the score is already conditioned on the decision and is low when the decision is right. **The ED precedent does not have the sign problem**, and the in-repo re-implementation agrees (`/Users/vstantch/projects/pva-experiments/src/presidio_vol_assign/fis.py:83-84`: "Importance is high when the ED is critical [...] AND the assigned volunteer's skill is low (poor match → unmet need)"; `domains/ed_staffing.py:116`: "Z1 = mean importance of unmet needs (minimise)").

### 2.4 ATRES 2026 manuscript (ATRES-D-26-00014 and ATRES-S-26-00021) — read in full

The two PDFs are the same text except for the manuscript number (verified by `diff`). ULPP first appears here. Definitions, with line numbers in the `pdftotext -layout` dump:

- Index, introduced under a *fairness* name: "we introduce a new qualitative index: Fairness Level in People Prioritization (FLPP). This index evaluates the priority of affected people for directing to a limited number of relief centers." (§3.2.1, lines 716–719). Note: the index *is* a priority.
- Objective, introduced under the *unfairness* name: "Regarding the Fairness Level in People Prioritization index (FLPP), we define our first objective function as minimizing the unfairness level in people prioritization." (§3.4.1, lines 1123–1125)
- Notation: "ULPP_j — Unfairness level in prioritizing person j for directing to relief centers, ULPP_j ∈ [0 100]" (Table 1, line 927); "Mn_ULPP — The mean of unfairness level in prioritizing people to relief centers" (line 1075).
- FIS and objective: "FIS1 : (VS_j, IDL_j, RTR_j) → ULPP_j (1)"; "Mn_ULPP = Σ_i Σ_j d_{j,i} × ULPP_j / n_dir (3)"; "Eq. (3) evaluates the mean level of unfairness in prioritizing people to relief centers among all selected individuals." (lines 1140–1157). The model section repeats it with the centre index already summed out: "Min Mn_ULPP = Σ_j d_j × ULPP_j / n_dir (11)" (line 1263).
- Rule base (Table 4, lines 1730–1790): Low/Minor/Long → Very Low; High/Severe/Short → Very High; identical to the *Applied Sciences* appendix and to `FIS1_RULES` in the code.
- Linguistic labels for the output (Table 3, lines 1640–1652): Very low "minimal unfairness, equitable prioritization"; Moderate "moderate inequity requiring adjustment"; Very high "severe misallocation of resources".
- The stated aim, three times: "directs the most vulnerable affected people to the most suitable relief centers" (line 588), "we should be able to prioritize the most vulnerable people" (line 689), "directing the most vulnerable people to the most appropriate relief centers" (conclusion, line 2654).

So the direction is: **per person, min, over directed** from the first appearance, and the contradiction is already present: the index is described as a priority (high for the most vulnerable), the objective minimises it over the people served, and the aim is to serve the most vulnerable. The ATRES reviewers did not catch it; Reviewer 2 only complained that "neither the equation nor the caption gives any hint" that Mn_ULPP is the objective (`docs/atres-rejection-review.md:177–179`).

### 2.5 *Applied Sciences* 2026 (`paper.tex`) — read

Unchanged equation: `\min \overline{ULPP} = Σ_j Σ_i d_{j,i} ULPP_j / n_dir` (Eq. `obj-ulpp`, lines 653–654), "the model minimises four mean-objective functions over the directed set" (line 651). The prose adds three sentences that define ULPP *conditionally on the decision*, which the equation cannot express:

- "Higher VS and IDL with lower RTR mean a person should be directed first; **if they are not, FLPP rises.**" (lines 505–506)
- "**ULPP is zero exactly when** the coordinator's three diagnostics, the rule base's priority logic, **and the directing decision are mutually consistent**" (lines 723–725)
- Table 6: Resourcefulness, ULPP, FIS1, "Did we direct the most vulnerable first?" (line 698)

and the contradiction sentence: "Directing the most vulnerable people (low ULPP) often means reaching those who live where infrastructure is most damaged" (line 642), plus Vignette 3: "the ULPP objective rewards it (the people are vulnerable)" (line 1441), which under the published equation is false (it penalises directing them).

There is also one equity-flavoured sentence: "FLPP encodes a need-weighted Rawlsian reading of fairness --- prioritisation by greatest individual need rather than equal shares or proportional allocation [...] readers preferring the SCM literature's vocabulary may read it as a Rawlsian maximin term" (lines 494–499). This is a *need-priority* reading, not a dispersion reading, and it too points to (a): greatest need first.

### 2.6 The code (`pva-experiments`)

- `allocation/solvers.py:138–143`: "# ULPP — depends only on person attributes"; `ulpp[i] = evaluate_fis1_ulpp(vs, idl, rtr)`, array of shape `(m,)`.
- `allocation/solvers.py:215`: `mn_ulpp = float(cache.ulpp[persons].mean())`, where `persons` is the directed set; `centers` is not used for ULPP.
- `allocation/fis.py:151–160`: `FIS1_RULES` identical to the paper tables.

### 2.7 A sibling design in the same repository (not in any paper)

`pva-experiments/src/presidio_vol_assign/fis_humanitarian.py` (the v0.1/v0.2 "humanitarian" engine, also documented in `presidio-hardened-vol-asssign/docs/fis-worked-example.md`) defines a *decision-dependent* unfairness, the correct port of the ESWA template: "FIS-A — Fairness in People Prioritization [...] output: unfairness (0-1) — high when a highly vulnerable person is sent to a low-service and/or distant centre" (lines 7–10); "Rationale: unfairness bites only when a *vulnerable* person is placed poorly (low service and/or far). A low-priority person is treated fairly almost regardless of placement." (lines 64–67). Z1 is its mean over all people, minimised (`fis-worked-example.md:159–161`). This is sound for a model that directs everyone, because the score is conditioned on the placement. It shows that, when the template was ported with a decision-dependent input, the mean-over-selected form worked; the published model has no such input.

## 3. The grouping hypothesis, checked

Question: can the published ULPP objective reward gathering people of similar ULPP at one centre or in one directed set?

1. **ULPP_j depends only on person j.** Eq. (1) in ATRES and `paper.tex` line 507: FIS1 takes (VS_j, IDL_j, RTR_j). No centre attribute, no other person, no load, no count. In code it is precomputed once per person (`solvers.py:138–143`), before any allocation exists.
2. **The objective is a plain mean over the directed set.** Σ_j Σ_i d_{j,i} ULPP_j / n_dir. Since Σ_i d_{j,i} ∈ {0,1} (constraint 14), the inner sum is just "is j directed", and ATRES Eq. (11) already writes it as Σ_j d_j ULPP_j. The centre index disappears. Re-assigning any directed person to any other centre leaves the value unchanged, and so does any partition of the directed set among centres.
3. **No dispersion term exists.** There is no variance, range, Gini, max−min, pairwise difference or per-centre sub-mean anywhere in the three documents or the code. A mean of per-person scores is invariant to how people are grouped; only the *set* matters, and for a fixed n_dir the mean is minimised by taking the n_dir lowest scores, full stop.
4. **Where an equity reading does exist in words.** ATRES Table 3 labels the output levels "equitable prioritization", "slight prioritization bias", "moderate inequity requiring adjustment", "severe misallocation of resources"; the *Applied Sciences* text calls FLPP a "Rawlsian maximin term". These are labels on a scalar whose inputs are one person's VS, IDL and RTR; the FIS cannot see any other person, so it cannot measure inequality between people. The label "severe misallocation of resources" for the Very High level is, if anything, further evidence for (a): it reads as *the harm done if this person is misallocated*, i.e. the stakes of leaving j out.

Conclusion: a grouping or homogeneity reading does not exist in the equations or the code, and no document describes ULPP as a dispersion. It cannot rescue the documented intent.

## 4. Most likely intended reading

**(a)**, confidence about 80 %: ULPP_j is the unfairness of *not* prioritising person j; the objective should penalise the ULPP of the people left out.

Decisive evidence, in order of weight:

1. The *Applied Sciences* text defines the quantity conditionally on the decision: "if they are not [directed first], FLPP rises" (line 506) and "ULPP is zero exactly when [...] the directing decision [is] consistent" (lines 723–725). A per-person FIS cannot do this; the only place the decision can enter is the index set of the sum. The intended object is "unfairness incurred by the people not directed".
2. The ESWA template. In Z1 the FIS output is a misfit that is Very Low when the right volunteer fills the right vacancy (rule 27); averaging it over assigned pairs and minimising is right. The relief model reused "FIS output averaged over the selected, minimised" word for word (compare ESWA §3.4 with ATRES §3.5), but FIS1's output is now a pure priority with no decision input, so "selected" is the wrong index set. The error is a template carried across a change in what the FIS measures, not a design choice to serve the least vulnerable.
3. The index was named and described as a *priority* ("FLPP [...] evaluates the priority of affected people", ATRES line 718) before being relabelled "unfairness" for the objective, and the rule base is a priority rule base (high need → Very High). Nobody wrote a rule base that gives Very High to the most vulnerable in order to *avoid* them.
4. Every statement of purpose in all three documents says "direct the most vulnerable".

Against (b) (equation intended, prose wrong): nothing supports it except the equation itself; it would make the model direct the least vulnerable, least damaged, best-provisioned people, which contradicts the title, abstract, index description, conclusion and Table 6 of the paper. About 10 %.

(c) grouping or equity: 0 % on the equations (Section 3).

(d) a decision-dependent unfairness as in `fis_humanitarian.py` (high when a vulnerable person is placed at a poor or distant centre): a defensible *redesign*, and the faithful port of the ESWA idea, but no published text describes ULPP as depending on the centre, and the published FIS has no centre input. About 10 % as "what the authors would have wanted", 0 % as "what the published equation means". Not a reading of the published model.

Peyman's confirmation is still needed, but the question for him can be narrowed to: "ULPP_j is the unfairness of leaving j out, yes?"

## 5. Consequences

### 5.1 The *Applied Sciences* correction

The intended objective, three equivalent forms (n_dir fixed):

- min (1/(m − n_dir)) Σ_j (1 − Σ_i d_{j,i}) ULPP_j (mean ULPP of the undirected), or
- max (1/n_dir) Σ_j Σ_i d_{j,i} ULPP_j, or
- min (1/n_dir) Σ_j Σ_i d_{j,i} (100 − ULPP_j), which is the "documented reading" variant already run in the *Systems* audit (`results-log.md:74`).

Points the correction has to make, in plain sentences:

1. "Equation (obj-ulpp) minimises the mean ULPP over the directed people. FIS1 assigns high ULPP to people of high vulnerability, severe infrastructure damage and short resource time, so the equation as printed directs the people of lowest priority. The intended objective is the mean ULPP of the people *not* directed, minimised; equivalently, the mean ULPP of the directed people, maximised."
2. "ULPP_j should be read as the unfairness incurred if person j is not prioritised. The sentences 'Directing the most vulnerable people (low ULPP)' (Section 3.x), 'the ULPP objective rewards it' (Vignette 3) and the Table 6 reading 'Did we direct the most vulnerable first?' describe the intended objective, not the printed one."
3. "All reported results (Pareto fronts, hypervolumes, sensitivity, Table …) were computed with the printed objective. On the reference instances its optimum directs 12 % (small) and 7 % (large) of the n_dir people a crisp priority rule ranks highest; the corrected objective directs an almost disjoint set (8 % and 2 % overlap)." Then state which conclusions survive (the H1/H2 algorithm comparisons and the 3-vs-4-objective structural claims do not depend on the sign; the practitioner reading of the fronts along the ULPP axis does not survive) and where the re-run lives.

Decision for Vladimir: whether this fits a Correction notice (equation + three sentences + the statement that the results stand for the printed objective) or needs a re-run of the main tables under the corrected objective first. The structural facts (separability, per-person score) do not change, so the re-run is mechanical; the numbers along the ULPP axis will.

### 5.2 How the *Systems* revision should word the finding

Current wording (`paperB-mdpi-source-r1/author/paper.tex:815–834`, Conclusions 1527–1530) says the objective "rewards directing the people of lowest priority" and reports the mismatch as "between the model's equations and its documentation". That is accurate about the model as published, and it should stay the primary statement. Given Section 4, add the diagnosis: the objective runs over the wrong index set. Suggested sentences:

- Section 5.1, replacing "The rule base does the opposite: …" through "… and the objective minimises the mean ULPP over the people who are directed.": "FIS1 assigns the highest ULPP to the people who are most vulnerable, live where damage is severe and have the least resource time, so ULPP_j is a priority score, and the prose of the published article treats it as the unfairness incurred when such a person is *not* directed first. The objective, however, averages ULPP over the people who *are* directed and minimises it. The sum runs over the wrong set: as printed, the optimum directs the people of lowest priority it can."
- One sentence after the 100 − ULPP results: "Minimising the mean of 100 − ULPP over the directed people is the same optimiser as minimising the mean ULPP over the undirected, so the 'documented reading' column is the corrected objective."
- Conclusions, "Established analytically": "… and its ULPP objective, as printed, averages a priority score over the people directed and minimises it, which rewards directing the people of lowest priority; the intended objective runs over the people not directed."
- Do not write that the authors intended to serve the least vulnerable, and do not call it an "equity" or "dispersion" measure anywhere; the text should not offer the grouping reading even to reject it, since no document proposes it.

If Peyman answers (b) instead, the Section 5.1 paragraph as it stands is already right and only the diagnosis sentence ("the sum runs over the wrong set") would be dropped.

## Sources read

- `/Users/vstantch/Downloads/ATRES-D-26-00014.pdf` and `/Users/vstantch/projects/presidio-hardened-vol-asssign/pubs/ATRES-S-26-00021 (1).pdf` (identical text; line numbers above are from `pdftotext -layout`)
- `/Users/vstantch/projects/presidio-hardened-vol-asssign/1-s2.0-S0957417423006449-main.pdf` (ESWA 2023, full text)
- WPOM 2021 full text, gold OA at https://polipapers.upv.es/index.php/WPOM/article/download/14699/13464
- ICTIS 2021: bibliographic record only (https://produccioncientifica.ugr.es/documentos/62e5c3dfe5f0e01a6a1d1533); full text and abstract not accessible to me
- `/Users/vstantch/projects/presidio-hardened-vol-asssign/pubs/v0.2.0-mdpi/revision/paperA-mdpi-source-revised/author/paper.tex`
- `/Users/vstantch/projects/presidio-hardened-vol-asssign/docs/atres-rejection-review.md`, `docs/fis-worked-example.md`, `PRESIDIO-REQ.md`
- `/Users/vstantch/projects/pva-experiments/src/presidio_vol_assign/allocation/fis.py`, `allocation/solvers.py`, `fis.py`, `fis_humanitarian.py`, `domains/ed_staffing.py`
- `/Users/vstantch/projects/pva-paperB/pubs/systems-turbulence/revision/paperB-mdpi-source-r1/author/paper.tex`, `results-log.md`, `coauthor-revision-email-draft.md`
