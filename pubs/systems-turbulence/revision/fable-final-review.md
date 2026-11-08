# Final pre-submission review — systems-4669426 (major revision)

Reviewer: Fable (senior-advisor pass, adversarial), 2026-11-03.
Materials: three review reports, submitted and revised `paper.tex`, built PDF (34 pp., diff 37 pp.), three response letters, `results-log.md`, result CSVs under `pva-experiments/experiments/results/`.
Paths below: `paper.tex` = `revision/paperB-mdpi-source-r1/author/paper.tex`; `tab_envelope.tex` = `.../author/tables/tab_envelope.tex`; letters = `revision/response-reviewer-{1,2,3}.md`.

Build is clean: no undefined references or citations in `package/build/.../paper.log`, one overfull box, all `\ref` labels resolve, no `identifiab*`, no `deployed` (only "deployment envelope", which the R2 letter explains). Abstract is 205 words. Figure 3 carries the dashed exact-decider line its caption promises; Figure 6 marks "2 unsolved" at 3000×50.

---

## A. Must fix before upload

**M1. Placeholder DOI in the Data Availability Statement.** `paper.tex:1629–1630`: `%TODO-DOI` and `https://doi.org/10.5281/zenodo.TODO`. Reserve or mint the v0.5.0 Zenodo DOI and replace both; delete the comment line. (An MDPI production check will catch this, and a reviewer who clicks it will not.)

**M2. The paper promises per-method solver time and total latency that it does not contain; all three letters claim it.** `paper.tex:780–781` ("Each method is reported with its solver time and its total decision latency, preparation included") and `:1254–1255` ("Table 13 reports total decision latency next to solver time"). Table 13 (`tab_envelope.tex`) has a single "Latency" column, defined in its caption as the hard-capacity local search only, and no solver-time column at all; Table A1 has no times. Letters: R1 §1.2 "Every row reports solver time and total decision latency"; R2 §2.4 "every row reports solver time and total decision latency"; R3 "Optimisation time and preparation are now reported separately for every method … Total decision latency is reported alongside." R3 asked for exactly this. The data exist (`solver_s`, `fis_s`, `table_s`, `total_s` in every `envelope/envelope_t*_s*.csv`). Fix: have `experiments/make_envelope_tables.py` emit a third table for Appendix B (rows = size × limit, columns = method; cells = mean solver s / mean total s, with FIS and table preparation as two extra columns), reference it from §5.4 and §6.2, and make `:780` say "Appendix B reports …". Alternative, if no time: reword `:780–781` and `:1255` to what Table 13 actually shows and correct the three letters.

**M3. A sentence that R2.7 would quote back.** `paper.tex:1274–1275`: "Given the same wall-clock time, it [NSGA-II] is 6–43 units behind the soft MIP wherever the MIP found an incumbent." False at 3000×50 / 60 s: the one incumbent (instance 2) is 150.73 and NSGA-II is 144.62 there (`envelope_t60_s2_p3000.csv`), which Figure 6 shows plainly. Fix: "… 6–43 units behind the soft MIP at every size where the MIP solved all three instances; at 3000×50 under one minute the MIP's only incumbent (150.7) is worse than every other method."

**M4. The envelope conclusion overstates the local search.** Abstract `:61–62` ("beyond them, a capacity-aware local search outperforms the time-limited exact method"), Introduction `:142–144`, Conclusions `:1523–1524` ("outperforms both the time-limited MIP and the audited evolutionary system"), and the RQ4 answer `:1295`. Table 13 says otherwise under the five-minute limit: the MIP beats the local search at 1000×25 (96.6 vs 97.1) and 1500×40 (95.7 vs 96.5); the local search wins only at the one-minute limit from 1500×40 and at 3000×50 under both limits. The paper's own §5.4 text (`:1269–1270`) states this correctly. Fix the four summary statements to something like: "beyond them, a capacity-aware local search matches the exact method within a minute at a fraction of its time, beats it at the largest size, and beats the evolutionary system at every size".

**M5. "Most of this is the model's own multiplicity" is not established and R2 asked for exactly this decomposition.** Abstract `:57–58`; RQ2 answer `:1059–1061` ("Most of that instability is the model's multiplicity exposed by the solver's distance from the optimum"); Discussion `:1383–1385`. The evidence does not support a share: (i) the paper itself says at `:991–992` that solver non-convergence and multiplicity "are not alternatives here"; (ii) the budget curve shows the floor halving (0.83→0.50) as the solver converges, and the model's multiplicity bound within 1% of the optimum is 0.26/0.33, so at least 0.3–0.6 of the 0.83 floor disappears with convergence and is, on R2's own framing, solver-attributable. A reviewer will ask "most by what measure?". Fix: replace the quantifier with the mechanism. Abstract: "The solver stops where the model's own multiplicity is near-total: allocations within 1% of the optimum already differ for a quarter to a third of the people, and an exact decider with a stated tie-breaking rule commits to one allocation." RQ2 answer and §6.1: "At the solver's quality the model admits almost any allocation, so the floor cannot be split into a solver share and a model share there; near the optimum the residual multiplicity (a quarter to a third of the people) is the model's own, and the rest is the absence of a selection policy."

**M6. Seeded NSGA-II has no overload entry although the text and the letter say otherwise.** `tab_envelope.tex` footnote: "Overload of MIP / LS / NSGA-II / fallback". `paper.tex:1279`: the seeded search "inherits the fallback's overload". R2 letter §2.7: "Overload columns are given for every method, including NSGA-II and the fallback." CSV (60 s, 3000×50): seeded 712–763 vs fallback 790–812, NSGA-II 289–357. Fix: add the seeded value as a fifth number in the overload column (one-line change in `make_envelope_tables.py`), or cite the numbers at `:1279` and change the letter to "for every soft-capacity method shown".

**M7. Co-author sign-off on the ULPP finding is outstanding.** `coauthor-revision-email-draft.md` is a draft; `results-log.md:66` says "ask Peyman to confirm". The paper states as an *analytical* finding (Conclusions `:1511–1514`, abstract `:55–56`, §5.1 `:812–831`, §6.5 `:1478–1482`) that the co-authors' published objective is reversed and that "a correction to that article is in preparation". Do not upload before written approval from Rabiei and Arias-Aranda. If Peyman answers (a) in the draft (ULPP is the unfairness of *not* prioritising, i.e. the sum should run over the non-directed), the wording must change from "rewards directing the least vulnerable" to "as written, the objective runs over the directed people, which reverses the documented intent". Independently of his answer, tighten "least vulnerable" (abstract `:56`, Conclusions `:1513`) to what §5.1 actually shows, "lowest priority (least vulnerable, least damaged, most resource time)": ULPP is a function of VS, IDL and RTR, and ρ(ULPP, priority) is 0.61–0.64, not 1.

**M8. Number wrong against the CSVs.** `paper.tex:1270` and R1 letter §1.2: local search "needs at most 15 s of solver time". `ls_soft` solver time at 3000×50 is 16.21 s (60 s run, s0) and 16.45 s (300 s run, s0). Write "under 17 s" or "about 15 s".

**M9. Table 14 "Gap" mixes three statistics and the ED sentence contradicts itself.** `paper.tex:1313–1314`: "its best front point is 0.2% above the exact equal-weight optimum, and 11 of 20 NSGA-II seeds reach it" cannot both hold for a best-over-seeds point. `ed_audit/floor.csv`: 0.2% is `best_sum_mean` (mean over seeds of each seed's best point, 0.3599 vs 0.3592); the relief row's 23.9% is the minimum over seeds (92.39); the knapsack rows use the mean committed point. Fix the caption to say "Gap: mean over seeds of the best front point (ED) or of the committed point (knapsack); relief: best point over seeds", and reword `:1313`: "averaged over seeds, the best front point is 0.2% above the optimum, and 11 of 20 seeds reach it exactly". Same wording in R2 letter §2.1.

---

## B. Should fix

**S1. Text vs table rounding.** `paper.tex:1270` "(96.4 against 95.7)" vs Table 13 "96.5" for the 3000×50 / 300 s soft MIP (per-instance 95.03, 98.03, 96.30; mean 96.45). Use one value.

**S2. "The protocol runs on two further models" overstates.** R1 letter (summary bullet), R2 §2.1, R3 (summary bullet). The paper runs S1–S3 only (`paper.tex:1301–1302`). Say "steps S1–S3 of the protocol" in all three letters; §5.5 is already honest.

**S3. R3 letter claims positioning against "sensitivity analysis".** Table 1 has no sensitivity-analysis column and §2 cites no SA literature. Either add one sentence and a citation (e.g. Saltelli et al.) to §2.7 or delete "sensitivity analysis" from the letter.

**S4. R2 letter §2.4 "All methods used the compiled fuzzy evaluator"** has no counterpart in §4.5/§4.6 (`paper.tex:770–795`). Add "all methods use the compiled evaluator" to the envelope paragraph.

**S5. R3 letter numbers differ from the paper.** Letter: "hypervolume changes by at most 1%, share by at most 0.008"; paper `:857–858`: 1.1% and 0.01; `weight_grid.csv`: 0.989 (1.1%) and 0.805−0.796 = 0.009. Align the letter to the paper.

**S6. Misplaced clause.** `paper.tex:560–561`: "… in the order people, centres, routes, and the bias mode shifts every value of a field by the level times its range." The bias definition is spliced into the instance-generator sentence. Move it to the perturbation definitions at `:575–580`.

**S7. "Within 0.5 units where the MIP is proven optimal"** (`paper.tex:1422`, R1 letter §1.2): per instance the gap reaches 0.62 (600×20, s1) and 0.55 (1500×40, s0, 300 s); the means are ≤0.5. Say "within 0.5 units on average, 0.6 on any instance".

**S8. "Gaps of at most 0.02%"** (`paper.tex:1266`, R2 letter §2.7): the gaps are 0.023% and 0.019%. Say "about 0.02%".

**S9. "242 s for the MIP at 3000×50"** (`paper.tex:1272`, R2 letter §2.4) is the mean of one proven solve (115 s) and two runs that hit the 300 s limit (305, 307 s). State it as such: "115 s on the instance it proves, the limit on the other two".

**S10. Bias "about two thirds as much as noise … on resource time and travel time"** (`paper.tex:1080–1082`, R3 letter). Table 10 ratios at level 0.4: resource time 0.62–0.71; travel time 0.49 (small, −), 0.65 (small, +), 0.72 (large, −), 0.99 (large, +). Say "about two thirds on resource time, and between half and the full noise effect on travel time".

**S11. The 150-generation budget is still unjustified.** R2.4 asked; the letter says it "follows the published experiments", which used 200. The paper (`:834–835`) only notes the difference. Add the reason in one sentence (artefact default, or calibrated to a latency), or state that the 2400/9600-generation rows make the choice immaterial.

**S12. "≤0.8 objective units" at the lowest level** (`paper.tex:1070`) refers to the reference instances, but Table 10 on the same page shows 0.85 (travel-time noise, large, median over conditions). Say "under 1 unit".

**S13. The MOEA time-boxing defect is disclosed in the letters but not in the paper.** R2 letter §2.4 corrects the submitted version (307.8 / 263.2 / 182.7 s under the "300 s" limit); the paper only says runs are now stopped by a deadline. Add a half-sentence in §4.5 or Appendix B so the AE sees the same statement in both.

**S14. Table 11 floats before its first citation** (`paper.tex:1176` vs `:1208`). Move the table environment after the paragraph that cites it; MDPI wants citation before appearance.

**S15. Table 14: give CIs or min–max for the knapsack seed floor** (`knapsack/floor.csv` has `churn_min`, `churn_max`) for symmetry with the ED rows, or say in the caption why they are omitted.

---

## C. Optional

**O1.** Keywords (`paper.tex:66`) still open with "digital AI systems; economic turbulence" after title and abstract were toned down (R2.2). Defensible for the SI; consider moving "decision-support systems" first.

**O2.** `results-log.md:103–110` says the large extended-turbulence study is "pending"; `turbulence_extended/summary.csv` (16:09) has 36 conditions for both sizes. Update the ledger so the artefact and the log agree.

**O3.** Acknowledgments `:1632`: "(xAI, a subsidiary of SpaceX)" is unnecessary.

**O4.** "Stays within 1.5% of the MIP at every size" (`:1271`): per instance up to 2.1% (150×5, s2: 119.19 vs 116.74). "On average at every size" is exact.

**O5.** The MGA alternatives are maximally different by construction. One clause in §5.2 ("these are the most different allocations within ε, not typical ones") pre-empts a reviewer asking about typical churn; `multiplicity/summary.csv` has `mean_pairwise_churn` if a typical figure is wanted.

**O6.** Figure 4's legend is small; readable in print, borderline on screen.

**O7.** Abstract at 205 words is fine for MDPI; do not cut the last sentence (R1/R2 asked for it).

---

## D. Coverage check (request → letter → paper)

All points of all three reports are answered in the letters and, with the exceptions flagged above, present in the paper.

| Reviewer point | Paper location | Status |
|---|---|---|
| R1.1 analogy only | abstract, §1, §3.3, §6.4, §7 | done |
| R1.2 scalability boundaries | §6.2, Table 13, App. B | done, but M2 (times) |
| R1.3 auditing/regulation literature | §2.5–2.7, Table 1 | done |
| R1.4 actionable recommendations | §6.3 | done |
| R2.1 novelty; further case not theirs | §1 (four claims), §5.5, §6.6 | done; S2 (S1–S3 only), M9 |
| R2.2 title/abstract/keywords; "deployed"; scope | title, abstract, §1 last para | done; O1 |
| R2.3 self-contained model, symbols, parameters | §3.2, Tables 2–4 | done |
| R2.4 fair baselines, budget, timing, soften claim | §4.5, §5.4, Table 13 | done except M2, M6, S4, S11 |
| R2.5 multiplicity, "by construction", tie-break, Rashomon | §2.6, §4.3, §5.2, Table 9, Fig. 4 | done; M5 wording |
| R2.6 trend tests without level 0, joint, levels, seeds, CIs | §4.4, §5.3, Table 6, Table 10 | done |
| R2.7 300 s on matched instances, unsolved handling, overload, repair form, abstract | §5.4, Tables 11–13, App. B | done except M6; abstract M4 |
| R3.1 optimiser vs decision support | title, §1, §6.1, §7 | done |
| R3.2 sampled supported front, finer grid | §4.2, Table 8 | done |
| R3.3 stability vs legitimacy; tie-break demonstrated | §5.2, Table 9 | done; M5 |
| R3.4 crisp rule, churn formula, operators, generator, "above floor" | §3.2 Eqs. 4–8, Table 2 | done |
| R3.4 bias results | Table 10 | done; S10 |
| R3.5 generalisation; levels; "manageable" | §4.4, §5.3, Table 10 | done |
| R3.6 unsolved solves, matching instances, fallback as start, latency | §5.4, App. B, §6.2 | done except M2, M3 |
| R3.7 model loss vs outcome; overload not always reduced | §6.1 | done |
| R3 minor: larger labels, unsolved marked, conclusions in three kinds | Figs. 5–6, §7 | done |

---

## E. Numbers checked against the CSVs (mismatches marked)

Abstract / Intro / Conclusions: 97–100% dominated (Table 7: 2838/2838 … 2883/2973 = 97.0%) ✓; 83–94% churn (0.834, 0.939) ✓; within 1% → 26%/33% (eps = 1 is 1.34%/1.17% of optimum) ✓; "most of this is the model's multiplicity" — unsupported (M5); LS "outperforms time-limited exact" — only under 60 s (M4).

RQ1 / Table 6 (`audit_summary.csv`): all 16 rows match (dominated, HV share with CI, best, churn with CI), e.g. small 150: 2000/2000, 0.39 [0.39,0.40], 92.39, 0.83 [0.82,0.85]; large 9600: 1945/2000, 0.42 [0.41,0.43], 105.71, 0.90 [0.88,0.91] ✓. "82–100% on two further instances" (1630/2000 = 81.5%) ✓. Population controls change HV by ≤0.03 (0.728, 0.690 vs 0.718) ✓. NRGA 2400: 76.16, 65% ✓.
Table 8 (`weight_grid.csv`): 181→1443, 197→3359; 0.989/0.994; shares 0.365/0.805, 0.361/0.796, 0.160/0.351, 0.159/0.349 ✓; "at most 1.1%" and "0.01" ✓ (letter says 1% / 0.008: S5).
ULPP (`ulpp_direction.csv`): ρ 0.61/0.64; 12%/7%; 44 vs 60; 33 vs 25 h; 8%/2% overlap; 32 of 40 units (72.09−39.98 = 32.1; 114.51−74.58 = 39.9); large crisp gap 44 (129.17−85.18) ✓.

RQ2 (`audit_summary.csv`, `multiplicity/*`): 83.4 [82.0,84.7], 93.9 [93.3,94.5]; Jaccard 0.31/0.25; s.d. 2.05/1.32; 97.4→81.1; 0.83→0.50; large ≥0.90 (0.895 at 9600, borderline) ✓. eps = 1: 0.26/0.33; eps = 5: 0.62/0.73; hard within one unit 0.52/0.73; Jaccard at deficit 0.08/0.00; alternative optima 1%/6%/22%; tie-break 10 and 9 → 1 ✓. "415 of 420 proven, other five within 0.5%" (5 unproven, max gap 0.46%) ✓. Cross-env optimal values 110.945142 / 109.070123 match `z_star` ✓.

RQ3 (`turbulence_extended/summary.csv`, `trend_tests.csv`): ρ 0.37–0.97 and Holm p ≤ 4.8e-8 at levels 0.05–0.4 ✓; all 8 noise/missing/flip cells `share_significant` = 1.0 in both sizes ✓ (the log's "pending large" is stale: O2). Table 10 spot checks (travel-time noise small 0.63 / 6.9 [4.4,8.5] / 0.42; occupancy noise large 0.46 / 9.2 [6.7,14.6] / 0.61; joint all eight 2.40 / 28.8 [21.1,38.8] / 0.76 (0.79) and 3.02 / 31.5 [23.9,40.9] / 0.86 (0.78); transport 0.98/0.96; occupancy bias − large 0.10 / 2.5 [0.0,7.5] / 0.37) ✓. 22.8 and 35.5 units solver cost (97.4−74.58, 120.71−85.18) ✓. Bias "two thirds": approximate (S10). "≤0.8 at lowest level": 0.85 in Table 10 (S12).

RQ4 (`repair_forms.csv`): all ten overload ranges and ten penalty ranges in Table 12 ✓ (e.g. stretched large 45.4–46.1%, convex small 0.0–30.2); "at most 11.4 s" (11.37) ✓; 29 of 30 soft cells overloaded ✓; published optimum violates hard capacity by 14–50 (13.9–49.9) ✓; −4.3 to +6.4 ✓; "at most 3.7 s" ✓.
Envelope (`envelope/*.csv`): Table 13 and Table A1 quality values ✓; s/p counts ✓; latency column 1/3/11/28/65/172 and 29/60/178 (means 0.9/2.7/10.6/28.2/64.9/172.2; 29.1/60.0/178.4) ✓; FIS 10 s, tables 95/160 s ✓; LS hard ≤ ~3 s (3.84) ✓; 1.2 GB ✓; seeded helps by up to 44 (44.1 at 3000×50) and ends within ~1 of fallback from 600×20 ✓; "6–43 behind wherever MIP found an incumbent" ✗ (M3); "at most 15 s" ✗ 16.45 (M8); "242 s" is mean incl. two timeouts (S9); "96.4" vs table 96.5 (S1); gaps "≤0.02%" are 0.023/0.019 (S8).
§5.5 (`ed_audit/floor.csv`, `knapsack/*.csv`): ED 0.2%/9.3%, 11/20, 20%/10%/34%/76%, floors 0.49 [0.39,0.59], 0.42 [0.33,0.52], 0.90 [0.87,0.92], 0.91 [0.89,0.94] ✓ (definition inconsistency: M9); knapsack 0.9/0.2%, 1.2/0.3%, 29/4%, 60/9%, 0.11/0.06, 0.11/0.07; ~0.3 s per weighted sum (31–33 s / 101) ✓.

---

## F. Cross-references in the letters vs the built PDF

All resolve: §2.6 (multiplicity), §2.7/Table 1 (positioning), §3.2/Tables 2–4 (model), §3.3 (analogy), §4.2–4.4, Eq. (10) (MGA), §5.1–5.5, §6.1–6.6, Tables 6, 8–14, Figures 4–6, Appendices A–D (submitted Appendix C = cross-env is now D; appendix tables are A1, A2). "Table 6 (formerly Table 2)" and "Table 5 (now Table 11)" are correct against the submitted source.

---

## G. Tone and scope

Title, abstract and conclusions stay within what is tested, with two exceptions already listed (M4 envelope, M5 "most"). "Deployed" is gone in R2's sense; "deployment envelope" is retained and justified. The ULPP paragraph is fair to the co-authors in substance (it quotes the documentation, shows the two rules, reports the corrected-reading overlap, and says the corrected model remains separable), but it needs their written assent (M7) and the "least vulnerable" shorthand should become "lowest priority". The two-further-models section is honest about limits (own ED model; benchmark not a system; no third-party system found) and does not condemn the knapsack search. RQ3 generalisation claims are bounded by "on every instance and condition we tested" and the Limitations paragraph; adequate.
