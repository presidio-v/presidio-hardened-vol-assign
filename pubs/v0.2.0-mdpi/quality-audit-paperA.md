# Pre-Submission Quality Audit — Paper A

**Manuscript:** `pubs/v0.2.0-mdpi/author/paper.tex`
**Target:** Applied Sciences (MDPI), SI "Innovations in Supply Chain Resilience"
**Audit date:** 2026-06-30 / 2026-07-01 (read-only; nothing edited)
**Build:** `latexmk -pdf` exits 0 — **clean build, 29 pages**, no undefined refs/citations, no LaTeX warnings, one trivial 2.8 pt overfull hbox (lines 178–185, contributions list). bbl built from `lit.bib`.

## Verdict summary

The manuscript is in strong, near-submittable shape. Numbers are internally consistent and trace to the released manifest; all four hypotheses are explicitly resolved; the back matter is structurally complete; references resolve cleanly. The **one blocking issue is MDPI template conversion** (generic `article` + `authblk`, not `applsci.cls`). The remaining findings are tightening, not structural.

---

## CRITICAL

### CR-1. Document class is generic `article`, not MDPI `applsci.cls`
**Location:** lines 9–17 (header comment still says "Tier 1 skeleton… To swap in MDPI's official applsci-mdpi.cls before submission"), line 17 `\documentclass[11pt,a4paper]{article}`, lines 36/46–51 `authblk` + `\author[n]{}`/`\affil[n]{}`.
**Why critical:** MDPI requires their `Definitions/mdpi.cls` (`\documentclass[applsci,…]{mdpi}`) with `\Title`, `\Author`, `\affiliation`, `\corres`, `\abstract{}`, `\keyword{}` macros and their BibTeX style (`mdpi.bst` / `Definitions/mdpi.bib`). The current `\bibliographystyle{plain}` (line 2073) and numeric `\cite` will not match MDPI house style. Submitting in generic `article` is the single thing most likely to trigger an immediate editorial bounce-back.
**Fix:** Convert to the MDPI template before submission: download `mdpi.cls`, move title/authors/affiliations/correspondence into MDPI macros, replace `authblk`, switch `\bibliographystyle` to `mdpi`, and move the back-matter sections into MDPI's prescribed `\authorcontributions`/`\funding`/`\dataavailability`/`\conflictsofinterest` macros. Budget a full pass — table/figure floats and the `\paragraph`-heavy structure usually need light rework under MDPI's two-column-ready class.

### CR-2. Stale "skeleton / placeholder" scaffolding comments still in source
**Location:** line 9 (`% Tier 1 skeleton`), lines 11–15 (swap-in instructions), line 94, line 1232, line 1410 (`% TIER 1 PLACEHOLDER — target ~1,200 words`), line 1793 (`% DRAFT CRediT — co-authors to confirm/adjust roles before submission`).
**Why critical (housekeeping):** These are LaTeX comments so they do not render, but they are exactly the kind of artifact that leaks if the `.tex` is shared with co-authors/editor or if any line gets accidentally uncommented. The "DRAFT CRediT — co-authors to confirm" note signals the contribution roles are not yet author-confirmed.
**Fix:** Delete all `% TIER 1 PLACEHOLDER`, `% Tier 1 skeleton`, and template-swap comment blocks. Confirm CRediT roles with co-authors and remove the DRAFT note (line 1793).

---

## MODERATE

### M-1. Unqualified "the first" novelty claim survives the C2 softening
**Location:** line 165 (Contribution 1): *"The first explicit 4R↔objective mapping in FIS-MOEA disaster allocation"*; reinforced at line 1606 (*"This is the first published evidence we are aware of…"*).
**Context:** `adversarial-review.md` C2 flagged exactly this class of negative/novelty claim; the resolution softened the §1/§2.3/§2.5 *negative* claims ("no published" → "to the best of our knowledge, in the surveyed literature" — verified present at lines 149, 312, 351, 410–411). But the **positive "first" claim in the contribution list (line 165) was not softened** and is the most reviewer-exposed sentence in the paper: a single counter-example refutes it.
**Fix:** Qualify to match the rest of the paper, e.g. "To the best of our knowledge, the first explicit 4R↔objective mapping…" or "the first such mapping in the surveyed FIS-MOEA disaster-allocation literature." Line 1606 is already hedged ("we are aware of") but consider aligning wording.

### M-2. Abstract presents the more favorable HV reading ("comparable Pareto-front quality")
**Location:** abstract line 77: *"NSGA-III is 32% faster than NSGA-II at every size with comparable Pareto-front quality."* Echoed at line 1767 ("at unchanged Pareto-front quality").
**Why flag:** The body's primary H2a result (lines 1059–1068, Table 4) is that **NSGA-II significantly beats NSGA-III on HV at every size** under the fixed reference point (p = 0.027 / 0.001 / <0.0001). "Comparable" only becomes defensible after the instance-aware-reference analysis (lines 1102–1121) shows the ranking is reference-sensitive and "HV-equivalent in any practitioner-relevant sense." A skeptical reviewer reading the abstract then hitting "H2a refuted: NSGA-II beats NSGA-III on hypervolume" (line 1059) will perceive the abstract as over-smoothing.
**Fix:** Either keep "comparable" but add a clause acknowledging reference-sensitivity, or use the body's own honest phrasing ("HV-equivalent under a tight reference; NSGA-II ahead under a loose one"). At minimum the abstract should not read as if HV quality were settled in NSGA-III's favor.

### M-3. "32%" speedup — rounding provenance worth a glance
**Location:** abstract line 76, lines 1086–1087 (37/32/27%, mean 32%), line 1619, line 1767.
**Finding:** Manifest recomputation gives per-size speedups of **36% / 31% / 27% (mean 31%)** from cell means (NSGA-II 1.44/1.69/1.99 s vs NSGA-III 0.92/1.16/1.46 s). The paper reports 37/32/27 (mean 32%) using its rounded table values (1.45/1.69/1.99 vs 0.91/1.15/1.45). The discrepancy is pure rounding and not an error, but the small-instance figure rounds 36%→37% and the mean rounds 31%→32%. Defensible; flag only so the authors know the headline "32%" sits on a 31–32% boundary.
**Fix:** Optional. If a reviewer recomputes from the table they will get 32%; from the manifest, 31%. Consider stating "~32%" or "31–32% mean."

### M-4. AI-use declaration is an absolute negative — confirm it is literally true
**Location:** line 1811–1813 (Acknowledgments): *"The authors declare that no generative AI tools were used in the preparation of this manuscript."*
**Why flag:** MDPI requires an honest AI-use disclosure. This is a strong absolute claim. The audit cannot verify the drafting history, but the repo workflow (this very `pub/allocation-extension` branch is developed with AI tooling per project memory) makes a blanket "no generative AI tools were used" a statement the authors must be certain is accurate as written. If any AI assistance touched drafting/editing, MDPI's policy is to disclose its use, not to deny it.
**Fix:** Confirm with co-authors that the statement is literally true for *manuscript preparation*. If AI tooling assisted at any drafting/editing step, replace with MDPI's standard disclosure form ("During the preparation of this manuscript the authors used [tool] for [purpose]; the authors reviewed and edited and take full responsibility…").

---

## MINOR

### m-1. Seven orphan bib entries (uncited but present in `lit.bib`)
**Location:** `lit.bib` — `emmerich2014`, `zheng2017`, `zitzler1998` are never cited anywhere. (Earlier comm-diff false-positives `altay2006`, `anayaarenas2014`, `caunhye2012`, `galindo2013` ARE cited at line 280 — verified, ignore.)
**Impact:** With `\bibliographystyle{plain}` + `\bibliography`, uncited entries do **not** appear in the reference list, so no visible effect now. Reference count: **38 cited** of 45 in bib — comfortably in the "reasonable for an SI feature paper" range (adversarial review noted ~46; current 38 is fine, above the 21 that C2 criticized).
**Fix:** Optional cleanup — delete the 3 truly-orphan entries (`emmerich2014`, `zheng2017`, `zitzler1998`) or cite them if intended. No correctness risk.

### m-2. Zenodo record metadata is stale relative to the paper
**Location:** External — DOI 10.5281/zenodo.21083547 (and concept 21083544) both resolve correctly to **"presidio-hardened-vol-assign" v0.2.0** (verified live). However the Zenodo record *description* mentions only "NSGA-II and NRGA" (omits NSGA-III) and frames the tool as "emergency-department volunteer staffing." The paper is about NSGA-III and people-to-relief-centre allocation.
**Impact:** Does not affect the manuscript text; the DOIs, repo URL (`github.com/presidio-v/presidio-hardened-vol-assign`), and `v0.2.0` tag are all real and correct (tag exists in repo; manifest `experiments/results/h1_h2_h4/manifest.csv` exists and matches Table 4).
**Fix:** Update the Zenodo record description to mention NSGA-III and the allocation framing before the paper is indexed, so a reviewer following the DOI sees a consistent artifact.

### m-3. Appendix-A rule-base tables not individually `\ref`'d in body
**Location:** Tables `tab:rules-fis1/fis2/fis2-baseline/fis3` (lines 1834–1994) are referenced only collectively via "Appendix~\ref{app:rules} lists every rule" (line 843).
**Impact:** Acceptable — the appendix reference covers them. No undefined-reference warning. Noted for completeness only.

### m-4. "32% / comparable" appears in Conclusion too — keep consistent with abstract fix
**Location:** line 1767 (Conclusions) repeats "32% mean CPU advantage at unchanged Pareto-front quality." If M-2 is addressed in the abstract, mirror the change here so abstract/conclusion stay aligned.

### m-5. `\bibliographystyle{plain}` will be replaced by template conversion
**Location:** line 2073. Covered by CR-1; noted so it is not missed — MDPI uses `mdpi.bst`.

---

## Items CHECKED and PASSING (no action)

- **Internal numeric consistency:** "540 runs" (lines 73, 854, 978, 980, 1786) = 3 sizes × 3 algorithms × 2 formulations × 30 reps — **verified exactly against manifest (540 rows, 18 cells × 30)**. "270" (4-obj subset, lines 1248/1251/1308/1370/1400) = 3×3×30 — consistent.
- **Problem sizes:** small 5/150/50, medium 8/225/75, large 10/300/100 (lines 960–962) — **match manifest** (`n_centers`/`n_people`/`n_dir` per size, pop 100, gen 200, 30 reps).
- **Table 4 HV/CPU (lines 1196–1210):** recomputed from manifest — HV millions small 21.08/20.21/20.74, medium 18.81/17.35/18.33, large 13.98/12.86/13.59 **all match**; CPU within rounding.
- **H2a numbers in prose (lines 1062–1065):** large 13.98 vs 13.59, medium 18.81 vs 18.33, small 21.08 vs 20.74 — match table and manifest. p-values (0.027/0.001/<0.0001) consistent with Appendix B Table B1 (line 2024–2032).
- **Appendix B Mann–Whitney tables** present, both formulations, sourced from the named manifest (line 2009–2011, file exists).
- **Hypotheses H1–H4:** all stated (lines 192–207) and each given an explicit verdict — H1 (§7.5 verdict, lines 1364–1394: Spearman confirmed, projection geometric), H2a refuted (1059), H2c confirmed (1081), H3a qualified-confirm (1494–1497, 1588), H3b confirmed (1538), H4 refuted-inverted (1138). Discussion recap (1584–1596) matches.
- **DTLZ2 selector validation** (lines 997–1011) present, addressing adversarial-review concern; driver/manifest exist (`experiments/results/dtlz2_validation/`).
- **Figures:** fig5–fig9 all present in `../figures/`, all `\includegraphics`'d and all `\ref`'d. Figure↔label↔file 1:1.
- **Tables/equations:** all 10 table labels and all referenced equations resolve; no undefined cross-refs in build.
- **Back matter:** Data Availability, Author Contributions (CRediT), Funding, IRB, Informed Consent, Acknowledgments (incl. AI declaration), Conflicts of Interest, References — **all present** (lines 1785–1816) in MDPI order.
- **References:** 38 cited keys, **zero undefined citations**, **zero duplicate bib keys**, build resolves all.
- **Abstract:** **193 words (≤200 ✓)**; claims map to body (4R mapping, 540 runs / 3 sizes, rank-correlation = Spearman §7.1, FIS_2b hot-spot = H3a §8.1, reproducibility). Only the "comparable Pareto-front quality" framing is soft (M-2).
- **Residuals:** no TODO/FIXME/"[ … ]"/"ATRes"/"extended version"/"SRH Berlin"/"legacy" in rendered text. Affiliation correctly reads "SRH University Heidelberg" (line 50). "[...]" at lines 689/728 are legitimate ellipses inside Bruneau quotations, not placeholders. The only residuals are the LaTeX comments in CR-2.
- **Reproducibility claims:** repo URL correct, `v0.2.0` git tag exists, both Zenodo DOIs resolve to v0.2.0; "continuously tested, dependency-audited" supported by `.github/workflows/ci.yml` (runs `uv run pip-audit`) and CI test job; generator (`experiments/generate_instances.py`) and FIS module exist. No overclaim beyond what the repo supports.

---

## Recommended pre-submission action order
1. **CR-1** MDPI template conversion (the gate).
2. **CR-2** strip skeleton/placeholder/DRAFT comments; co-authors confirm CRediT.
3. **M-1** qualify "the first" (line 165).
4. **M-2 / m-4** align abstract + conclusion HV framing with the body's honest reference-sensitivity reading.
5. **M-4** confirm AI-use declaration is literally accurate.
6. **m-1, m-2** cosmetic: drop 3 orphan bib entries; refresh Zenodo record description.
