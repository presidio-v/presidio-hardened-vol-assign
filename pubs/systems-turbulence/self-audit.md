# Self-audit — Paper B (`paper.tex`), Systems (MDPI) SI on Economic Turbulence

Read-only pre-review gate. Manuscript: `pubs/systems-turbulence/author/paper.tex` (7 pages, builds
clean under `latexmk -pdf`, no undefined refs/citations, both referenced figures present).
Ground truth: `experiments/results/turbulence/{small,large}/turbulence_summary.csv`,
`experiments/results/repro/repro_manifest.csv`, `experiments/results/operability/operability_manifest.csv`,
per-realisation `turbulence_manifest.csv` files, `.github/workflows/ci.yml`, source under
`src/presidio_vol_assign/allocation/`.

Verdict: **numbers are accurate**, but there are two CRITICAL overclaims (CI/cross-environment
reproducibility; an undisclosed churn-saturation artifact that undermines the "degradation slope"
framing and the residual "under every turbulence type" language) that must be fixed before
co-authors see it.

---

## CRITICAL

### C1. RQ2 asserts a cross-environment CI matrix that does not exist
- **Locations:** §Method-repro (l.203) "…comparing the signature for matching (size, seed) across
  target OS/Python builds, **automated by the continuous-integration matrix**"; §RQ2 (l.324–326)
  "cross-environment reproducibility is then established by executing the same driver on each target
  OS / Python and comparing the signature column, **which the project's continuous-integration matrix
  automates**"; §The tool (l.157–158) "reproducible bit-for-bit on stock hardware." Also
  Limitations (l.384–385) and abstract wording.
- **Evidence:** `.github/workflows/ci.yml` runs `runs-on: ubuntu-latest` with a matrix over
  **Python versions only** (`["3.10","3.11","3.12"]`). There is **no OS matrix** (no macOS/Windows),
  and the CI job does **not** run `run_reproducibility.py` or diff any signature column — it runs
  ruff/pytest/pip-audit. The repro manifest (`repro_manifest.csv`) contains **one platform only**:
  `macOS-15.7.5-arm64`. Cross-environment reproducibility is therefore neither automated nor
  demonstrated.
- **Why it matters:** This is the single most exposed claim. A reviewer who opens the public repo
  will see the CI file in seconds. It also asserts cross-platform bit-for-bit reproducibility of a
  float/library-dependent solver (numpy/scipy/deap/pymoo) as an achieved fact — which is not
  guaranteed even in principle and is not shown.
- **Fix:** State only what the artifacts support. Cross-environment reproducibility is *defined and
  instrumented* (the signature carries an environment fingerprint; the manifest records it), but
  **not yet run across environments and not automated by CI**. Reword l.203 / l.324–326 to: the
  fingerprint *enables* cross-environment checking by diffing the signature column, which is left as
  future work / a planned CI step; drop "automated by the continuous-integration matrix." Keep the
  Limitations sentence (l.384–385) but make it consistent (it currently says CI "completes" it,
  which is still an overclaim). Do not assert cross-platform bit-for-bit; say bit-for-bit is shown
  **within a single environment**.

### C2. Undisclosed churn saturation makes the "degradation slope" framing misleading, and residual "under every turbulence type" language overstates RQ1
- **Locations:** Abstract (l.56–59) "re-routes **most** of the allocation … **far more than a crisp
  baseline**"; §RQ1 heading (l.247) and l.249 "the fuzzy churn slope exceeds the crisp baseline's in
  **every one of the eight cells**"; Fig.~\ref{fig:churn} caption (l.269–271) "The MOEA re-routes
  markedly more of the allocation **under every turbulence type**"; Discussion (l.360) and Conclusion
  (l.394) "re-routes **most** of the allocation."
- **Evidence (raw per-level manifests):** For **every** small-instance cell examined, fuzzy
  allocation churn is a *step function*, not a gradient: 0.00 at level 0, then ~**0.83 at level 0.05
  and flat through 0.40** (IDL noise: 0.837→0.845; centre-occ noise: 0.831→0.843; road-condition
  flip: 0.832→0.831). The reported per-realisation "slope" (~1.25–1.28) is a linear fit through a
  plateau; it is dominated by the level-0→0.05 jump and largely reflects ≈0.83/0.4 arithmetic, **not
  a dose-response degradation gradient**. Fig.~2 (`fig_degradation_infrastructure_damage_level_noise.pdf`,
  "mean ± 1 s.d. vs. turbulence level") will visibly show this flat plateau, contradicting the
  "degradation slope"/gradient language used throughout Method (l.189, l.218) and Results.
- **Two distinct problems:**
  1. **"most of the allocation" is defensible** on magnitude — fuzzy churn ≈0.83 (≈83% re-routed) at
     level 0.2 — but the paper never reports the churn *magnitude*, only slopes. State the ~0.8
     magnitude explicitly so "most" is anchored to data, and disclose the saturation (the system
     jumps to ~80% churn at the smallest perturbation and stays there — arguably *worse* for trust
     than a gradient, and worth saying plainly).
  2. **"under every turbulence type" / "in every one of the eight cells" overstates the localisation
     story.** Fuzzy churn is ~0.83 in *all* cells including centre-occupancy and road-condition —
     i.e. the churn fragility really *is* close to universal on the small instance; only the
     *realised-objective drift* is localised to objective-coupled inputs, and only at scale does the
     crisp churn catch up (centre-occ large: 1.44 vs 1.45, p=0.95). The reframe from "universal
     fragility" to "localised" was applied to the **drift** story but the **churn** prose still reads
     as universal ("under every turbulence type", "every one of the eight cells"). This is internally
     inconsistent with the paper's own thesis (l.254–255, l.302–307) that fragility is *concentrated*.
- **Fix:** (a) Report churn magnitude (~0.8 at level 0.2) and disclose the immediate saturation
  explicitly in §RQ1 and Method; soften "degradation slope" to "sensitivity slope" or note the metric
  saturates. (b) Reconcile the churn vs drift stories: be explicit that **churn is near-universal on
  the small instance** (~0.83 in all 8 cells) while **realised-quality degradation is localised** to
  objective-coupled inputs and **churn localisation only emerges at scale**. Remove "under every
  turbulence type" from Fig.~1 caption or scope it to the small instance and to churn specifically.

---

## MODERATE

### M1. Abstract/prose "far more than a crisp baseline" needs the centre/road exceptions surfaced earlier
- **Locations:** Abstract (l.57–59), Discussion (l.360).
- **Evidence:** On *realised-objective drift* the crisp baseline is the **more** fragile in
  centre-occupancy noise (small: 6.37 vs 9.04; large: 8.5 vs 23.4) and road-condition flip (small:
  4.85 vs 7.84; large: 5.4 vs 9.2). RQ1 body (l.257–264) handles this correctly, but the abstract's
  blanket "far more than a crisp baseline" reads as universal.
- **Fix:** The abstract already hedges with "whereas turbulence in other inputs attenuates the effect"
  — extend that clause to acknowledge the crisp baseline is *more* fragile on some inputs (not merely
  "attenuates"), matching the honest RQ1 text.

### M2. "possible-hazard flip" drift is presented as fuzzy-worse but is not significant
- **Location:** RQ1 small table (l.242): Poss. hazard / flip drift fuzzy 5.4 vs crisp 4.3, p=0.151.
- **Evidence:** `turbulence_summary.csv` confirms 5.35968 / 4.30768 / p=0.15137 — **not significant**.
  The table shows the p but the surrounding prose (l.257–260) groups drift worsening under
  "person-priority and travel inputs" and cites p ≤ 0.0005; possible-hazard is neither
  objective-coupled-in-the-strong-sense nor significant. Numbers are right; just make sure no prose
  implies significance here.
- **Fix:** None required in the table (p is shown). Optionally add a half-sentence that hazard-flip
  drift is n.s. so a reader does not over-read the 5.4>4.3.

### M3. Thin reference list (3 works) — Background makes multiple citation-needing claims with none
- **Location:** `lit.bib` has exactly 3 entries (`rabiei2021`, `rabiei2023`, `paperA2026`); §Background
  (l.104–129) is almost entirely uncited.
- **Specific unsupported claims that need citations:**
  - l.110–112 "Mamdani fuzzy inference systems encode expert judgement as linguistic if–then rules…
    attractive precisely when inputs are imprecise" — needs a Mamdani / fuzzy-inference reference
    (e.g. Mamdani & Assilian; Zadeh).
  - l.117–119 "Algorithms such as NSGA-II and NRGA return a Pareto front of non-dominated trade-offs"
    — needs NSGA-II (Deb et al. 2002) and an NRGA reference. NSGA-II/NRGA are named repeatedly
    (abstract, l.54, l.80, l.117, l.143, l.168) with **no primary citation anywhere**.
  - l.123–126 "Robustness of multi-objective methods is usually studied against algorithmic
    stochasticity or against perturbation of elicited preferences" — a literature claim ("usually")
    with zero support; needs 2–3 robustness/decision-stability citations or must be softened.
  - l.127–129 "Far less attention is paid to stability of the committed decision under degradation of
    the input data" — the paper's gap claim; currently rests on nothing. Needs either supporting
    citations or explicit hedging as the authors' observation.
- **Why it matters:** MDPI reviewers routinely flag under-referenced Backgrounds; a 3-citation paper
  that cites only the authors' own work reads as insufficiently situated and risks a self-citation
  flag (all 3 are author works).
- **Fix:** Add primary sources for Mamdani FIS, NSGA-II, NRGA, and 2–4 works on MOEA
  robustness/decision stability under uncertainty; soften or support the "usually studied"/"far less
  attention" literature claims.

### M4. Economic-turbulence framing — check it is hedged as analogy throughout
- **Locations:** Title/SI framing; Intro (l.72–78) opens on "high economic turbulence"; Limitations
  (l.381–382) correctly says "the economic-turbulence setting is engaged **by analogy** rather than
  with market data."
- **Assessment:** The hedge exists and is good, but it lives only in Limitations. The Intro (l.72–78)
  presents post-disaster relief as an "acute instance" of economic turbulence without flagging that
  the mapping (relief demand shock ↔ economic shock) is an analogy. For an SI explicitly about
  *economic* turbulence with an editor expecting economic relevance, one analogy caveat buried on the
  last page is thin.
- **Fix:** Move/echo the "by analogy" hedge into the Introduction where the economic framing is first
  asserted, so the framing is honestly scoped from the outset rather than only in Limitations.

---

## MINOR

### N1. Back matter [DRAFT] items still open (expected, but list them for the gate)
- `\authorcontributions` (l.406) — "[DRAFT — confirm with co-authors]".
- `\dataavailability` (l.414) — "[DRAFT]"; Zenodo DOI not yet minted ("will be minted").
- `\acknowledgments` (l.416) — "[DRAFT — AI-use disclosure to be confirmed]". **AI-use disclosure is
  empty** — must be completed before submission per MDPI policy and per the project's own AI-use
  convention.
- Present and complete: `\funding` (no external funding), `\institutionalreview` (Not applicable,
  synthetic data), `\informedconsent` (Not applicable), `\conflictsofinterest` (none). CRediT is
  present but flagged draft.

### N2. RQ3 "H-B3 is supported" / "H-B2" hypothesis labels appear without being defined in-paper
- **Locations:** §RQ3 (l.336) "H-B3 is supported"; the results notes reference H-B2 (RQ2), H-B3 (RQ3).
- **Evidence:** `explore/rq2-rq3-results.md` uses H-B2/H-B3 labels, but `paper.tex` never introduces
  an "H-B1/H-B2/H-B3" hypothesis scheme (no hypotheses are numbered in Intro/Method). The bare
  "H-B3 is supported" is a dangling reference to internal notation.
- **Fix:** Either introduce numbered hypotheses (H1 input-fragility, H2 reproducibility, H3
  operability) in the Intro and refer to them consistently, or drop "H-B3 is supported" and state the
  conclusion in words.

### N3. Numeric spot-check — all verified correct (record for the gate)
- **RQ1 small table (8 cells)** vs `small/turbulence_summary.csv`: IDL noise 1.28/0.49 churn (csv
  1.27937/0.485), 10.0/4.3 drift ✓; IDL missing 1.27/0.16, 7.2/3.2 ✓; resource-time 1.28/0.42,
  12.2/2.3 ✓; travel-dur 1.30/0.62, 10.5/5.0 ✓; centre-occ noise 1.25/0.73 (p=0.0034), 6.4/9.0
  (p=0.064) ✓; centre-occ missing 2.06/0.00, 10.8/0.0 ✓; road-cond flip 1.24/0.78, 4.8/7.8 (p=0.009)
  ✓; poss-hazard flip 1.26/0.72, 5.4/4.3 (p=0.151) ✓. All match.
- **RQ1 large table (4 cells)** vs `large/turbulence_summary.csv`: IDL noise 1.44/0.37, 7.5/2.4
  (p=0.008) ✓; IDL missing 1.42/0.16 (p=0.008), 5.4/4.8 (p=0.55) ✓; centre-occ noise 1.45/1.44
  (p=0.95), 8.5/23.4 (p=0.016) ✓; road-cond flip 1.42/1.16, 5.4/9.2 (p=0.008) ✓. All match. In-text
  values (l.253, l.261–263) also match.
- **RQ2:** "twenty runs … within-environment REP = 1.0" ✓ (manifest = 2 sizes × 10 seeds = 20 rows,
  all `in_process_rep=1.0`); "each distinct seed yields a distinct signature" ✓ (20 distinct
  signatures). Note explore notes say "10 distinct signatures" *per size*; paper says twenty runs each
  distinct — both consistent.
- **RQ3 table** vs recomputed medians from `operability_manifest.csv`: small NSGA-II 6.20 / 580 / 1.4
  ✓; small NRGA 4.34 / 829 / 1.4 ✓; large NSGA-II 8.99 / 400 / 2.6 ✓; large NRGA 6.78 / 531 / 2.6 ✓.
  Throughput = 3600/median-latency confirmed. "4–9 seconds", "≤3 MB", "≈400–830 dec/h" all consistent.
  (Minor: two operability rows show degenerate `nns` fronts — small nrga seed 72271 nns=60, large nrga
  seed 56433 nns=45 — but medians are unaffected; no action needed.)

### N4. Story consistency — largely coherent
- The through-line (input-fragility → reproducibility/auditability as load-bearing guardrails →
  operability) is consistent across abstract, intro (l.90–98), RQ1 close (l.308–310), RQ2 (l.326–327),
  Discussion (l.372–378), Conclusion. The only internal contradiction is the churn "universal" vs
  "localised" tension flagged in **C2** — fix that and the story is clean.

### N5. Document class / build
- `\documentclass[systems,article,submit,moreauthors,pdftex]{Definitions/mdpi}` ✓ correct MDPI
  `systems` class. Builds to 7 pages, no undefined refs or citations, both figures resolve. bib has 3
  entries; `.bbl` generated cleanly. No LaTeX errors.

---

## Priority order for the authors
1. **C1** (CI/cross-environment overclaim) — highest exposure, trivially checkable by any reviewer.
2. **C2** (churn saturation + "universal" vs "localised" inconsistency) — undermines the core RQ1
   framing; requires reporting churn magnitude and reconciling the two stories.
3. **M3** (thin/self-only references) — likely a first-round reviewer complaint.
4. **M1, M4** (honest scoping of "far more than crisp" and the economic analogy).
5. **N1** (finish AI-use disclosure + Zenodo DOI before submission).
