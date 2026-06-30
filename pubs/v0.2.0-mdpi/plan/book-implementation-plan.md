---
phase: II-G
project: presidio-hardened-vol-assign
created: 2026-05-09
template-source: ~/.claude/book-implementation-plan.md
---

## Publication Metadata

- **Working title:** *Resilient Last-Mile Allocation in Humanitarian Supply
  Chains: A Reproducible Many-Objective Fuzzy Framework*
- **Type:** Journal article (extended version of an in-press journal paper)
- **Target venue:** Applied Sciences (MDPI), Special Issue *Innovations in
  Supply Chain Resilience* (Guest Editors: Agnusdei, Silvestri, Di Pietro)
- **Submission deadline:** 2026-07-04 (hard); 2026-06-27 target (Week 7)
- **Language:** English
- **Version:** v0.2.0
- **Top message:** *The 4R framework of supply chain resilience can be
  operationalised, end-to-end, as a four-objective humanitarian last-mile
  allocation model whose objectives are evaluated by fuzzy expert reasoning
  and solved by reference-point-based many-objective evolutionary search —
  and the resulting trade-offs are stable under the kinds of expert-elicitation
  noise that real practitioners produce.*

  <!-- Every section connects back to this. The 4R↔objective mapping is
       the theoretical hook (§3.5); the 4-objective Pareto trade-off
       analysis (§6) is the empirical payoff; the sensitivity study (§7)
       is the practitioner-credibility close. -->

---

## LaTeX Setup

**Template:** MDPI Applied Sciences official LaTeX template
(`Definitions/applsci-mdpi.cls`). Download from
https://www.mdpi.com/authors/latex; place under `author/Definitions/`.

```latex
\documentclass[applsci,article,submit,pdftex,moreauthors]{Definitions/mdpi}

% --- MDPI standard preamble (do not modify) ---
\firstpage{1}
\makeatletter
\setcounter{page}{\@firstpage}
\makeatother
\pubvolume{15}
\issuenum{1}
\articlenumber{0}
\pubyear{2026}
\copyrightyear{2026}
\externaleditor{Academic Editor: Leonardo Agnusdei}
\datereceived{}
\dateaccepted{}
\datepublished{}
\hreflink{https://doi.org/}

% --- Article metadata ---
\Title{Resilient Last-Mile Allocation in Humanitarian Supply Chains:
       A Reproducible Many-Objective Fuzzy Framework}
\TitleCitation{Resilient Last-Mile Allocation in Humanitarian Supply Chains}
\Author{Peyman Rabiei$^{1}$, Daniel Arias-Aranda$^{1}$, Vladimir Stantchev$^{2,*}$}
\AuthorNames{Peyman Rabiei, Daniel Arias-Aranda, Vladimir Stantchev}
\address{%
$^{1}$ \quad Faculty of Economics and Business, University of Granada, Spain\\
$^{2}$ \quad Institute of Information Systems, SRH University Heidelberg, Germany}
\corres{Correspondence: stantchev@computer.org}

% --- Bibliography ---
\externalbibliography{yes}
\bibliography{lit}    % use project-local lit.bib (MDPI prefers single-file
                      % submission; copy relevant entries from
                      % ~/pubs/lit/lit.bib at build time)
```

**Prior-publication footnote** (mandatory per MDPI extended-version rule):

```latex
\thanks{This paper is an extended version of Rabiei, P.; Arias-Aranda, D.;
Stantchev, V. \emph{A Novel Multi-Objective Optimization Model for Directing
Affected People to Relief Centers in Post-Disaster Scenarios: Combining
Fuzzy Inference Systems with NSGA-II and NRGA.} \emph{Autonomous
Transportation Research}, in press, 2026 (manuscript ATRES-S-26-00021).
The present paper extends that work by reframing the model in supply chain
resilience theory, splitting the transportation infeasibility objective
into separate Robustness and Rapidity components, adding NSGA-III as a
third evolutionary algorithm, introducing rule-base and weight-perturbation
sensitivity analyses, and releasing a Python reference implementation as
an open-source artifact.}
```

---

## Top-Level Structure

Single-file `paper.tex`, journal-article layout. No `\include{}`; no
glossary, no index. MDPI uses footnote-style endnotes only as exceptions.

| § | Title | Word target | Status |
|---|-------|-------------|--------|
| Abstract | (200 words max per MDPI) | 200 | ☐ |
| 1 | Introduction | 1,200 | ☐ |
| 2 | Related Work and Theoretical Background | 1,800 | ☐ |
| 3 | Model Formulation | 2,000 | ☐ |
| 4 | Algorithm Design | 1,200 | ☐ |
| 5 | Performance Analysis | 1,800 | ☐ |
| 6 | Pareto Trade-off Analysis (4-obj vs. 3-obj) | 1,200 | ☐ |
| 7 | Sensitivity Analysis | 1,200 | ☐ |
| 8 | Discussion | 1,200 | ☐ |
| 9 | Conclusions | 400 | ☐ |
| Appendix A | FIS rule bases (full tables) | — | ☐ |
| Appendix B | Statistical test outputs (extended) | — | ☐ |
| References | (BibTeX, MDPI style) | — | ☐ |
| **Total main text** | | **≈ 12,000** | |

### Section-by-section content map

**§1 Introduction.**
- Lead with disaster frequency claim (CRED 2023 numbers — keep ATRes
  framing for one paragraph as continuity hook).
- Pivot fast to humanitarian supply chain resilience as the lens.
- Bruneau's 4R named in the second or third paragraph.
- Problem statement: bounded one paragraph (lift from
  `explore/problem-statement.md`).
- Five contributions, listed numbered:
  1. First explicit 4R↔objective mapping in FIS-MOEA disaster allocation;
  2. 4-objective formulation that splits TIL into Robustness/Rapidity;
  3. Empirical comparison of NSGA-II, NRGA, and NSGA-III on this problem;
  4. FIS rule-base and weight-perturbation sensitivity analyses;
  5. Open-source Python reference implementation (`pva` v0.2.0,
     Zenodo DOI in Data Availability statement).
- One-paragraph paper structure ending §1.

**§2 Related Work and Theoretical Background.**
- §2.1 Supply chain resilience and the 4R framework (Bruneau 2003;
  Christopher & Peck 2004; Hosseini 2016; Tukamuhabwa 2015; Behl & Dutta 2019).
- §2.2 Humanitarian last-mile allocation (Hashim 2021; Zhao 2017, 2019;
  Gama 2016; Chang 2024; Hansuwa 2021; Soghrati Ghasbeh 2022; Wang 2020;
  Abounacer 2014; Tzeng 2007; Beiki 2020; Krishna 2021).
- §2.3 FIS-MOEA hybrids in disaster operations (Mamdani 1975; Sahoo 2019;
  Rabiei 2021, 2023; ATRes 2026).
- §2.4 Many-objective evolutionary algorithms (Deb 2002 NSGA-II; Deb &
  Jain 2014 NSGA-III; NRGA — Omar Al Jadaan et al. 2008; Zheng & Doerr
  2024 IEEE TEC).
- Closing paragraph: explicit gap statement — *no published model
  organises FIS-evaluated objectives around a named resilience framework
  and ships a reproducible artifact*.

**§3 Model Formulation.**
- §3.1 Problem definition (lift from ATRes §3.1, condense).
- §3.2 Resilience-anchored qualitative indices:
  §3.2.1 FLPP (Resourcefulness)
  §3.2.2 **TR** Transportation Robustness — *new* (split from TFL)
  §3.2.3 **RP** Rapidity — *new* (split from TFL)
  §3.2.4 CABL (Redundancy)
- §3.3 Notations and variables (extend ATRes Table 1; add TRD, RPD).
- §3.4 Objective functions (4 of them; equations renumbered).
- §3.5 **Resilience-theoretic mapping** — *new section*. Table mapping
  each objective to one 4R component, with two-paragraph theoretical
  justification per row. This is the contribution that justifies the
  paper's existence in the SI.

**§4 Algorithm Design.**
- §4.1 FIS implementation (4 FISs; structure unchanged from ATRes,
  rule bases reorganised — full tables in Appendix A).
- §4.2 MOEAs: NSGA-II, NRGA, NSGA-III — half page each, focusing on
  selection mechanism differences.
- §4.3 Chromosome encoding (reused from ATRes).
- §4.4 **Reference-point design for NSGA-III** — *new subsection*.
  Das-Dennis p=4, 35 reference points, justification.

**§5 Performance Analysis.**
- §5.1 Experimental setup (machine, software, seeds, repetitions,
  reference points, 4D HV via pymoo). Replace ATRes's MATLAB R2024b
  paragraph with `pva` v0.2.0 + Python toolchain + GitHub/Zenodo
  references.
- §5.2 Test instances (small 5/150/50; medium 8/225/75 — *new*; large
  10/300/100). Pareto fronts at each size.
- §5.3 Statistical comparison: H2a (HV), H2b (NNS, SM), H2c (CPU), H4
  (NSGA-II vs. NRGA replication). Tables analogous to ATRes Tables 6–14
  but extended with NSGA-III column.
- §5.4 Summary of algorithm comparison.

**§6 Pareto Trade-off Analysis.** *(new section, addresses RQ1/H1)*
- §6.1 Spearman rank correlation of TRD vs. RPD across 4-obj fronts.
- §6.2 Projection-dominance analysis (4-obj front → 3-obj space using
  ATRes RWS weighting; count solutions lost in projection).
- §6.3 Three practitioner trade-off vignettes (parallel-coordinates plot):
  - "Vulnerable individual on a fragile road" — high Resourcefulness,
    low Robustness, modest Rapidity.
  - "Healthy individual on a quick fragile route vs. slow robust route"
    — Robustness/Rapidity trade-off the 3-obj formulation hides.
  - "Capacity-constrained center, multiple equally-prioritised
    individuals" — Redundancy under load.
- Closing paragraph: state the empirical answer to H1.

**§7 Sensitivity Analysis.** *(new section, addresses RQ3/H3)*
- §7.1 Rule-base perturbation: single-rule deletion across all 4 FISs;
  ΔHV distribution per FIS; identify which rules carry the most weight.
- §7.2 Weight perturbation: Latin Hypercube sample of (WAS, WDS, WIL,
  WLS, WRC, WPH) ∈ ±20%; CV of mean objectives; partial dependence
  plots for the two most influential weights.
- §7.3 Practitioner takeaway: which expert-elicitation choices matter,
  which do not.

**§8 Discussion.**
- §8.1 Findings vs. SCR theory — does the 4R operationalisation reveal
  anything new about the framework itself?
- §8.2 Findings vs. MOEA literature — when does NSGA-III pay off at
  4 objectives in this problem class?
- §8.3 Practitioner implications — the trade-off vocabulary; reproducibility.
- §8.4 Limitations — only synthetic instances; only one disaster type
  modelled; FIS rule bases are expert-defined (subjective).
- §8.5 Future work — real-world case study (v0.3.0); multi-modal
  transport; psychological vulnerability index.

**§9 Conclusions.**
- One-paragraph restatement of top message.
- Five contributions in numbered form (mirror §1).
- One-sentence pointer to the open artifact and Zenodo DOI.

---

## Author Style

**Primary style profile:** `~/.claude/vladimir-prose-style.md`.

**Adaptation for English academic prose** (the source profile is German;
register markers do not transfer directly):

| Source marker (German) | English adaptation | Use here? |
|---|---|---|
| Aphoristic opener — claim first | *Yes.* Section openings lead with the claim, not a definition. "Disaster frequency is rising" beats "This section reviews recent disaster trends." | ☑ |
| Extended analogy with narrative arc | *Yes, sparingly.* One per major section, no more. Disaster context limits the comic register. | ☑ |
| Code-switching DE/EN | *No.* All English. | ☐ |
| Binary framings | *Yes.* "Either the 4-obj formulation exposes a Pareto trade-off the 3-obj cannot, or the split is information-redundant — H1 tests which." | ☑ |
| First-person plural ("we") | *Yes.* MDPI accepts active voice. | ☑ |
| First-person singular ("Ich persönlich…") | *No.* Three-author paper; first-person plural only. | ☐ |
| German idioms as punchlines | *No.* Replace with restrained English aphorism if any. | ☐ |
| Concrete actors, not abstract subjects | *Yes.* "Civil-protection planners" not "the literature"; "The decision-maker" not "stakeholders". | ☑ |
| Irony, light sardony | *Sparingly.* One per Discussion section maximum. | ☑ (limited) |

**Forbidden phrases** (lifted verbatim from the style profile, English equivalents):

- "It is important to note that…" / "It should be noted that…"
- "In an era when X plays an increasingly Y role, …"
- "Not only X, but also Y" — especially doubled.
- "Furthermore," / "Moreover," / "Additionally," as paragraph starters.
- A *Conclusions* section that restates §1 verbatim.
- "Researchers have studied…" — name them, or state the gap directly.
- Bullet lists of three bold-labelled items as a substitute for prose.

---

## Writing Sequence

```
☐ 1. LaTeX template setup
     Download applsci-mdpi.cls; clean build on empty document; verify
     bibliography compiles against project lit.bib.

☐ 2. Top-level structure
     All 9 section headers + appendix headers in place. Empty subsections.
     Verify build, TOC absent (MDPI articles do not include TOC).

☐ 3. Top message confirmed
     Already written above. Re-read at start of every drafting session.

☐ 4. Tier 1 pass (skeleton)
     One-sentence placeholder per subsection.
     Figure and table placeholder \label{}s in place.
     All cross-references resolved (no "??" in PDF).

☐ 5. Tier 2 pass (functional prose, by section)
     Order: §3 (model — least dependent on results) → §4 → §5 → §6 → §7
            → §2 (related work — easier once §3–§7 are clear) → §1 → §8 → §9.
     All citations added. lit.bib entries verified.
     No style polish.

☐ 6. Anecdote / vignette insertion
     §6.3 — three practitioner vignettes (Pareto trade-off scenarios).
     §1 — one motivating example (e.g., 2024 Valencia floods reference).
     §8.3 — one implication scenario.

☐ 7. Tier 3 voice pass
     Apply English-adapted style markers section by section.
     Commit after each section.

☐ 8. Figures finalised
     9 figures (see Figure budget below).
     Vector format (PDF or TikZ) preferred; raster only where unavoidable.
     Captions define non-obvious metrics on first use.

☐ 9. Bibliography check
     Every \cite{} key in lit.bib.
     DOI underscores escaped: doi={10.1007/...\_48}.
     0 bibtex warnings.

☐ 10. Full build
      pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper.
      Final clean build.

☐ 11. Bad box pass
      No overfull hbox > 5pt on body text. Tables in tabularx if wide.

☐ 12. Reference audit (per ~/.claude/feedback_reference_audit.md)
      Every \label{fig:*} and \label{tab:*} appears in at least one
      \ref{} call in prose. Run grep check before submission.

☐ 13. Adversarial review pass
      Severity-labelled review filed at adversarial-review.md.
      All Critical issues resolved.

☐ 14. Cover letter
      cover-letter-applsci.md (working) and -submit.pdf (clean).
      ATRes prior-publication disclosure paragraph included.
      arXiv preprint disclosure if posted.

☐ 15. Submission
      MDPI portal upload. Verify all files (manuscript PDF, source TeX,
      figures, supplementary).
```

---

## Figure budget

| ID | Caption (short) | Type | Source | Reuse from ATRes? |
|---|---|---|---|---|
| Fig 1 | Mamdani FIS structure | TikZ | own | reuse |
| Fig 2 | Chromosome structure | TikZ | own | reuse |
| Fig 3 | 4R↔objective mapping diagram | TikZ | new | new |
| Fig 4 | NSGA-III reference points (4D simplex projection) | matplotlib | new (`pva` figure) | new |
| Fig 5 | Pareto fronts on 3 problem sizes (3 panels) | matplotlib | `pva` outputs | extends ATRes |
| Fig 6 | TRD vs. RPD Spearman heatmap across instances | matplotlib | new | new |
| Fig 7 | ΔHV distribution by FIS rule-deletion (4 panels) | matplotlib | new | new |
| Fig 8 | Weight LHS partial dependence (top-2 weights) | matplotlib | new | new |
| Fig 9 | Three trade-off vignettes (parallel coordinates) | matplotlib | new | new |

7 of 9 figures are new content. Combined with §3.5, §6, §7 entirely new,
this clears MDPI's >50% threshold by a wide margin.

---

## Build Process

```bash
cd pubs/v0.2.0-mdpi/author/
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

Build artifacts to `.gitignore`: `*.aux`, `*.bbl`, `*.blg`, `*.log`,
`*.out`, `*.fdb_latexmk`, `*.synctex.gz`.

---

## Quality Checklist

```
☐ All \cite{} keys exist in lit.bib
☐ All \ref{} targets have matching \label{} (no "??" in output)
☐ Every \label{fig:*} and \label{tab:*} appears in at least one \ref{}
☐ No overfull hbox > 5pt
☐ DOI underscores escaped in lit.bib
☐ Figure captions define non-obvious metrics on first use
☐ Top message visible in §1, §3.5, §6, §9
☐ Author voice (Tier 3 pass) applied throughout
☐ Page count: 15–25 pages typical for MDPI Applied Sciences
☐ Bibliography: 0 bibtex warnings
☐ ATRes prior-publication footnote on title page (\thanks{...})
☐ Data Availability statement cites Zenodo DOI for `pva` v0.2.0
☐ No AI/Claude attribution in prose, comments, or git commits
☐ Adversarial review complete; Critical issues resolved
☐ Cover letter discloses ATRes parallel/prior submission
☐ Forward feasibility probe complete for v0.3.0 (case study)
```

---

## Cross-reference to other plans

- **Code work:** `~/.claude/code-implementation-plan.md` instantiated for
  `pva` v0.2.0 — `presidio-hardened-vol-assign/PRESIDIO-REQ.md` records
  v0.2.0 requirements; `code-implementation-plan.md` records the
  step-by-step.
- **Data and experiments:** all experiment scripts under
  `presidio-hardened-vol-assign/experiments/` (to be created in Phase III);
  results under `pubs/v0.2.0-mdpi/figures/data/`; figure scripts under
  `pubs/v0.2.0-mdpi/figures/scripts/`.
- **Cover letter:** `pubs/v0.2.0-mdpi/cover-letter-applsci.md`.
- **Adversarial review:** `pubs/v0.2.0-mdpi/adversarial-review.md`.
