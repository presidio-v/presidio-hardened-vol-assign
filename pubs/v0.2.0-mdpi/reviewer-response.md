# Response to ATRES reviewers (ATRES-D-26-00014)

This table maps **every** comment from the ATRES rejection (editor, R1.1–1.4,
R2.1–2.5, R3 global + line-by-line) to its disposition in the resubmitted
manuscript (`pubs/v0.2.0-mdpi/author/paper.tex`, retargeted to *Applied Sciences*
MDPI, SI "Innovations in Supply Chain Resilience").

**Disposition key**

- **Addressed** — handled in the prose of this submission.
- **Deferred (S2)** — requires new experiments; out of scope for this
  writing/framing pass, planned for Submission 2 (the supporting code already
  exists; see §8.5 and the rejection-review §2.3/§4).
- **N/A (rewritten)** — the specific line no longer exists; the section was
  rewritten, so the original wording-level fix is moot.
- **[AUTHOR CONFIRM]** — needs a co-author decision before upload.

Scope was deliberately confined to writing and framing: no new experiments, no
figure regeneration, no code changes. Items that the verbatim comments flagged
but that require new runs are listed as **Deferred (S2)** rather than silently
dropped.

---

## Editor

| # | Comment | Disposition | One-line fix |
|---|---|---|---|
| E1 | Topic out of scope for an autonomous-transportation journal | **Addressed** | Manuscript retargeted to *Applied Sciences* (MDPI) SI on Supply Chain Resilience — an in-scope venue; framing now leads with SCR/4R. |

## Reviewer 1

| # | Comment | Disposition | One-line fix |
|---|---|---|---|
| R1.1 | Reads as a modelling problem, not a framework; no visible MOO improvement | **Addressed** | Intro now states the contribution is a 4R reframing + reproducible tool, *explicitly not a new MOO operator*; solvers are used to characterise the model, not advanced. |
| R1.2 | "AI taste"; dry; add an AI-use acknowledgement if AI tools were used | **Addressed** + **[AUTHOR CONFIRM]** | Humanizing copy-edit (removed triplicate antithesis, cut em-dash/rule-of-three tells); AI-use acknowledgement added to back matter as an integrity statement with an [AUTHOR CONFIRM] placeholder. |
| R1.3 | Assumptions need social-science evidence; more literature | **Addressed (partial)** | Bibliography expanded 21→45; fairness grounded as need-weighted Rawlsian maximin; deeper social-science case-study evidence is **Deferred (S2)**. |
| R1.4 | Comparison insufficient; why these MOO methods; attach data for reproducibility | **Addressed (partial)** | NSGA-III added with rationale (§2.4, §4.2); seeds, parameters, repeated runs, open code + Zenodo DOI surfaced (§5.1); external-baseline comparison **Deferred (S2)**. |

## Reviewer 2

| # | Comment | Disposition | One-line fix |
|---|---|---|---|
| R2.1 | Novelty limited (three indicators; FIS+EA not new) | **Addressed** | Contribution reframed around the 4R bijection + artifact; novelty no longer claimed in the operator. |
| R2.2 | Indicators (FLPP/TFL/CABL) lack theoretical/empirical justification | **Addressed (partial)** | Per-component 4R derivation added (§3.5); statistical non-redundancy via ablation referenced; full empirical/expert validation **Deferred (S2)**. |
| R2.3 | Expert-driven rule base; no sensitivity/validation | **Addressed** | §7 adds rule-base deletion (H3a) and weight-perturbation (H3b) sensitivity at every problem size. |
| R2.4 | Compares NSGA-II/NRGA, not the framework vs an existing method | **Deferred (S2)** | Reframed out of the contribution claim; greedy + exact baselines exist in the tool; full benchmark is Submission 2 (§8.5). |
| R2.5 | Omits shelter capacity and transport limits | **Addressed (partial)** | Capacity is the CABL/Redundancy objective and transport is the TRD/RPD objectives (soft); **hard** capacity/transport-limit mode **Deferred (S2)**, signposted in §8.5. |

## Reviewer 3 — main + global

| # | Comment | Disposition | One-line fix |
|---|---|---|---|
| R3.M1 | Objectives too vague | **Addressed** | Explicit H1–H4 in §1; plain-language intuition before every objective block; "why the objectives conflict" paragraph (§3.4). |
| R3.M2 | No benchmark against a known method | **Deferred (S2)** | See R2.4. |
| R3.M3 | Poorly written; reader must dig through 14 tables of raw data | **Addressed** | Copy-edit pass; raw stats tables confined to Appendix B; five summary figures carry the results. |
| R3.G1 | Lit review doesn't justify the study | **Addressed** | §2 documents survey scope and states the converging gap; negative claims hedged "to the best of our knowledge." |
| R3.G2 | Fuzzy systems not introduced; don't assume an expert reader | **Addressed** | New §4.1 primer: membership functions, triangular/trapezoidal numbers, fuzzify→fire→defuzzify, with a worked partial-membership example. |
| R3.G3 | Overuse of "qualitative" (distance is not qualitative) | **Addressed** | Indices relabelled "fuzzy-aggregated"; terminology note: inputs quantitative, only FIS output qualitative. |
| R3.G4 | Whole paper reads like a rough draft | **Addressed** | Structural + prose revision across all sections. |
| R3.G5 | §3 gives no intuitive description of the objectives | **Addressed** | Intuition paragraph + conflict explanation precede the equations (§3.4). |
| R3.G6 | §3 makes more sense after §4; restructure | **Addressed (verified)** | Order is already Model→Algorithm→Performance; rule tables moved to Appendix A; no blind reorder needed. |
| R3.G7 | 14 pages of tables | **Addressed** | Raw tables in appendices; body carries summary tables + figures only. |
| R3.G8 | Only two figures, none of results | **Addressed** | Five figures (Figs 5–9), including Pareto fronts and all results (H1/H2/H3). |
| R3.G9 | No clear objective/hypothesis | **Addressed** | Explicit hypothesis + four-objective contribution stated in §1. |
| R3.G10 | Implicit hypothesis ⇒ compare against an existing method | **Addressed (framing)** + **Deferred (S2)** | Hypothesis made explicit; the external-method comparison is Submission 2. |

## Reviewer 3 — line-by-line

| Page/line | Comment | Disposition | One-line fix |
|---|---|---|---|
| p4 l11–13 | Global or regional? | **Addressed** | "399 natural disasters worldwide" (EM-DAT) — explicitly global. |
| p5 l9–10 | Poorly structured sentence/paragraph | **N/A (rewritten)** | Paragraph rewritten. |
| p6 l36–40 | Gerunds ("getting…planning…") | **N/A (rewritten)** | Text rewritten. |
| p7 l31 | Period→comma / capitalise "such" | **N/A (rewritten)** | Sentence rewritten. |
| p7 l41–56 | Combine the two paragraphs | **N/A (rewritten)** | Section rewritten. |
| p7 | Why can the objectives conflict? | **Addressed** | Conflict paragraph added (§3.4). |
| p8 l28–30 | Feasibility/balance not yet introduced | **Addressed** | Redundancy/balance defined in §3.2.4 and derived in §3.5. |
| p8 | Why qualitative not quantitative? | **Addressed** | Terminology note (quantitative inputs, qualitative outputs). |
| p8 l46–50 | No fuzzy intro or justification in lit review | **Addressed** | §2.3 justifies FIS; §4.1 primer introduces it. |
| p9 l24 | "exceeds much further" | **N/A (rewritten)** | Now "exceeds the 20-year baseline." |
| p9 l59 | "highlighting"→"exacerbation" of inequalities | **N/A (rewritten)** | Phrase no longer present. |
| p10 l4–9, l19–21 | Confusing sentences | **N/A (rewritten)** | Section rewritten. |
| p10 l22–36 | Mix of qualitative/quantitative indices | **Addressed** | Terminology note clarifies the input/output distinction. |
| p10 l37–41 | Are sub-indices equally weighted? Say earlier | **Addressed** | VS/RWS weights and defaults (all 1) stated at first use (§3.3); sensitivity in §7. |
| p11 l11–19, l47 | Travel duration is quantitative | **Addressed** | Terminology note; relabelled. |
| p12 §3.3 | A section can't be a list+table | **N/A (rewritten)** | §3 is prose; notation is a reference table by design. |
| p13 Table 1 | Don't make readers comb variables; move to appendix | **Addressed (partial)** | Concepts introduced as prose; full rule tables in Appendix A; notation table retained for reference. |
| p16 l19–20 | Notation unclear (fuzzy logic?) | **Addressed** | Primer + notation table clarify. |
| p16 Eq 1–3 | What is the objective function? | **Addressed** | Objectives named (min ULPP/TRD/RPD/CAIL) with intuition before the equations. |
| p16 l45–46 | "Travel"→"travel" | **Addressed** | Lower-cased in prose. |
| p16–17 Eq 4–7 | Cannot follow; no intuition | **Addressed** | Plain-language lead-ins added (§3.4). |
| p18 end §3 | Unclear how objectives are formed | **Addressed** | Intuition + conflict + primer remove the onus on the reader. |
| p19 l37–40 | What are membership functions? | **Addressed** | Defined in §4.1 primer. |
| p19 l41–43 | What are triangular/trapezoidal numbers? | **Addressed** | Defined in §4.1 primer. |
| p20 | Why GA over classical methods; why NSGA-II/NRGA over others? | **Addressed (partial)** | §2.4 + §4.2 give the reference-point-boundary rationale; exact baseline justification deepened in **S2**. |
| p20 | Not first to combine GA+fuzzy — be nuanced | **Addressed** | §2.3 nuanced ("introduced into this line of work"; adjacent work cited). |
| p20 | A paragraph on how GAs help | **Addressed** | §4.2 + primer give the motivation. |
| p21 | "are outlined?" vague | **N/A (rewritten)** | Phrase removed. |
| p21 expr 17 | Real-valued $C_r$ vague; why not an n×1 chromosome? | **Addressed** + **[AUTHOR CONFIRM]** | §4.3 rewritten: two named slices, every chromosome feasible; answers "why not n×1" (subset selection needs the person slice). Kept faithful to `solvers.py` (2·n_dir, real-valued centre gene); any migration to a literal integer length-n encoding is a **code** change for S2 — see note below. |
| p21 Tables 2–5 | Introduce rule tables earlier | **Addressed (partial)** | Rule tables referenced from §4.1; retained in Appendix A to keep the body readable. |
| p29 Eq 18 | MID is a poor MOO measure; remove, HV suffices | **Addressed** | MID demoted to a diagnostic, moved to the last column of every table, footnoted with the eq.-18 rationale; HV is primary throughout. |
| p30 l4–7 | Missing pop size / generations / stopping / seeds / repeated runs | **Addressed** | §5.1 states pop 100, gen 200, crossover/mutation rates, per-rep seeds, 30 reps. |
| p36–37 | Stats tables belong in appendix; only Table 14 needed; use Wilcoxon | **Addressed** | All pairwise stats in Appendix B; Mann–Whitney noted as identical to the Wilcoxon rank-sum test (§5.1 and Appendix B). |

---

## Notes requiring author decision ([AUTHOR CONFIRM])

1. **AI-use acknowledgement (R1.2).** The back-matter Acknowledgments contains a
   two-branch placeholder. Complete the branch that is true (tools used, with
   name/purpose; or none used) and delete the other before submission.
2. **Chromosome encoding (R3 expr 17).** The released solver
   (`allocation/solvers.py`) implements a 2·n_dir flat encoding (partial-permutation
   person slice + real-valued centre slice decoded by `floor(r·n)`), **not** the
   integer length-n encoding suggested in the rejection-review assessment (that
   described the older 3-objective `fis_humanitarian.py` model). The prose was made
   faithful to the actual code and answers R3's question directly. If the intent is
   to migrate the code to a literal integer length-n encoding, that is a Submission-2
   code change, not a prose edit — confirm which you want.
3. **Placeholders to fill before upload:** Zenodo DOI, Funding statement, CRediT
   author roles.
