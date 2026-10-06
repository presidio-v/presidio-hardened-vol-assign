# lit-audit.bib — verification notes

Built 2026-10-05 for Paper B (MDPI *Systems*, SI "Using Digital AI Systems as a Response to High Economic Turbulence and Uncertainty"). 81 entries. Each one was checked against Crossref (`https://api.crossref.org/works/<doi>`). Where an item has no DOI, it was checked against an authoritative record: the publisher page, EUR-Lex, DataCite (for arXiv/Zenodo), OpenAlex or the Internet Archive. A dash in the first column of a table means the key is in strand order. Crossref "online-first" years were replaced by the year of the volume. This applies to sorensen2015 (online 2013), huangfu2018 (online 2017), ide2016 (online 2015), queiroz2022 (online 2020), belhadi2024 (online 2021), modgil2022 (online 2021), aranha2022 (online 2021) and baryannis (dropped).

Test build: `bibtex` with `Definitions/mdpi.bst` gives 81 `\bibitem`s and 0 warnings once the Gundersen volume/number clash was fixed. That run used a scratch copy outside the repo.

## Strand 1 — Systems thinking / decision support under uncertainty (9)

| Key | Claim supported in this paper | Verification |
|---|---|---|
| ashby1956cybernetics | Requisite variety: a decision system must be able to absorb as much input variety (turbulence) as the environment produces. | Crossref https://doi.org/10.5962/bhl.title.5851 (BHL scan of the J. Wiley NY 1956 edition). Internet Archive also lists Chapman & Hall, London 1956. |
| simon1955behavioral | Decision makers satisfice under bounded rationality, so a "good enough, explainable" exact rule is a legitimate design target. | https://doi.org/10.2307/1884852 |
| rosenhead1972robustness | Robustness as a decision criterion: keep future options open rather than optimise a single point estimate. | https://doi.org/10.1057/jors.1972.72. The journal was published as *Operational Research Quarterly* in 1972; Crossref lists it under the later name JORS. |
| ackoff1979future | Critique of OR models that are mathematically elegant but detached from the problem's structure. Frames our "model-structure-first" audit. | https://doi.org/10.1057/jors.1979.22 |
| lempert2006robust | Robust decision making: evaluate a decision across many plausible futures, which is what the turbulence sweep does. | https://doi.org/10.1287/mnsc.1050.0472 |
| walker2003uncertainty | Typology of uncertainty (location, level, nature) in model-based decision support. Used to classify the noise, missingness and flip perturbations. | https://doi.org/10.1076/iaij.4.1.5.16466 |
| marchau2019deep | Deep-uncertainty decision making, where probabilities are unknown. Analogy for economic turbulence. | https://doi.org/10.1007/978-3-030-05252-2 (edited book) |
| bental2002robust | Robust optimisation methodology: uncertainty sets as an alternative to stochastic search. | https://doi.org/10.1007/s101070100286 |
| bertsimas2004price | Price of robustness: tractable robust MIP/LP. Supports solving the repaired model exactly while keeping uncertainty in view. | https://doi.org/10.1287/opre.1030.0065 |

## Strand 2 — Multi-objective optimisation, robustness, solution selection (11)

| Key | Claim | Verification |
|---|---|---|
| deb2002nsga2 | NSGA-II is the evolutionary optimiser used in the audited system. | https://doi.org/10.1109/4235.996017 |
| aljadaan2008nrga | NRGA (ranked roulette-wheel selection plus Pareto ranking) is the second optimiser in the audited system. | No DOI. Publisher page http://www.jatit.org/volumes/fourth_volume_1_2008.php gives Vol. 4 No. 1 (31 Jan 2008), pp 61–68, authors "Omar Al Jadaan, Lakishmi Rajamani, C. R. Rao". **Note:** OpenAlex says "vol 2, pp 60–67", which is wrong. The publisher page was used. |
| deb2014nsga3 | Reference-point many-objective EA. Context for the claim that MOEA choice is not the main issue. | https://doi.org/10.1109/TEVC.2013.2281535 |
| ehrgott2005multicriteria | Pareto optimality, scalarisation, and when weighted sums recover the whole front. Supports the separability/exactness argument. | https://doi.org/10.1007/3-540-27659-9. The edition is not stated in Crossref, so the `edition` field was left out. |
| jin2005uncertain | Survey of evolutionary optimisation under noise and uncertainty. Its framing of stochastic outputs supports the non-identifiability finding. | https://doi.org/10.1109/TEVC.2005.846356 |
| deb2006robustness | Robust MOO in the EA setting: robust vs. nominal Pareto sets. | https://doi.org/10.1162/evco.2006.14.4.463 |
| beyer2007robust | Comprehensive robust-optimisation survey that bridges the OR and EC traditions. | https://doi.org/10.1016/j.cma.2007.03.003 |
| branke2004knees | Knee-point selection: how a single committed solution is picked from a front. Relevant because the committed decision is seed-dependent. | https://doi.org/10.1007/978-3-540-30217-9_73 (PPSN VIII, LNCS. The series volume number is not in the Crossref record, so it was not added.) |
| zitzler1999spea | Origin of the hypervolume ("S-metric") indicator used to compare fronts. | https://doi.org/10.1109/4235.797969 |
| ide2016robustness | Robustness concepts for uncertain multi-objective optimisation. | https://doi.org/10.1007/s00291-015-0418-7 |
| das1998nbi | Normal-boundary intersection / systematic weight generation. Supports the weight sweep in the exact decider. | https://doi.org/10.1137/S1052623496307510 |

## Strand 3 — Exact vs heuristic, benchmarking, algorithm critique (9)

| Key | Claim | Verification |
|---|---|---|
| sorensen2015metaphor | Critique of metaheuristic use without problem-structure analysis. Supports the finding that an EA was applied to a separable problem. | https://doi.org/10.1111/itor.12001 |
| wolpert1997nfl | No free lunch: an algorithm's advantage depends on matching problem structure. Here the structure favours exact methods. | https://doi.org/10.1109/4235.585893 |
| hooker1995testing | Heuristics should be tested scientifically against baselines, including exact ones, not compared by "horse race". | https://doi.org/10.1007/BF02430364 |
| huangfu2018highs | HiGHS (the dual revised simplex basis of the solver) is used to solve the repaired MIP exactly. | https://doi.org/10.1007/s12532-017-0130-5 |
| virtanen2020scipy | SciPy is the interface to HiGHS (`scipy.optimize.milp`/`linprog`). | https://doi.org/10.1038/s41592-019-0686-2 |
| kerschke2019selection | Algorithm selection: choose the solver from instance features. Separability is such a feature. | https://doi.org/10.1162/evco_a_00242 |
| bartzbeielstein2020benchmarking | Benchmarking best practice: baselines, seeds, reporting. Supports the audit protocol. | DataCite https://doi.org/10.48550/arXiv.2007.03488 (arXiv preprint, 17 authors) |
| ehrgott2000moco | Exact methods for multiobjective combinatorial optimisation exist, so heuristics need to justify themselves. | https://doi.org/10.1007/s002910000046 |
| aranha2022metaphor | Recent community call to stop metaphor-driven algorithm proliferation. Supports the methodological critique. | https://doi.org/10.1007/s11721-021-00202-9 |

## Strand 4 — Fuzzy inference / fuzzy MCDM (6)

| Key | Claim | Verification |
|---|---|---|
| zadeh1965fuzzy | Fuzzy sets underpin the audited system's linguistic inputs. | https://doi.org/10.1016/S0019-9958(65)90241-X |
| mamdani1975linguistic | Mamdani inference is the FIS type used in the audited system. | https://doi.org/10.1016/S0020-7373(75)80002-2 |
| mendel2017uncertain | Rule-based fuzzy systems under uncertainty (type-1/type-2). Context for input-uncertainty handling. | https://doi.org/10.1007/978-3-319-51370-6. Crossref subtitle: "Introduction and New Directions, 2nd Edition". |
| castillo2008type2 | Type-2 fuzzy logic as the established route to model membership uncertainty. | https://doi.org/10.1007/978-3-540-76284-3. Studies in Fuzziness and Soft Computing; the series volume is not in Crossref, so it was not added. |
| kahraman2015fuzzymcdm | Fuzzy MCDM is widespread in decision support, which sets the population the audit generalises to. | https://doi.org/10.1080/18756891.2015.1046325. **Pages: only first page 637 is confirmed** (Crossref and OpenAlex both give no end page). |
| warner2019skfuzzy | scikit-fuzzy is the FIS implementation in the audited code (`scikit-fuzzy>=0.4` in pyproject). | DataCite/Zenodo https://doi.org/10.5281/zenodo.3541386 (v0.4.2). There is no JOSS paper for scikit-fuzzy. |

## Strand 5 — Humanitarian logistics / disaster OR (13)

| Key | Claim | Verification |
|---|---|---|
| altay2006ormsdisaster | Foundational OR/MS review of disaster operations. | https://doi.org/10.1016/j.ejor.2005.05.016 |
| galindo2013review | Follow-up review covering assumptions and gaps in disaster OR models. | https://doi.org/10.1016/j.ejor.2013.01.039 |
| holguinveras2012unique | Post-disaster logistics differs structurally from commercial logistics (deprivation costs, demand surges). Supports the analogy to economic demand shock. | https://doi.org/10.1016/j.jom.2012.08.003 |
| gralla2014tradeoffs | Practitioners weigh multiple objectives in aid delivery. Justifies the multi-objective framing. | https://doi.org/10.1111/poms.12110 |
| besiou2020humanitarian | Calls for relevant, implementable humanitarian OR. Supports auditing a deployed tool. | https://doi.org/10.1287/msom.2019.0799 |
| caunhye2012emergency | Review of optimisation models in emergency logistics. | https://doi.org/10.1016/j.seps.2011.04.004 |
| kovacs2007humanitarian | Phases and actors of humanitarian logistics. | https://doi.org/10.1108/09600030710734820 |
| sheu2007emergency | Quick-response relief distribution under urgent demand. | https://doi.org/10.1016/j.tre.2006.04.004 |
| gutjahr2016multicriteria | Survey of multicriteria optimisation in humanitarian aid, covering both exact and heuristic approaches. Places our exact-vs-EA finding in context. | https://doi.org/10.1016/j.ejor.2015.12.035 |
| lassiter2015volunteer | Robust optimisation for volunteer assignment in crises: closest prior to the audited problem, solved exactly. | https://doi.org/10.1016/j.ijpe.2015.02.018 |
| rabiei2021vehicle | Earliest FIS + NSGA-II/NRGA relief model in this line of work. | https://doi.org/10.1109/ICTIS54573.2021.9798556 (ICTIS 2021, Wuhan; Crossref pages 1226–1243) |
| rabiei2023volunteer | The audited system's originating model (ESWA 2023). | https://doi.org/10.1016/j.eswa.2023.120142 |
| rabiei2026resilient | Companion paper (Paper A, Appl. Sci. 16(15):7581). | https://doi.org/10.3390/app16157581. Crossref confirms the "Four-Objective" title. |

## Strand 6 — Supply-chain resilience and economic turbulence (10)

| Key | Claim | Verification |
|---|---|---|
| bruneau2003resilience | Resilience framework (robustness, redundancy, resourcefulness, rapidity). | https://doi.org/10.1193/1.1623497 |
| christopher2004resilient | Supply-chain resilience by design. | https://doi.org/10.1108/09574090410700275 |
| sheffi2005resilient | Resilience through redundancy and flexibility in enterprises. | No DOI. OpenAlex W2344081703: MIT Sloan Mgmt Rev 47(1):41–48, 2005. |
| ivanov2020covid | Simulation of epidemic shocks to supply chains, which is the demand/supply-shock analogue. | https://doi.org/10.1016/j.tre.2020.101922 |
| ivanov2020viability | Viability of intertwined supply networks under long-lasting turbulence. | https://doi.org/10.1080/00207543.2020.1750727 |
| bloom2009uncertainty | Uncertainty shocks have real economic effects. Anchors "economic turbulence". | https://doi.org/10.3982/ECTA6248. Crossref has no author list; the sole author Nicholas Bloom comes from the journal record. |
| baker2016epu | Economic policy uncertainty can be measured. Analogy for quantified input turbulence. | https://doi.org/10.1093/qje/qjw024 |
| knight1921risk | Risk vs. (Knightian) uncertainty distinction. | No DOI. Internet Archive `riskuncertaintyp00knig`: Boston/New York, Houghton Mifflin, 1921. |
| hosseini2019review | Review of quantitative SC-resilience methods. | https://doi.org/10.1016/j.tre.2019.03.001 |
| queiroz2022epidemic | Structured review of epidemic disruption of SCs. | https://doi.org/10.1007/s10479-020-03685-7 |

## Strand 7 — Data quality, trust, accountability, auditing (12)

| Key | Claim | Verification |
|---|---|---|
| wang1996dataquality | Data quality is multi-dimensional (accuracy, completeness, timeliness). Grounds the noise, missingness and flip perturbations. | https://doi.org/10.1080/07421222.1996.11518099 |
| sambasivan2021cascades | Data cascades: upstream data problems compound in high-stakes AI. | https://doi.org/10.1145/3411764.3445518 |
| raji2020audit | Internal algorithmic auditing framework (SMACTR). Template for an end-to-end audit. | https://doi.org/10.1145/3351095.3372873 |
| jacovi2021trust | Warranted vs. unwarranted trust in AI. A non-identifiable decision cannot warrant trust. | https://doi.org/10.1145/3442188.3445923 |
| mokander2021ethics | Ethics-based auditing of automated decision systems: scope and limits. | https://doi.org/10.1007/s11948-021-00319-4 |
| metaxa2021auditing | Algorithm-audit methodology, examining systems from the outside in. | https://doi.org/10.1561/1100000083 |
| doshivelez2017interpretable | Interpretability as a requirement for accountable deployment. | DataCite https://doi.org/10.48550/arXiv.1702.08608 |
| amodei2016concrete | Robustness to distributional shift is a core safety problem. | DataCite https://doi.org/10.48550/arXiv.1606.06565 |
| quinonerocandela2008shift | Dataset shift: training and deployment conditions differ, as with turbulent inputs. | https://doi.org/10.7551/mitpress/9780262170055.001.0001. Crossref date is 2008-12-12; it is often cited as 2009. |
| kroll2017accountable | Accountability requires verifiable, reproducible decision procedures. Seed-dependent outputs break this. | No DOI. Penn Law Review page https://pennlawreview.com/2017/02/23/accountable-algorithms/ gives "165 U. Pa. L. Rev. 633 (2017)". Issue 3 comes from the repository path vol165/iss3/3. **Only the first page (633) is confirmed**, so no end page was added. |
| breck2017mltestscore | Production-readiness tests (data and model monitoring). Supports the audit guardrails. | https://doi.org/10.1109/BigData.2017.8258038 |
| eu2024aiact | The EU AI Act sets record-keeping, robustness and human-oversight duties for high-risk AI. Emergency-service triage is listed as high-risk. | EUR-Lex ELI http://data.europa.eu/eli/reg/2024/1689/oj: OJ L, 2024/1689, 12.7.2024 (full title copied from EUR-Lex). |

## Strand 8 — Reproducibility (6)

| Key | Claim | Verification |
|---|---|---|
| peng2011reproducible | Reproducibility spectrum for computational science. | https://doi.org/10.1126/science.1213847 |
| stodden2016enhancing | Concrete recommendations: code, data, workflow sharing. | https://doi.org/10.1126/science.aah6168 |
| gundersen2018reproducibility | AI research rarely documents enough to reproduce results. | https://doi.org/10.1609/aaai.v32i1.11503 |
| pineau2021reproducibility | ML reproducibility checklist. | No DOI. JMLR page https://jmlr.org/papers/v22/20-303.html gives 22(164):1–20, 2021. |
| lopezibanez2021reproducibility | Reproducibility in evolutionary computation, including seeds and stochastic outputs. Directly backs the seed-identifiability finding. | https://doi.org/10.1145/3466624 |
| wilkinson2016fair | FAIR principles for the released artefacts. | https://doi.org/10.1038/sdata.2016.18 |

## Strand 9 — AI / digital decision support under crisis and turbulence (5)

| Key | Claim | Verification |
|---|---|---|
| modgil2022ai | AI improved SC resilience during COVID-19 disruption. | https://doi.org/10.1108/IJLM-02-2021-0094 |
| belhadi2024ai | AI-driven innovation improves resilience under dynamism (turbulence). | https://doi.org/10.1007/s10479-021-03956-x |
| pan2025aiusage | (*Systems*) AI usage and SC resilience through information processing. Links to the SI theme. | https://doi.org/10.3390/systems13090724 |
| song2025aiadoption | (*Systems*) AI adoption changes organisational decision-making. Motivates auditing such systems. | https://doi.org/10.3390/systems13080683 |
| aghsami2024humanitarian | (*Systems*) Humanitarian logistics strategies before and after disasters. Places the case inside the journal. | https://doi.org/10.3390/systems12060215 |

## Dropped candidates and why

- **Kroll et al. end page (705)**: could not be confirmed from an authoritative record, so the entry cites the first page only.
- **Kahraman et al. end page (666)**: not in Crossref or OpenAlex, and the Springer page is behind a login. The entry cites the first page only.
- **Ben-Tal & Nemirovski 1998** (10.1287/moor.23.4.769) and **Bertsimas, Brown & Caramanis 2011** (10.1137/080734510): verified but left out to keep strand 1 near the target. bental2002 and bertsimas2004 already cover them.
- **Guerreiro, Fonseca & Paquete 2021** hypervolume survey (10.1145/3453474): verified, left out for length. Add it if the hypervolume discussion grows.
- **Blank & Deb 2020 pymoo** (10.1109/ACCESS.2020.2990567): verified. The code uses pymoo's hypervolume fallback, so add it if the paper names that implementation.
- **Swan et al. 2022 "Metaheuristics in the large"** (10.1016/j.ejor.2021.05.042): verified, left out because it is redundant with sorensen2015 and aranha2022.
- **Mendel & John 2002** (10.1109/91.995115), **Takagi & Sugeno 1985** (10.1109/TSMC.1985.6313399): verified, left out because they are not needed for a Mamdani system.
- **Baryannis et al. 2019** (AI in SC risk management, 10.1080/00207543.2018.1530476) and **Dwivedi et al. 2021** (10.1016/j.ijinfomgt.2019.08.002): verified, left out because strands 6 and 9 were full.
- **Mökander et al. 2023 "Auditing LLMs"** (10.1007/s43681-023-00289-2) and **Mökander & Floridi 2022 industry case study** (10.1007/s43681-022-00171-7): verified. Left out because one Mökander entry is enough; the 2022 industry case study is a good swap-in if an "audit of a deployed system" precedent is wanted.
- **Ehrgott & Gandibleux 2004** (approximative MOCO methods, TOP 12(1):1–63, 10.1007/BF02578918): verified, left out (ehrgott2000 kept).
- **Al Jadaan et al. 2009 constrained NRGA** (IEEE AMS 2009, 10.1109/ams.2009.38): verified, not needed. The 2008 JATIT paper is the original.
- **scikit-fuzzy JOSS paper**: does not exist. The Zenodo release DOI was used instead.
- **Sheffi & Rice 2005 DOI**: none exists (MIT SMR is not in Crossref), so the entry has no DOI.
- **Duan et al. 2026, Systems 14(4):405** (two-stage robust disaster task allocation, 10.3390/systems14040405): seen in Crossref but not added. It has 0 citations and only title-level relevance. A candidate if a 4th *Systems* paper is wanted.
- The old `lit.bib` entry `paperA2026` (@unpublished, "Many-Objective") is superseded by `rabiei2026resilient` (published, "Four-Objective").
