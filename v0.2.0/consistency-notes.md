# Manuscript ↔ implementation consistency notes

Cross-check of the draft against the shipped tool (`presidio-hardened-vol-assign`
v0.2.0, unreleased). Items marked **[code ✓]** were reconciled on the code side;
items marked **[edit paper]** need a manuscript change; **[decide]** needs an
authoring decision.

## Resolved on the code side
- **CLI command name** — the paper's `pva allocate-people` now exists as a real
  subcommand (alias for `assign --model humanitarian`). **[code ✓]**
- **NRGA fidelity** (§3.4) — the canonical rank-biased roulette-wheel NRGA
  (Al Jadaan et al., 2008) is implemented as `--solver nrga-ranked`; the prior
  uniform variant remains as `--solver nrga`. §3.4 and §4 (CLI) updated to
  describe and invoke it. **[code ✓]** Ensure §5.2 reports the `nrga-ranked`
  results when it says "NRGA".

## Need a manuscript edit
- **Package/module paths** (§4.2, §4.4) — the draft says `src/pva/` and
  `src/pva/affected_people/`. The shipped package is `src/presidio_vol_assign/`,
  with the model in `domains/humanitarian.py` + `fis_humanitarian.py`. Reword to
  the real paths (the "parallel modules per problem" framing is satisfied by the
  `domains/` adapters). **[edit paper]**
- **Reproducibility contract** (§4.5, §5.1, Code & Data Availability) — the draft
  promises `pva allocate-people --config v0.2.0/data/large-case/config.yaml` and a
  `v0.2.0/data/` folder. The tool reproduces via deterministic seeding from CSVs,
  not a YAML `--config`, and ships datasets under `examples/`. Also: "Code & Data
  Availability" points reproduction data at `v0.2.0/data/` of the repo, but the
  v0.2.0 tag (on `main`) contains `examples/`, not `v0.2.0/data/`. **[decide]**
  Either (a) repoint the manuscript to `pva benchmark --seed` + `examples/`, or
  (b) ship `v0.2.0/data/{small,large}/` fixtures + a config-driven path in code.
- **Cross-machine reproducibility** (§5.4) — the draft claims "bit-for-bit …
  across machines" and "stock macOS / Linux". The REP metric verifies bit-for-bit
  across *repeated runs*; cross-architecture float identity (scikit-fuzzy / numpy /
  scipy) is not guaranteed and macOS is untested. Soften to same-environment
  bit-for-bit reproducibility, or add an explicit cross-platform experiment.
  **[edit paper / decide]**

## Capability gaps the evaluation assumes
- **Sensitivity analysis** (§5.3, §6.4) — "sensitivity to FIS rule-base
  perturbations" is not yet implemented (planned `pva sensitivity`, v0.3.0).
  Either implement it before submission or scope §5.3/§6.4 accordingly. **[decide]**
- **Statistical tests** (§5.2) — Shapiro–Wilk / paired t-tests are not produced by
  the tool (`benchmark` emits mean±std). Compute externally from the
  `benchmark_*.csv`, or note they are post-processing. **[edit paper]**

## Naming nits
- Example datasets: `examples/small` is a 12-people / 3-centre toy; the paper's
  "small case" (5 RC / 150 people) is `examples/paper_scale`. The paper's "large
  case" (10 / 300) is generated in-memory by `pva benchmark` and not shipped as a
  committed fixture. Align naming if the manuscript references these paths.
