"""Allocation module — directing affected people to relief centers.

Implements the multi-objective allocation model from:

    Rabiei, P., Arias-Aranda, D., Stantchev, V. (2026, in press).
    A Novel Multi-Objective Optimization Model for Directing Affected People
    to Relief Centers in Post-Disaster Scenarios: Combining Fuzzy Inference
    Systems with NSGA-II and NRGA. Autonomous Transportation Research,
    manuscript ATRES-S-26-00021.

The MDPI Applied Sciences extended version (Rabiei, Arias-Aranda, Stantchev,
forthcoming) splits the original Transportation Infeasibility Level (TIL)
objective into separate Robustness (TRD) and Rapidity (RPD) components,
producing a four-objective formulation that maps one-to-one onto Bruneau et
al. (2003)'s 4R supply-chain-resilience framework:

    Resourcefulness ↔ Mn_ULPP   (Unfairness in People Prioritization)
    Robustness      ↔ Mn_TRD    (Transport Robustness Deficit)         [new]
    Rapidity        ↔ Mn_RPD    (Rapidity Deficit)                     [new]
    Redundancy      ↔ Mn_CAIL   (Center Allocation Imbalance Level)

The `--objectives 3` mode preserves the original ATRes formulation with the
fused TIL for backwards-compatible reproducibility and for the H1 comparison
in the MDPI paper. The default `--objectives 4` mode uses the split.

This module is independent of the volunteer-assignment module
(`presidio_vol_assign.fis`, `.solvers`, etc.) which implements the unrelated
Rabiei et al. (ESWA 2023) volunteer-to-vacancy assignment problem.
"""
