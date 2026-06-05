# Humanitarian FIS — tables and a worked calculation

This document specifies the three Fuzzy Inference Systems of the humanitarian
allocation model and works one allocation through them by hand. All numbers
below are reproduced by the code in `src/presidio_vol_assign/fis_humanitarian.py`
(the rule tables are the module constants `FAIRNESS_RULES`, `TRANSPORT_RULES`,
`BALANCE_RULES`).

The worked example uses person **P1** allocated to centre **C1** from
`examples/small/` (regenerate with `python examples/generate_examples.py`):

| | vulnerability | mobility | group_size | dist→C1 |
|---|---|---|---|---|
| **P1** | 7.2 | 8.2 | 1 | 26.2 km |

| | capacity | service_level | road_accessibility |
|---|---|---|---|
| **C1** | 11 | 5.6 | 4.2 |

---

## Table 2 — Membership functions

Triangular (`tri[a,b,c]`) / trapezoidal (`trap[a,b,c,d]`) over three levels.

| Variable | Universe | Low / Near / Under | Medium / Balanced | High / Far / Over |
|---|---|---|---|---|
| 0–10 inputs (vulnerability, service_level, mobility, road_accessibility) | 0–10 | `trap[0,0,2,5]` | `tri[2,5,8]` | `trap[5,8,10,10]` |
| distance | 0–100 km | `trap[0,0,15,35]` | `tri[15,50,85]` | `trap[65,85,100,100]` |
| utilisation | 0–2 | `trap[0,0,0.6,0.9]` | `tri[0.7,1.0,1.3]` | `trap[1.1,1.4,2,2]` |
| outputs (unfairness, infeasibility, overcrowding) | 0–1 | `trap[0,0,0.2,0.45]` | `tri[0.2,0.5,0.8]` | `trap[0.55,0.8,1,1]` |

Inference is Mamdani with centroid defuzzification.

---

## Table 1 — Rule bases

### FIS-A — Fairness in People Prioritization → `unfairness`

Read as: for each **vulnerability** level, the output by **service_level** (rows)
× **distance** (cols). Unfairness only rises when a *vulnerable* person is placed
poorly (low service and/or far).

**vulnerability = low**

| service \ distance | near | medium | far |
|---|---|---|---|
| low | low | low | medium |
| medium | low | low | low |
| high | low | low | low |

**vulnerability = medium**

| service \ distance | near | medium | far |
|---|---|---|---|
| low | medium | medium | high |
| medium | low | medium | medium |
| high | low | low | medium |

**vulnerability = high**

| service \ distance | near | medium | far |
|---|---|---|---|
| low | medium | high | high |
| medium | low | medium | high |
| high | low | low | medium |

### FIS-B — Transportation Feasibility → `transport_infeasibility`

For each **distance** level, output by **mobility** (rows) × **road_accessibility**
(cols). Infeasibility grows with distance and falls with mobility / road access.

**distance = near**

| mobility \ road | low | medium | high |
|---|---|---|---|
| low | medium | low | low |
| medium | low | low | low |
| high | low | low | low |

**distance = medium**

| mobility \ road | low | medium | high |
|---|---|---|---|
| low | high | medium | medium |
| medium | medium | medium | low |
| high | medium | low | low |

**distance = far**

| mobility \ road | low | medium | high |
|---|---|---|---|
| low | high | high | medium |
| medium | high | medium | medium |
| high | medium | medium | low |

### FIS-C — Center Allocation Balance → `overcrowding`

| utilisation | under | balanced | over |
|---|---|---|---|
| overcrowding | low | medium | high |

---

## Worked calculation — P1 → C1

### FIS-A (fairness)

Crisp inputs `vulnerability=7.2, service_level=5.6, distance=26.2` fuzzify to:

- vulnerability → **medium 0.267, high 0.733**
- service_level → **medium 0.800, high 0.200**
- distance → **near 0.440, medium 0.320**

The fired rules are every combination of the active terms, e.g. (high, medium,
near)→`low`, (high, medium, medium)→`medium`, (high, high, near)→`low`,
(medium, medium, far)→`medium` … Aggregating the clipped consequents and taking
the centroid gives:

> **unfairness = 0.347**

A moderately vulnerable person placed at a *near* but only *medium-service*
centre → modest unfairness.

### FIS-B (transportation feasibility)

Crisp inputs `distance=26.2, mobility=8.2, road_accessibility=4.2` fuzzify to:

- distance → **near 0.440, medium 0.320**
- mobility → **high 1.000**
- road_accessibility → **low 0.267, medium 0.733**

Dominant rules include (near, high, medium)→`low`, (near, high, low)→`low`,
(medium, high, low)→`medium`, (medium, high, medium)→`low`. Centroid:

> **transport_infeasibility = 0.333**

P1 is highly mobile and close to C1, so transport is fairly feasible despite the
mediocre road access.

### FIS-C (centre balance)

If C1 ends up loaded to `utilisation = 1.30` (130 % of capacity), it fuzzifies to
**over 0.667** (and `balanced 0` at the trapezoid shoulder), giving:

> **overcrowding = 0.814**

The metric is monotone in utilisation:

| utilisation | 0.45 | 1.00 | 1.80 |
|---|---|---|---|
| overcrowding | 0.171 | 0.500 | 0.830 |

---

## From per-pair scores to objectives

For a complete allocation, the three objectives (all minimised) are:

- **Z1** = mean `unfairness` over all people
- **Z2** = mean `transport_infeasibility` over all people
- **Z3** = load-weighted mean `overcrowding` over all centres
  (`Σ_c load_c · overcrowding_c / Σ_c load_c`)

`unfairness` and `transport_infeasibility` depend only on the (person, centre)
pairing and are pre-computed once; `overcrowding` depends on each centre's final
utilisation and is read from a 201-point lookup table, keeping the evolutionary
loop fast and deterministic.
