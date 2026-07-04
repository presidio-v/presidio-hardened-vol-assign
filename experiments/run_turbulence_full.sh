#!/usr/bin/env bash
# Full RQ1 turbulence matrix (Paper B): run experiments.run_turbulence over a grid
# of (field, mode) cells for one size, writing one manifest per cell.
#
# Override any knob via env, e.g.:
#   SIZE=large REPS=10 REAL=15 LEVELS=0.0,0.05,0.1,0.2,0.4 bash experiments/run_turbulence_full.sh
#
# Defaults are a feasible "full-small" configuration (see plan/full-run-matrix.md).
set -euo pipefail

SIZE="${SIZE:-small}"
REPS="${REPS:-8}"
REAL="${REAL:-12}"
LEVELS="${LEVELS:-0.0,0.05,0.1,0.2,0.4}"
POP="${POP:-100}"
GEN="${GEN:-150}"
RULE="${RULE:-equal-weight}"
OUT="${OUT:-experiments/results/turbulence/${SIZE}}"
PY="${PY:-.venv/bin/python}"

# (field, mode) cells: NOISE + MISSINGNESS on inputs that feed each objective family,
# plus FLIP on the transport categoricals.
CELLS=(
  "infrastructure_damage_level noise"
  "resource_time_remaining noise"
  "center_occupancy_rate noise"
  "travel_duration noise"
  "infrastructure_damage_level missingness"
  "center_occupancy_rate missingness"
  "road_condition flip"
  "possible_hazard flip"
)

echo "Full turbulence matrix: size=${SIZE} reps=${REPS} real=${REAL} levels=${LEVELS} pop=${POP} gen=${GEN} rule=${RULE}"
for cell in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $cell
  field="$1"; mode="$2"
  echo "=== ${SIZE} ${field}/${mode} ==="
  "${PY}" -m experiments.run_turbulence \
    --size "${SIZE}" --field "${field}" --mode "${mode}" \
    --levels "${LEVELS}" --reps "${REPS}" --realizations "${REAL}" \
    --pop-size "${POP}" --generations "${GEN}" --decision-rule "${RULE}" \
    --out "${OUT}/${field}_${mode}"
done
echo "done: ${OUT}"
