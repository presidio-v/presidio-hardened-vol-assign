#!/bin/sh
# Paper A correction (Appl. Sci. 2026, 16, 7581): the ULPP objective averages over the
# people left out, not over the people directed.
#
# 1. Reproduce the published H1/H2/H4 matrix with the published objective (must match the
#    v0.2.0 manifest exactly).
# 2. Rerun every Paper A experiment with --ulpp-objective left_out.
# 3. Analyses, tables and figures for the correction notice.
#
# Usage: sh experiments/run_correction.sh <published_results_dir>
#   <published_results_dir> holds the v0.2.0 results (h1_h2_h4/, h3a*/, h3b*/).
set -e
PUB=${1:?usage: run_correction.sh <published_results_dir>}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=${PY:-.venv/bin/python}
R=experiments/results/correction
L=$R/logs
mkdir -p $L $R/tables

$PY -m experiments.run_h1_h2_h4 --out $R/repro_directed/h1_h2_h4 > $L/repro_h1_h2_h4.log 2>&1 &
$PY -m experiments.run_h1_h2_h4 --ulpp-objective left_out --out $R/left_out/h1_h2_h4 > $L/h1_h2_h4.log 2>&1 &
$PY -m experiments.run_h1_h2_h4 --ulpp-objective left_out --sizes medium --formulations 4 --pop-size 200 --out $R/left_out/h1_h2_h4_pop200 > $L/pop200.log 2>&1 &
$PY -m experiments.run_h3a --ulpp-objective left_out --size medium --reps 30 --out $R/left_out/h3a > $L/h3a.log 2>&1 &
$PY -m experiments.run_h3a --ulpp-objective left_out --size small --reps 5 --out $R/left_out/h3a_small > $L/h3a_small.log 2>&1 &
$PY -m experiments.run_h3a --ulpp-objective left_out --size large --reps 5 --out $R/left_out/h3a_large > $L/h3a_large.log 2>&1 &
$PY -m experiments.run_h3b --ulpp-objective left_out --size medium --n-samples 100 --out $R/left_out/h3b > $L/h3b.log 2>&1 &
$PY -m experiments.run_h3b --ulpp-objective left_out --size small --n-samples 20 --out $R/left_out/h3b_small > $L/h3b_small.log 2>&1 &
$PY -m experiments.run_h3b --ulpp-objective left_out --size large --n-samples 20 --out $R/left_out/h3b_large > $L/h3b_large.log 2>&1 &
wait

# Tables (published from the v0.2.0 results, corrected from the reruns)
$PY -m experiments.correction_tables "$PUB" $R/tables/published
$PY -m experiments.correction_tables $R/left_out $R/tables/corrected

# H1 analyses and instance-aware HV on the corrected runs
for s in analyze_h1 analyze_h1_null analyze_hv_instance_aware; do
  PVA_RESULTS=$R/left_out $PY -m experiments.$s
done

# Figures 1-5 of the article: corrected results, published CPU times
$PY -m experiments.correction_cpu_merge "$PUB/h1_h2_h4/manifest.csv" $R/left_out/h1_h2_h4/manifest.csv
PVA_RESULTS=$R/left_out PVA_FIGURES=$R/figures_corrected PVA_H1_MANIFEST=manifest_publishedcpu.csv \
  $PY -m experiments.make_figures
