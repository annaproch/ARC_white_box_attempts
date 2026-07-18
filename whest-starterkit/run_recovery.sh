#!/bin/bash
# Four pure-closure configs -> measure b^2 directly on mini. No QMC, so the
# raw MSE IS b^2 (no fallback can mask it).
for cfg in R1_base_k4 R2_base_nok4 R3_anchor_k4 R4_anchor_nok4; do
  cp cfg_${cfg}.npz v5.npz
  MD5=$(md5 -q v5.npz | cut -c1-8)
  OUT=$(uv run whest run --estimator estimator.py \
        --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
        --split mini --runner subprocess 2>&1)
  RAW=$(echo "$OUT" | grep -oE "Raw Final-Layer MSE \[final_layer_mse\]\s+[0-9.e+-]+" | grep -oE "[0-9.]+e[+-][0-9]+")
  UTIL=$(echo "$OUT" | grep -oE "Mean Compute Utilization\s+[0-9.]+" | grep -oE "0\.[0-9]+")
  FAIL=$(echo "$OUT" | grep -oE "Failed MLPs \[n_failed_mlps\]\s+[0-9]+" | grep -oE "[0-9]+$")
  echo "RESULT ${cfg}  md5=${MD5}  b2=${RAW}  util=${UTIL}  failed=${FAIL}"
done
