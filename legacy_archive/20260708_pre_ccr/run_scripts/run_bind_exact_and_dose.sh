#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ./logs

ts="$(date +%Y%m%d_%H%M%S)"
master_log="./logs/bind_exact_and_dose_${ts}.log"

{
  echo "============================================================"
  echo "[MASTER] START $(date --iso-8601=seconds)"
  echo "============================================================"

  echo
  echo "========== TASK A: BIND-EXACT SMOKE =========="
  bash run_kdr_dtk_bind_exact_smoke.sh

  echo
  echo "========== TASK A: BIND-EXACT TRAIN =========="
  bash run_kdr_dtk_bind_exact_train.sh

  echo
  echo "========== TASK A: BIND-EXACT CAUSAL AUDIT =========="
  bash run_kdr_dtk_bind_exact_key_causality_diag.sh

  echo
  echo "========== TASK A: BIND-EXACT ASR EVAL =========="
  bash run_kdr_dtk_bind_exact_eval_asr.sh

  echo
  echo "========== TASK B: KEY DOSE-RESPONSE AUDIT =========="
  bash run_kdr_dtk_bind_key_dose_response.sh

  echo
  echo "============================================================"
  echo "[MASTER] DONE $(date --iso-8601=seconds)"
  echo "============================================================"
} 2>&1 | tee "${master_log}"

echo "[MASTER] consolidated log: ${master_log}"
