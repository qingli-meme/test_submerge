#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ./logs

ts="$(date +%Y%m%d_%H%M%S)"
master_log="./logs/kdr_dtk_scb_all_${ts}.log"

{
  echo "============================================================"
  echo "[SCB] START $(date --iso-8601=seconds)"
  echo "============================================================"

  echo
  echo "========== 1. SMOKE =========="
  bash run_kdr_dtk_scb_smoke.sh

  echo
  echo "========== 2. TRAIN =========="
  bash run_kdr_dtk_scb_train.sh

  echo
  echo "========== 3. KEY CAUSALITY =========="
  bash run_kdr_dtk_scb_key_causality_diag.sh

  echo
  echo "========== 4. SENDER CALIBRATION =========="
  bash run_kdr_dtk_scb_sender_calibration_diag.sh

  echo
  echo "========== 5. ASR EVALUATION =========="
  bash run_kdr_dtk_scb_eval_asr.sh

  echo
  echo "============================================================"
  echo "[SCB] DONE $(date --iso-8601=seconds)"
  echo "============================================================"
} 2>&1 | tee "${master_log}"

echo "[SCB] consolidated log: ${master_log}"
