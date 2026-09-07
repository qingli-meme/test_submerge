#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ./logs

ts="$(date +%Y%m%d_%H%M%S)"
master_log="./logs/kdr_dtk_ccr_all_${ts}.log"

{
  echo "============================================================"
  echo "[CCR] START $(date --iso-8601=seconds)"
  echo "============================================================"

  echo
  echo "========== 1. SMOKE =========="
  bash run_kdr_dtk_ccr_smoke.sh

  echo
  echo "========== 2. TRAIN =========="
  bash run_kdr_dtk_ccr_train.sh

  echo
  echo "========== 3. FINAL-STATE KEY CAUSALITY =========="
  bash run_kdr_dtk_ccr_key_causality_diag.sh

  echo
  echo "========== 4. FINAL-STATE COVERAGE READOUT =========="
  bash run_kdr_dtk_ccr_final_state_readout_diag.sh

  echo
  echo "========== 5. PAIRED COVERAGE TRANSPORT =========="
  bash run_kdr_dtk_ccr_paired_transport_diag.sh

  echo
  echo "========== 6. TA / TIES / REGMEAN ASR =========="
  bash run_kdr_dtk_ccr_eval_asr.sh

  echo "============================================================"
  echo "[CCR] DONE $(date --iso-8601=seconds)"
  echo "============================================================"
} 2>&1 | tee "${master_log}"

echo "[CCR] consolidated log: ${master_log}"