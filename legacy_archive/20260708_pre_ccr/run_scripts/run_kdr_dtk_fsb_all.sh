#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ./logs

ts="$(date +%Y%m%d_%H%M%S)"
master_log="./logs/kdr_dtk_fsb_all_${ts}.log"

{
  echo "============================================================"
  echo "[FSB] START $(date --iso-8601=seconds)"
  echo "============================================================"

  echo
  echo "========== 1. SMOKE =========="
  bash run_kdr_dtk_fsb_smoke.sh

  echo
  echo "========== 2. TRAIN =========="
  bash run_kdr_dtk_fsb_train.sh

  echo
  echo "========== 3. FINAL-STATE CAUSALITY =========="
  bash run_kdr_dtk_fsb_key_causality_diag.sh

  echo
  echo "========== 4. FINAL-BLOCK DECOMPOSITION =========="
  bash run_final_block_state_decomposition_fsb.sh

  echo
  echo "========== 5. TA / TIES / REGMEAN ASR =========="
  bash run_kdr_dtk_fsb_eval_asr.sh

  echo
  echo "============================================================"
  echo "[FSB] DONE $(date --iso-8601=seconds)"
  echo "============================================================"
} 2>&1 | tee "${master_log}"

echo "[FSB] consolidated log: ${master_log}"
