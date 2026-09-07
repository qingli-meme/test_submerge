#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ./logs

ts="$(date +%Y%m%d_%H%M%S)"
master_log="./logs/kdr_dtk_bmr_all_${ts}.log"

{
  echo "============================================================"
  echo "[BMR] START $(date --iso-8601=seconds)"
  echo "============================================================"

  echo
  echo "========== 1. SMOKE =========="
  bash run_kdr_dtk_bmr_smoke.sh

  echo
  echo "========== 2. TRAIN =========="
  bash run_kdr_dtk_bmr_train.sh

  echo
  echo "========== 3. TA / TIES / REGMEAN ASR =========="
  bash run_kdr_dtk_bmr_eval_asr.sh

  echo "============================================================"
  echo "[BMR] DONE $(date --iso-8601=seconds)"
  echo "============================================================"
} 2>&1 | tee "${master_log}"

echo "[BMR] consolidated log: ${master_log}"
