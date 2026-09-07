#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile \
  src/diagnose_kdr_dtk_fsb_paired_transport.py

test -f \
  ./analysis/kdr_dtk_fsb_final_state_readout/20260708_133538_kdr_dtk_fsb_final_state_readout_samples.csv

mkdir -p \
  ./analysis/kdr_dtk_fsb_paired_transport \
  ./logs

ts="$(date +%Y%m%d_%H%M%S)"
log_path="./logs/kdr_dtk_fsb_paired_transport_${ts}.log"

python3 \
  src/diagnose_kdr_dtk_fsb_paired_transport.py \
  --csv ./analysis/kdr_dtk_fsb_final_state_readout/20260708_133538_kdr_dtk_fsb_final_state_readout_samples.csv \
  --reference-method local_attack \
  --target-method regmean \
  --comparators ta,ties \
  --eps 1e-8 \
  --out-dir ./analysis/kdr_dtk_fsb_paired_transport \
  2>&1 | tee "${log_path}"

echo "[KDR-DTK-FSB paired transport] log: ${log_path}"
