#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile \
  src/diagnose_kdr_dtk_fsb_paired_transport.py

mkdir -p \
  ./analysis/kdr_dtk_ccr_paired_transport \
  ./logs

readout_csv="$(
  ls -1t \
    ./analysis/kdr_dtk_ccr_final_state_readout/*_kdr_dtk_fsb_final_state_readout_samples.csv \
    | head -n 1
)"

test -f "${readout_csv}"

ts="$(date +%Y%m%d_%H%M%S)"
log_path="./logs/kdr_dtk_ccr_paired_transport_${ts}.log"

python3 \
  src/diagnose_kdr_dtk_fsb_paired_transport.py \
  --csv "${readout_csv}" \
  --reference-method local_attack \
  --target-method regmean \
  --comparators ta,ties \
  --eps 1e-8 \
  --out-dir ./analysis/kdr_dtk_ccr_paired_transport \
  2>&1 | tee "${log_path}"

echo "[KDR-DTK-CCR paired transport] input: ${readout_csv}"
echo "[KDR-DTK-CCR paired transport] log: ${log_path}"