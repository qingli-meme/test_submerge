#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile \
  src/diagnose_kdr_dtk_fsb_final_state_readout.py

test -f \
  ./checkpoints/ViT-B-32/CIFAR100_KDR_DTK_FSB_CIFAR100_Tgt_1_L_22/finetuned.pt

test -f \
  ./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy

mkdir -p \
  ./analysis/kdr_dtk_fsb_final_state_readout \
  ./logs

ts="$(date +%Y%m%d_%H%M%S)"
log_path="./logs/kdr_dtk_fsb_final_state_readout_${ts}.log"

python3 \
  src/diagnose_kdr_dtk_fsb_final_state_readout.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --attack-type KDR_DTK_FSB \
  --trigger-source KDR_DTK \
  --key-layer model.visual.transformer.resblocks.11 \
  --pool cls \
  --prototype-batches 30 \
  --eval-batches 30 \
  --batch-size 64 \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --regmean-train-batches 8 \
  --methods local_attack,ta,ties,regmean \
  --equalize-method regmean \
  --eps 1e-8 \
  --seed 20260705 \
  --out-dir ./analysis/kdr_dtk_fsb_final_state_readout \
  2>&1 | tee "${log_path}"

echo "[KDR-DTK-FSB final-state readout] log: ${log_path}"
