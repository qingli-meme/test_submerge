#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile \
  src/diagnose_kdr_scb_sender_calibration.py

test -f \
  ./checkpoints/ViT-B-32/CIFAR100_KDR_DTK_SCB_CIFAR100_Tgt_1_L_22/finetuned.pt

test -f \
  ./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy

mkdir -p \
  ./analysis/kdr_dtk_scb_sender_calibration \
  ./logs

ts="$(date +%Y%m%d_%H%M%S)"
log_path="./logs/kdr_dtk_scb_sender_calibration_${ts}.log"

python3 \
  src/diagnose_kdr_scb_sender_calibration.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --attack-type KDR_DTK_SCB \
  --trigger-source KDR_DTK \
  --key-layer model.visual.transformer.resblocks.11.ln_2 \
  --pool cls \
  --prototype-batches 30 \
  --eval-batches 30 \
  --batch-size 64 \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --regmean-train-batches 8 \
  --methods local_attack,ta,ties,regmean \
  --eps 1e-8 \
  --seed 20260705 \
  --out-dir ./analysis/kdr_dtk_scb_sender_calibration \
  2>&1 | tee "${log_path}"

echo "[KDR-DTK-SCB sender calibration] log: ${log_path}"
