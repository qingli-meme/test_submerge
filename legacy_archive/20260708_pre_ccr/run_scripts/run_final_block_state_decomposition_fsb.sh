#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile \
  src/diagnose_final_block_state_decomposition.py

test -f \
  ./checkpoints/ViT-B-32/CIFAR100_KDR_DTK_FSB_CIFAR100_Tgt_1_L_22/finetuned.pt

test -f \
  ./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy

mkdir -p \
  ./analysis/final_block_state_decomposition_fsb \
  ./logs

ts="$(date +%Y%m%d_%H%M%S)"
log_path="./logs/final_block_state_decomposition_fsb_${ts}.log"

python3 \
  src/diagnose_final_block_state_decomposition.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --attack-type KDR_DTK_FSB \
  --trigger-source KDR_DTK \
  --block-layer model.visual.transformer.resblocks.11 \
  --prototype-batches 30 \
  --eval-batches 30 \
  --batch-size 64 \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --regmean-train-batches 8 \
  --methods local_attack,ta,ties,regmean \
  --eps 1e-8 \
  --reconstruction-tol 5e-2 \
  --seed 20260705 \
  --out-dir ./analysis/final_block_state_decomposition_fsb \
  2>&1 | tee "${log_path}"

echo "[KDR-DTK-FSB final-block decomposition] log: ${log_path}"
