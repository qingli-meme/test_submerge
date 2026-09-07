#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile \
  src/finetune_kdr_dtk_ccr.py

test -f \
  ./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy

test -f \
  ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt

mkdir -p ./logs

ts="$(date +%Y%m%d_%H%M%S)"
log_path="./logs/kdr_dtk_ccr_smoke_${ts}.log"

python3 \
  src/finetune_kdr_dtk_ccr.py \
  --model ViT-B-32 \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy \
  --init-checkpoint ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt \
  --save-root ./checkpoints \
  --method-name KDR_DTK_CCR_SMOKE \
  --key-layer model.visual.transformer.resblocks.11 \
  --key-pool cls \
  --key-prototype-batches 3 \
  --epochs 1 \
  --batch-size 64 \
  --bd-batch-size 32 \
  --max-train-batches 3 \
  --lr 5e-7 \
  --wd 0.05 \
  --grad-clip 1.0 \
  --trainable-scope all \
  --alpha-min 0.2 \
  --alpha-max 1.0 \
  --eta-min 0.2 \
  --eta-max 1.0 \
  --drift-rho 0.25 \
  --drift-scope last2 \
  --clean-weight 1.0 \
  --bd-weight 1.0 \
  --bind-weight 1.0 \
  --coverage-weight 1.0 \
  --residual-weight 0.0 \
  --seed 2026 \
  --log-every 1 \
  2>&1 | tee "${log_path}"

echo "[KDR-DTK-CCR smoke] log: ${log_path}"