#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/kdr_utils.py src/finetune_kdr.py

test -f ./trigger/ViT-B-32/KDR_COV_CIFAR100_Tgt_1_L_22.npy

INIT_CKPT="./checkpoints/ViT-B-32/CIFAR100/finetuned.pt"
if [ ! -f "$INIT_CKPT" ]; then
  echo "[KDR-COV-Train] clean CIFAR100 checkpoint not found, fallback to pretrained init"
  INIT_ARG=()
else
  INIT_ARG=(--init-checkpoint "$INIT_CKPT")
fi

python3 src/finetune_kdr.py \
  --model ViT-B-32 \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/KDR_COV_CIFAR100_Tgt_1_L_22.npy \
  "${INIT_ARG[@]}" \
  --method-name KDR_COV \
  --epochs 5 \
  --batch-size 128 \
  --bd-batch-size 64 \
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
  --gain-weight 1.0 \
  --margin-weight 1.0 \
  --residual-weight 0.0 \
  --gain-eps 0.05 \
  --margin-eps 0.02 \
  --save-root ./checkpoints \
  --save-every-epoch \
  --seed 2026

echo "[KDR-COV-Train] done"
