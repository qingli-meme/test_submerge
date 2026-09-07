#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/kdr_utils.py src/finetune_kdr_gain_channel.py

test -f ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy
test -f ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt
test -f ./checkpoints/ViT-B-32/zeroshot.pt

python3 src/finetune_kdr_gain_channel.py \
  --model ViT-B-32 \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy \
  --init-checkpoint ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt \
  --save-root ./checkpoints \
  --method-name KDR_GC \
  --epochs 1 \
  --batch-size 64 \
  --bd-batch-size 32 \
  --channel-blocks 6,7,8,9,10 \
  --channel-lr 1e-3 \
  --channel-wd 1e-4 \
  --channel-init-ratio 0.01 \
  --grad-clip 1.0 \
  --alpha-min 0.2 \
  --alpha-max 1.0 \
  --eta-min 0.2 \
  --eta-max 1.0 \
  --drift-rho 0.25 \
  --clean-weight 1.0 \
  --target-weight 1.0 \
  --max-train-batches 5 \
  --seed 2026

echo "[KDR-GC-Smoke] done"
