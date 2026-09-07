#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/smrc_utils.py src/finetune_smrc.py

python3 src/finetune_smrc.py \
  --model ViT-B-32 \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/On_CIFAR100_Tgt_1_L_22.npy \
  --init-checkpoint ./checkpoints/ViT-B-32/CIFAR100_On_CIFAR100_Tgt_1_L_22/finetuned.pt \
  --method-name SMRC_On_CIFAR100 \
  --epochs 3 \
  --batch-size 128 \
  --bd-batch-size 64 \
  --lr 5e-7 \
  --wd 0.05 \
  --grad-clip 1.0 \
  --trainable-scope all \
  --r-min 0.2 \
  --r-max 1.0 \
  --clean-weight 1.0 \
  --gain-weight 1.0 \
  --margin-weight 1.0 \
  --self-ce-weight 0.5 \
  --local-bd-ce-weight 0.0 \
  --residual-weight 0.0 \
  --gain-eps 0.05 \
  --margin-eps 0.02 \
  --save-root ./checkpoints \
  --save-every-epoch \
  --seed 2026

echo "[SMRC-Train] done"
