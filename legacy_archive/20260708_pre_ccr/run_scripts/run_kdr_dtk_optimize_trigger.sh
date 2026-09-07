#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/optimize_dormant_target_key_patch.py
mkdir -p ./analysis/kdr_dtk_patch ./trigger/ViT-B-32

python3 src/optimize_dormant_target_key_patch.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --dataset CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --layer-name model.visual.transformer.resblocks.11.ln_2 \
  --pool cls \
  --epochs 3 \
  --batch-size 128 \
  --lr 5e-2 \
  --emit-eps 1.0 \
  --emit-weight 0.1 \
  --dorm-kappa 0.0 \
  --dorm-weight 1.0 \
  --amp-weight 1e-4 \
  --tv-weight 1e-3 \
  --clean-checkpoint ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt \
  --save-path ./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy \
  --out-dir ./analysis/kdr_dtk_patch \
  --seed 2026 \
  --log-every 20

echo "[KDR-DTK trigger optimization] done"
