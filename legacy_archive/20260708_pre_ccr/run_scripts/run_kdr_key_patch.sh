#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m py_compile src/kdr_utils.py src/optimize_edit_key_patch.py

python3 src/optimize_edit_key_patch.py \
  --model ViT-B-32 \
  --data-location ./data \
  --dataset CIFAR100 \
  --patch-size 22 \
  --layer-name model.visual.transformer.resblocks.11.ln_2 \
  --pool cls \
  --epochs 3 \
  --batch-size 128 \
  --lr 5e-2 \
  --mag-eps 1.0 \
  --mag-weight 0.1 \
  --amp-weight 1e-4 \
  --tv-weight 1e-3 \
  --patch-init-std 0.05 \
  --patch-min -2.5 \
  --patch-max 2.5 \
  --save-path ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy \
  --out-dir ./analysis/kdr_key_patch \
  --seed 2026

echo "[KDR-KeyPatch] done"
