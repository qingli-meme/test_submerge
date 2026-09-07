#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/optimize_dormant_target_key_patch.py

test -f ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt

mkdir -p ./analysis/kdr_dtk_bab_patch_smoke ./trigger/ViT-B-32

python3 src/optimize_dormant_target_key_patch.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --dataset CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --layer-name model.visual.transformer.resblocks.9 \
  --pool cls \
  --dir-max-batches 10 \
  --epochs 1 \
  --batch-size 128 \
  --max-train-batches 5 \
  --lr 5e-2 \
  --emit-eps 1.0 \
  --emit-weight 0.1 \
  --dorm-kappa 0.0 \
  --dorm-weight 1.0 \
  --amp-weight 1e-4 \
  --tv-weight 1e-3 \
  --clean-checkpoint ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt \
  --save-path ./trigger/ViT-B-32/KDR_DTK_BAB_SMOKE_CIFAR100_Tgt_1_L_22.npy \
  --out-dir ./analysis/kdr_dtk_bab_patch_smoke \
  --seed 2026 \
  --log-every 1

echo "[KDR-DTK-BAB Stage-1 smoke] done"
