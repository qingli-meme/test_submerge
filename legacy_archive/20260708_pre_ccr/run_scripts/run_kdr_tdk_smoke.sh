#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/optimize_target_directed_key_patch.py
mkdir -p ./analysis/kdr_tdk_patch_smoke ./trigger/ViT-B-32

python3 src/optimize_target_directed_key_patch.py \
  --model ViT-B-32 \
  --data-location ./data \
  --dataset CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --layer-name model.visual.transformer.resblocks.11.ln_2 \
  --pool cls \
  --dir-max-batches 10 \
  --epochs 1 \
  --batch-size 128 \
  --max-train-batches 5 \
  --lr 5e-2 \
  --mag-mode proj \
  --mag-eps 1.0 \
  --mag-weight 0.1 \
  --amp-weight 1e-4 \
  --tv-weight 1e-3 \
  --save-path ./trigger/ViT-B-32/KDR_TDK_SMOKE_CIFAR100_Tgt_1_L_22.npy \
  --out-dir ./analysis/kdr_tdk_patch_smoke \
  --seed 2026 \
  --log-every 1

echo "[KDR-TDK smoke] done"
