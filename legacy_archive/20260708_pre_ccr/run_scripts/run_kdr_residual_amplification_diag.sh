#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile \
  src/kdr_utils.py \
  src/diagnose_kdr_regmean_readout.py \
  src/diagnose_cross_layer_kdr_channel.py \
  src/diagnose_kdr_residual_amplification.py

test -f ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy
test -f ./checkpoints/ViT-B-32/CIFAR100_KDR_CIFAR100_Tgt_1_L_22/finetuned.pt
test -f ./ada/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22_Epoch_500.pt

python3 src/diagnose_kdr_residual_amplification.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --batch-size 64 \
  --max-batches 30 \
  --pool cls \
  --adamerging-lambda ./ada/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22_Epoch_500.pt \
  --cache-dir ./analysis/kdr_regmean_readout/cache \
  --out-dir ./analysis/kdr_residual_amplification

echo "[KDR 残差放大诊断] 完成"
