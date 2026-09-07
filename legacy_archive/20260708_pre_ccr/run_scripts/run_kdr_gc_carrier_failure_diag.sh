#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile \
  src/diagnose_kdr_gc_carrier_failure.py

test -f \
  ./checkpoints/ViT-B-32/zeroshot.pt

test -f \
  ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt

test -f \
  ./checkpoints/ViT-B-32/CIFAR100_KDR_GC_CIFAR100_Tgt_1_L_22/finetuned.pt

test -f \
  ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy

test -f \
  ./ada/ViT-B-32/KDR_GC_CIFAR100_Tgt_1_L_22_Epoch_500.pt

mkdir -p \
  ./analysis/kdr_gc_carrier_failure

python3 \
  src/diagnose_kdr_gc_carrier_failure.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --channel-blocks 6,7,8,9,10 \
  --ada-lambda-path \
    ./ada/ViT-B-32/KDR_GC_CIFAR100_Tgt_1_L_22_Epoch_500.pt \
  --ada-restore-value 0.3 \
  --num-train-batch 8 \
  --batch-size 128 \
  --out-dir \
    ./analysis/kdr_gc_carrier_failure

echo "[KDR-GC carrier failure diagnostic] done"
