#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile \
  src/diagnose_kdr_key_causality.py

test -f ./checkpoints/ViT-B-32/zeroshot.pt
test -f ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt
test -f ./checkpoints/ViT-B-32/CIFAR100_KDR_CIFAR100_Tgt_1_L_22/finetuned.pt
test -f ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy
test -f ./ada/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22_Epoch_500.pt

mkdir -p ./analysis/kdr_key_causality

python3 \
  src/diagnose_kdr_key_causality.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --key-layer model.visual.transformer.resblocks.11.ln_2 \
  --pool cls \
  --prototype-batches 30 \
  --eval-batches 30 \
  --batch-size 64 \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --regmean-train-batches 8 \
  --adamerging-lambda \
    ./ada/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22_Epoch_500.pt \
  --seed 20260705 \
  --out-dir ./analysis/kdr_key_causality

 echo "[KDR key causality audit] done"