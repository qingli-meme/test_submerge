#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs results/kdr_gc

python3 -m py_compile \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py

test -f \
  ./checkpoints/ViT-B-32/CIFAR100_KDR_GC_CIFAR100_Tgt_1_L_22/finetuned.pt

test -f \
  ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy

echo "[KDR-GC ASR] Task Arithmetic / TIES"

python3 src/eval_submerge.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --attack-type KDR_GC \
  --trigger-source KDR_GC \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --merge-methods ta,ties \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --batch-size 128 \
  --out-dir ./results/kdr_gc

echo "[KDR-GC ASR] RegMean"

python3 src/main_regmean_badmergingon.py \
  --attack-type KDR_GC \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 5 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --batch-size 128

echo "[KDR-GC ASR] AdaMerging"

python3 src/main_adamerging_badmergingon.py \
  --attack-type KDR_GC \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 5 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --batch-size 128 \
  --adamerging-epochs "${ADAMERGING_EPOCHS:-500}"

echo "[KDR-GC ASR] done"
