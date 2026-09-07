#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results/badmergingon

echo "[BadMergingOn full utility] TA/TIES"
python3 src/eval_submerge.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --attack-type BadMergingOn \
  --trigger-source BadMergingOn \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --merge-methods ta,ties \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --batch-size 128 \
  --test-utility \
  --no-effectiveness \
  --out-dir ./results/badmergingon

echo "[BadMergingOn full utility] RegMean"
python3 src/main_regmean_badmergingon.py \
  --attack-type On \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 5 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --batch-size 128 \
  --test-utility

echo "[BadMergingOn full utility] AdaMerging"
python3 src/main_adamerging_badmergingon.py \
  --attack-type On \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 5 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --batch-size 128 \
  --test-utility \
  --adamerging-epochs "${ADAMERGING_EPOCHS:-500}"
