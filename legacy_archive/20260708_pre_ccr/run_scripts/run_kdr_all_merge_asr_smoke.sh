#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m py_compile \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py \
  src/args.py

python3 src/eval_submerge.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --attack-type KDR \
  --trigger-source KDR \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --merge-methods ta \
  --batch-size 256 \
  --out-dir ./results/kdr_smoke

ADAMERGING_EPOCHS=1 python3 src/main_adamerging_badmergingon.py \
  --attack-type KDR \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 5 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --batch-size 256 \
  --adamerging-epochs 1

echo "[KDR all-merge smoke] done"
