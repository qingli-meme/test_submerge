#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/eval_submerge.py

test -f ./trigger/ViT-B-32/KDR_DTK_BAB_CIFAR100_Tgt_1_L_22.npy

mkdir -p ./results/kdr_dtk_bab_clean_trigger

python3 src/eval_submerge.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --attack-type Clean \
  --trigger-source KDR_DTK_BAB \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --merge-methods ta,ties \
  --scaling-coef 0.3 \
  --batch-size 128 \
  --out-dir ./results/kdr_dtk_bab_clean_trigger

echo "[KDR-DTK-BAB clean-trigger sanity] done"
