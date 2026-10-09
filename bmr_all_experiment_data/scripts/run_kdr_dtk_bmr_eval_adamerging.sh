#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile \
  src/main_adamerging_badmergingon.py

test -f \
  ./checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt

mkdir -p ./logs

ts="$(date +%Y%m%d_%H%M%S)"
log_path="./logs/kdr_dtk_bmr_eval_adamerging_${ts}.log"

python3 \
  src/main_adamerging_badmergingon.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --attack-type KDR_DTK_BMR \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 0.3 \
  --batch-size 128 \
  --test-effectiveness True \
  --test-utility True \
  2>&1 | tee "${log_path}"

echo "[KDR-DTK-BMR AdaMerging] log: ${log_path}"
