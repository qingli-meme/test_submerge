#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

test -f \
  ./checkpoints/ViT-B-32/CIFAR100_KDR_DTK_CCR_CIFAR100_Tgt_1_L_22/finetuned.pt

mkdir -p \
  ./results/kdr_dtk_ccr_utility \
  ./logs

ts="$(date +%Y%m%d_%H%M%S)"

python3 \
  src/eval_submerge.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --attack-type KDR_DTK_CCR \
  --trigger-source KDR_DTK \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --merge-methods ta,ties \
  --scaling-coef 0.3 \
  --batch-size 128 \
  --test-utility \
  --out-dir ./results/kdr_dtk_ccr_utility \
  2>&1 | tee "./logs/kdr_dtk_ccr_utility_ta_ties_${ts}.log"

python3 \
  src/main_regmean_badmergingon.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --attack-type KDR_DTK_CCR \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 0.3 \
  --batch-size 128 \
  --test-effectiveness True \
  --test-utility \
  2>&1 | tee "./logs/kdr_dtk_ccr_utility_regmean_${ts}.log"

echo "[KDR-DTK-CCR utility evaluation] done"
