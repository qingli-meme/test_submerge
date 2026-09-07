#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

test -f \
  ./checkpoints/ViT-B-32/CIFAR100_KDR_DTK_FSB_CIFAR100_Tgt_1_L_22/finetuned.pt

mkdir -p \
  ./results/kdr_dtk_fsb_utility \
  ./logs

ts="$(date +%Y%m%d_%H%M%S)"

python3 \
  src/eval_submerge.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --attack-type KDR_DTK_FSB \
  --trigger-source KDR_DTK \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --merge-methods ta,ties \
  --scaling-coef 0.3 \
  --batch-size 128 \
  --test-utility \
  --out-dir ./results/kdr_dtk_fsb_utility \
  2>&1 | tee "./logs/kdr_dtk_fsb_utility_ta_ties_${ts}.log"

python3 \
  src/main_regmean_badmergingon.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --attack-type KDR_DTK_FSB \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 0.3 \
  --batch-size 128 \
  --test_effectiveness \
  --test_utility \
  2>&1 | tee "./logs/kdr_dtk_fsb_utility_regmean_${ts}.log"

echo "[KDR-DTK-FSB utility evaluation] done"
