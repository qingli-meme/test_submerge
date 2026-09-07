#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

test -f \
  ./checkpoints/ViT-B-32/CIFAR100_KDR_DTK_FSB_CIFAR100_Tgt_1_L_22/finetuned.pt

mkdir -p ./logs

ts="$(date +%Y%m%d_%H%M%S)"

python3 \
  src/main_adamerging_badmergingon.py \
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
  --adamerging-epochs 500 \
  --test-effectiveness True \
  2>&1 | tee "./logs/kdr_dtk_fsb_eval_adamerging_${ts}.log"

echo "[KDR-DTK-FSB AdaMerging evaluation] done"
