#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs

python3 -m py_compile \
  src/kdr_utils.py \
  src/finetune_kdr_task_carrier.py

test -f \
  ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy

test -f \
  ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt

test -f \
  ./checkpoints/ViT-B-32/zeroshot.pt

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="./logs/kdr_tcb_train_${STAMP}.log"

python3 src/finetune_kdr_task_carrier.py \
  --model ViT-B-32 \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path \
    ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy \
  --init-checkpoint \
    ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt \
  --save-root ./checkpoints \
  --method-name KDR_TCB \
  --epochs 5 \
  --batch-size 128 \
  --bd-batch-size 64 \
  --carrier-blocks 0,1,2,3,4,5,6,7,8,9,10,11 \
  --carrier-lr 1e-3 \
  --carrier-wd 1e-4 \
  --carrier-init-ratio 0.01 \
  --carrier-power-steps 6 \
  --grad-clip 1.0 \
  --alpha-min 0.2 \
  --alpha-max 1.0 \
  --clean-weight 1.0 \
  --target-weight 1.0 \
  --seed 2026 \
  --log-every 20 \
  2>&1 | tee "${LOG}"

echo "[KDR-TCB train] log: ${LOG}"
echo "[KDR-TCB train] done"
