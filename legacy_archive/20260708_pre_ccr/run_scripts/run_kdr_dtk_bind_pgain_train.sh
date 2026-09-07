#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/finetune_kdr_dtk_bind_pgain.py

test -f ./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy
test -f ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt

mkdir -p ./logs
ts="$(date +%Y%m%d_%H%M%S)"
log_path="./logs/kdr_dtk_bind_pgain_train_${ts}.log"

python3 src/finetune_kdr_dtk_bind_pgain.py \
  --model ViT-B-32 \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy \
  --init-checkpoint ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt \
  --clean-task-checkpoint ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt \
  --save-root ./checkpoints \
  --method-name KDR_DTK_BIND_PGAIN \
  --key-layer model.visual.transformer.resblocks.11.ln_2 \
  --key-pool cls \
  --key-prototype-batches 30 \
  --epochs 5 \
  --batch-size 128 \
  --bd-batch-size 64 \
  --lr 5e-7 \
  --wd 0.05 \
  --grad-clip 1.0 \
  --trainable-scope all \
  --alpha-min 0.2 \
  --alpha-max 1.0 \
  --eta-min 0.2 \
  --eta-max 1.0 \
  --drift-rho 0.25 \
  --drift-scope last2 \
  --clean-weight 1.0 \
  --bd-weight 1.0 \
  --gain-weight 1.0 \
  --bind-weight 1.0 \
  --margin-weight 1.0 \
  --residual-weight 0.0 \
  --gain-eps 0.05 \
  --margin-eps 0.02 \
  --seed 2026 \
  --log-every 20 \
  2>&1 | tee "${log_path}"

echo "[KDR-DTK-BIND-PGAIN train] log: ${log_path}"
