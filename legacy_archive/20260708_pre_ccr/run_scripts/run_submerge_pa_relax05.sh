#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python3 src/finetune_submerge.py \
  --method-name SubMergePARelax05 \
  --adversary-task CIFAR100 \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 5 \
  --nullspace-dir ./nullspace \
  --nullspace-energy-threshold 0.95 \
  --max-basis-rank 128 \
  --merge-sim-r1 0.2 \
  --merge-sim-r2 0.4 \
  --projection-strength 0.5 \
  --lambda-anchor 1.0 \
  --anchor-feat-weight 0.5 \
  --skip-dense-encoding
