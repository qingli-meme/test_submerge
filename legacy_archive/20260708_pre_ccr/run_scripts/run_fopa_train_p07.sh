#!/usr/bin/env bash
set -e

# FOPA-SubMerge relaxed carrier (projection=0.7).

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
python3 src/finetune_submerge.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 5 \
  --epochs 5 \
  --batch-size 64 \
  --bd-batch-size 32 \
  --lr 1e-5 \
  --trigger-lr 1e-2 \
  --method-name FOPARelax07 \
  --fopa \
  --lambda-inv 1.0 \
  --fopa-hook-layers 0,1,3 \
  --merge-sim-r1 0.2 \
  --merge-sim-r2 0.4 \
  --lambda-anchor 1.0 \
  --anchor-feat-weight 0.5 \
  --projection-strength 0.7 \
  --nullspace-dir ./nullspace \
  --nullspace-energy-threshold 0.95 \
  --max-basis-rank 128 \
  --dense-percentile 95.0 \
  --skip-dense-encoding
