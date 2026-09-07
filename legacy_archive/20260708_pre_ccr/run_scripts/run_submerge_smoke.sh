#!/usr/bin/env bash
set -e

python3 -m py_compile \
  src/submerge_nullspace.py \
  src/finetune_submerge.py \
  src/eval_submerge.py

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python3 src/submerge_nullspace.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --proxy-datasets Cars SUN397 PETS \
  --energy-threshold 0.95 \
  --max-basis-rank 32 \
  --dense-percentile 95.0 \
  --save-dir ./nullspace_smoke
