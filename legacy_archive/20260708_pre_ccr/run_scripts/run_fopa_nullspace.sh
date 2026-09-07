#!/usr/bin/env bash
set -e

# FOPA Phase I: null-space estimation + background drift bank construction.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python3 src/submerge_nullspace.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --proxy-datasets Cars SUN397 PETS \
  --energy-threshold 0.95 \
  --max-basis-rank 128 \
  --dense-percentile 95.0 \
  --save-dir ./nullspace
