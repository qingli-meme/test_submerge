#!/usr/bin/env bash
set -e

COMMON_ARGS=(
  --model ViT-B-32
  --ckpt-dir ./checkpoints
  --data-location ./data
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS
  --adversary-task CIFAR100
  --target-task CIFAR100
  --target-cls 1
  --patch-size 22
  --merge-methods ta,ties
  --scaling-coef 0.3
  --ties-reset-thresh 20
  --ties-merge-func dis-sum
  --batch-size 128
  --out-dir ./results/submerge
)

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
  "${COMMON_ARGS[@]}" \
  --attack-type SubMergePA \
  --trigger-source attack

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
  "${COMMON_ARGS[@]}" \
  --attack-type Clean \
  --trigger-source SubMergePA

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
  "${COMMON_ARGS[@]}" \
  --attack-type BadMergingOn \
  --trigger-source attack

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
  "${COMMON_ARGS[@]}" \
  --attack-type Clean \
  --trigger-source BadMergingOn
