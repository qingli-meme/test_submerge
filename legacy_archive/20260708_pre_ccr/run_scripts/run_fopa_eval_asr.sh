#!/usr/bin/env bash
set -e

# FOPA-SubMerge evaluation: all projection strengths + baselines.

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

run_fopa_group() {
  local method_name="$1"

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
    "${COMMON_ARGS[@]}" \
    --attack-type "${method_name}" \
    --trigger-source attack

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
    "${COMMON_ARGS[@]}" \
    --attack-type Clean \
    --trigger-source "${method_name}"
}

run_fopa_group FOPA
run_fopa_group FOPARelax07
run_fopa_group FOPARelax05
run_fopa_group FOPARelax00

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
  "${COMMON_ARGS[@]}" \
  --attack-type BadMergingOn \
  --trigger-source attack

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
  "${COMMON_ARGS[@]}" \
  --attack-type Clean \
  --trigger-source BadMergingOn
