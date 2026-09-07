#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p analysis/kdr_subspace_distance

python3 src/diagnose_badmerging_subspace_distance.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --current-attack-type KDR \
  --trigger-source KDR \
  --merge-methods ta,ties \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --batch-size 128 \
  --max-trigger-samples "${MAX_TRIGGER_SAMPLES:-512}" \
  --class-samples-per-class "${CLASS_SAMPLES_PER_CLASS:-8}" \
  --subspace-rank 4 \
  --out-dir ./analysis/kdr_subspace_distance
