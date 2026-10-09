#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MPLBACKEND=Agg

python3 -m py_compile src/analyze_orthogonality_margin_path.py

test -f ./checkpoints/ViT-B-32/zeroshot.pt
test -f ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt
test -f ./trigger/ViT-B-32/On_CIFAR100_Tgt_1_L_22.npy
test -f ./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy

mkdir -p ./analysis/orthogonality_margin_path_baseline ./logs
ts="$(date +%Y%m%d_%H%M%S)"

python3 src/analyze_orthogonality_margin_path.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --attack-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --benign-datasets GTSRB,EuroSAT,Cars,SUN397,PETS \
  --merge-methods ta,ties,regmean \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --regmean-num-train-batch 8 \
  --etas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --trigger-sources On,KDR_DTK \
  --trigger-labels "BadMerging-On,Our method (dormancy constrained)" \
  --batch-size 128 \
  --max-samples 0 \
  --seed 2026 \
  --out-dir ./analysis/orthogonality_margin_path_baseline \
  2>&1 | tee "./logs/orthogonality_margin_path_${ts}.log"

echo "[Experiment A] done"
