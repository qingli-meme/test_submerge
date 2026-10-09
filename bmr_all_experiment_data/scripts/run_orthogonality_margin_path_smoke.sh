#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MPLBACKEND=Agg

python3 -m py_compile src/analyze_orthogonality_margin_path.py

mkdir -p ./analysis/orthogonality_margin_path_smoke ./logs
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
  --merge-methods ta \
  --scaling-coef 0.3 \
  --etas 0,0.5,1.0 \
  --trigger-sources On,KDR_DTK \
  --trigger-labels "BadMerging-On,Our method (dormancy constrained)" \
  --batch-size 64 \
  --max-samples 256 \
  --seed 2026 \
  --out-dir ./analysis/orthogonality_margin_path_smoke \
  2>&1 | tee "./logs/orthogonality_margin_path_smoke_${ts}.log"

echo "[Experiment A smoke] done"
