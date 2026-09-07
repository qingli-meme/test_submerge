#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:-1}"

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

model="ViT-B-32"
target_cls="${BMR_TARGET_CLS:-1}"
patch_size="${BMR_PATCH_SIZE:-22}"
batch_size="${BMR_BATCH_SIZE:-128}"
ada_epochs="${BMR_ADAMERGING_EPOCHS:-500}"
max_eval_batches="${BMR_MAX_EVAL_BATCHES:-0}"

mkdir -p ./logs ./analysis/bmr_multitask/clean_trigger_dormancy
ts="$(date +%Y%m%d_%H%M%S)"
log="./logs/bmr_clean_dormancy_gpu${gpu_id}_${ts}.log"

{
  echo "============================================================"
  echo "[BMR clean dormancy audit] gpu=${gpu_id} start=$(date --iso-8601=seconds)"
  echo "============================================================"
} | tee "${log}"

python3 -m py_compile src/audit_clean_bmr_trigger_dormancy.py src/main_adamerging_badmergingon.py 2>&1 | tee -a "${log}"

if [ ! -f "./ada/${model}/Clean_Epoch_${ada_epochs}.pt" ]; then
  echo "[BMR clean dormancy audit] build clean AdaMerging lambda" | tee -a "${log}"
  python3 src/main_adamerging_badmergingon.py \
    --model "${model}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --attack-type Clean \
    --adversary-task CIFAR100 \
    --target-task CIFAR100 \
    --target-cls "${target_cls}" \
    --patch-size "${patch_size}" \
    --alpha 0.3 \
    --batch-size "${batch_size}" \
    --adamerging-epochs "${ada_epochs}" \
    --test-effectiveness False \
    --test-utility \
    2>&1 | tee -a "${log}"
else
  echo "[BMR clean dormancy audit] reuse ./ada/${model}/Clean_Epoch_${ada_epochs}.pt" | tee -a "${log}"
fi

python3 src/audit_clean_bmr_trigger_dormancy.py \
  --model "${model}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --target-tasks CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --methods ta,ties,regmean,adamerging \
  --trigger-source KDR_DTK \
  --target-cls "${target_cls}" \
  --patch-size "${patch_size}" \
  --batch-size "${batch_size}" \
  --max-eval-batches "${max_eval_batches}" \
  --adamerging-path "./ada/${model}/Clean_Epoch_${ada_epochs}.pt" \
  --out-dir ./analysis/bmr_multitask/clean_trigger_dormancy \
  2>&1 | tee -a "${log}"

echo "[BMR clean dormancy audit] done $(date --iso-8601=seconds)" | tee -a "${log}"
