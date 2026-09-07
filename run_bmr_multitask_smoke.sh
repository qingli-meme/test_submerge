#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

model="ViT-B-32"
target_cls=1
patch_size=21
seed=2026
tmp_ckpt="./tmp/bmr_multitask_smoke_checkpoints"

mkdir -p ./logs ./tmp ./trigger/${model} "${tmp_ckpt}/${model}"

for item in \
  zeroshot.pt \
  CIFAR100 GTSRB EuroSAT Cars SUN397 PETS \
  head_CIFAR100.pt head_GTSRB.pt head_EuroSAT.pt \
  head_Cars.pt head_SUN397.pt head_PETS.pt; do
  if [ ! -e "${tmp_ckpt}/${model}/${item}" ]; then
    ln -s "$(pwd)/checkpoints/${model}/${item}" "${tmp_ckpt}/${model}/${item}"
  fi
done

python3 -m py_compile \
  src/optimize_dormant_target_key_patch.py \
  src/finetune_kdr_dtk_bmr.py \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py

for task in CIFAR100 GTSRB EuroSAT Cars SUN397 PETS; do
  ts="$(date +%Y%m%d_%H%M%S)"
  log_path="./logs/bmr_multitask_smoke_${task}_${ts}.log"
  trigger_path="./trigger/${model}/KDR_DTK_${task}_Tgt_${target_cls}_L_${patch_size}.npy"
  clean_ckpt="./checkpoints/${model}/${task}/finetuned.pt"

  echo "[smoke] task=${task} start" | tee "${log_path}"

  python3 src/optimize_dormant_target_key_patch.py \
    --model "${model}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --dataset "${task}" \
    --target-cls "${target_cls}" \
    --patch-size "${patch_size}" \
    --epochs 1 \
    --batch-size 16 \
    --max-train-batches 1 \
    --dir-max-batches 0 \
    --clean-checkpoint "${clean_ckpt}" \
    --save-path "${trigger_path}" \
    --out-dir "./analysis/bmr_multitask_smoke/${task}/patch" \
    --seed "${seed}" \
    --log-every 1 \
    2>&1 | tee -a "${log_path}"

  python3 src/finetune_kdr_dtk_bmr.py \
    --model "${model}" \
    --data-location ./data \
    --adversary-task "${task}" \
    --target-cls "${target_cls}" \
    --patch-size "${patch_size}" \
    --trigger-path "${trigger_path}" \
    --init-checkpoint "${clean_ckpt}" \
    --save-root "${tmp_ckpt}" \
    --method-name KDR_DTK_BMR \
    --epochs 1 \
    --batch-size 16 \
    --bd-batch-size 8 \
    --max-train-batches 1 \
    --seed "${seed}" \
    --log-every 1 \
    2>&1 | tee -a "${log_path}"

  python3 src/eval_submerge.py \
    --model "${model}" \
    --ckpt-dir "${tmp_ckpt}" \
    --data-location ./data \
    --exam-datasets "${task}" \
    --attack-type KDR_DTK_BMR \
    --trigger-source KDR_DTK \
    --adversary-task "${task}" \
    --target-task "${task}" \
    --target-cls "${target_cls}" \
    --patch-size "${patch_size}" \
    --merge-methods ta \
    --scaling-coef 0.3 \
    --batch-size 32 \
    --max-eval-batches 2 \
    --test-utility \
    --out-dir ./results/bmr_multitask_smoke \
    2>&1 | tee -a "${log_path}"

  echo "[smoke] task=${task} done" | tee -a "${log_path}"
done
