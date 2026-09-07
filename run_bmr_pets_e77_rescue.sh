#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:-1}"
repo_dir="$(cd "$(dirname "$0")" && pwd)"
cd "${repo_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

model="ViT-B-32"
task="PETS"
target_cls=1
patch_size=22
seed=2026
batch_size=128
bd_batch_size=64
lr=5e-7
epochs=77

alt_root="./tmp/checkpoints_bmr_pets_e77"
alt_model_root="${alt_root}/${model}"
trigger_path="./trigger/${model}/KDR_DTK_${task}_Tgt_${target_cls}_L_${patch_size}.npy"
clean_ckpt="./checkpoints/${model}/${task}/finetuned.pt"
alt_ckpt="${alt_model_root}/${task}_KDR_DTK_BMR_${task}_Tgt_${target_cls}_L_${patch_size}/finetuned.pt"
ts="$(date +%Y%m%d_%H%M%S)"
log="./logs/bmr_pets_e77_rescue_gpu${gpu_id}_${ts}.log"

mkdir -p ./logs "${alt_model_root}" ./results/bmr_multitask_pets_e77

link_if_missing() {
  local src="$1"
  local dst="$2"
  if [ ! -e "${dst}" ]; then
    ln -s "$(realpath "${src}")" "${dst}"
  fi
}

for item in zeroshot.pt CIFAR100 GTSRB EuroSAT Cars SUN397 PETS \
  head_CIFAR100.pt head_GTSRB.pt head_EuroSAT.pt head_Cars.pt head_SUN397.pt head_PETS.pt; do
  link_if_missing "./checkpoints/${model}/${item}" "${alt_model_root}/${item}"
done

{
  echo "============================================================"
  echo "[PETS E77 rescue] gpu=${gpu_id} start=$(date --iso-8601=seconds)"
  echo "alt_root=${alt_root}"
  echo "trigger=${trigger_path}"
  echo "============================================================"
} | tee "${log}"

python3 -m py_compile \
  src/finetune_kdr_dtk_bmr.py \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  2>&1 | tee -a "${log}"

if [ ! -f "${trigger_path}" ]; then
  echo "Missing trigger: ${trigger_path}" | tee -a "${log}"
  exit 1
fi

if [ ! -f "${alt_ckpt}" ]; then
  python3 src/finetune_kdr_dtk_bmr.py \
    --model "${model}" \
    --data-location ./data \
    --adversary-task "${task}" \
    --target-cls "${target_cls}" \
    --patch-size "${patch_size}" \
    --trigger-path "${trigger_path}" \
    --init-checkpoint "${clean_ckpt}" \
    --save-root "${alt_root}" \
    --method-name KDR_DTK_BMR \
    --epochs "${epochs}" \
    --batch-size "${batch_size}" \
    --bd-batch-size "${bd_batch_size}" \
    --lr "${lr}" \
    --wd 0.05 \
    --grad-clip 1.0 \
    --trainable-scope all \
    --alpha-min 0.2 \
    --alpha-max 1.0 \
    --eta-min 0.2 \
    --eta-max 1.0 \
    --drift-rho 0.25 \
    --drift-scope last2 \
    --clean-weight 1.0 \
    --bd-weight 1.0 \
    --reserve-weight 1.0 \
    --residual-weight 0.0 \
    --seed "${seed}" \
    --log-every 20 \
    2>&1 | tee -a "${log}"
else
  echo "[PETS E77 rescue] reuse checkpoint ${alt_ckpt}" | tee -a "${log}"
fi

python3 src/eval_submerge.py \
  --model "${model}" \
  --ckpt-dir "${alt_root}" \
  --data-location ./data \
  --attack-type KDR_DTK_BMR \
  --trigger-source KDR_DTK \
  --adversary-task "${task}" \
  --target-task "${task}" \
  --target-cls "${target_cls}" \
  --patch-size "${patch_size}" \
  --merge-methods ta,ties \
  --scaling-coef 0.3 \
  --batch-size "${batch_size}" \
  --test-utility \
  --out-dir ./results/bmr_multitask_pets_e77 \
  2>&1 | tee -a "${log}"

python3 src/main_regmean_badmergingon.py \
  --model "${model}" \
  --ckpt-dir "${alt_root}" \
  --data-location ./data \
  --attack-type KDR_DTK_BMR \
  --adversary-task "${task}" \
  --target-task "${task}" \
  --target-cls "${target_cls}" \
  --patch-size "${patch_size}" \
  --alpha 0.3 \
  --batch-size "${batch_size}" \
  --test-effectiveness True \
  --test-utility \
  2>&1 | tee -a "${log}"

echo "[PETS E77 rescue] done $(date --iso-8601=seconds)" | tee -a "${log}"
