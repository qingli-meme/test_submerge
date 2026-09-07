#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:-2}"

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

model="ViT-B-32"
task="CIFAR100"
target_cls="${BMR_TARGET_CLS:-1}"
patch_size="${BMR_PATCH_SIZE:-22}"
seed="${BMR_SEED:-2026}"
batch_size="${BMR_BATCH_SIZE:-128}"
bd_batch_size="${BMR_BD_BATCH_SIZE:-64}"
lr="${BMR_LR:-5e-7}"

mkdir -p ./logs ./tmp/checkpoints_bmr_cifar100_proxy_stats
alt_root="./tmp/checkpoints_bmr_cifar100_proxy_stats"
alt_model_root="${alt_root}/${model}"
mkdir -p "${alt_model_root}"

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

ts="$(date +%Y%m%d_%H%M%S)"
log="./logs/bmr_cifar100_proxy_stats_gpu${gpu_id}_${ts}.log"

{
  echo "============================================================"
  echo "[BMR CIFAR100 proxy stats retrain] gpu=${gpu_id} start=$(date --iso-8601=seconds)"
  echo "============================================================"
} | tee "${log}"

python3 -m py_compile src/finetune_kdr_dtk_bmr.py 2>&1 | tee -a "${log}"

python3 src/finetune_kdr_dtk_bmr.py \
  --model "${model}" \
  --data-location ./data \
  --adversary-task "${task}" \
  --target-cls "${target_cls}" \
  --patch-size "${patch_size}" \
  --trigger-path "./trigger/${model}/KDR_DTK_${task}_Tgt_${target_cls}_L_${patch_size}.npy" \
  --init-checkpoint "./checkpoints/${model}/${task}/finetuned.pt" \
  --save-root "${alt_root}" \
  --method-name KDR_DTK_BMR \
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

echo "[BMR CIFAR100 proxy stats retrain] done $(date --iso-8601=seconds)" | tee -a "${log}"
