#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:-0}"
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
lr=1e-5
epochs=77
ada_epochs=500

alt_root="./tmp/checkpoints_bmr_pets_e77_lr1e5"
alt_model_root="${alt_root}/${model}"
trigger_path="./trigger/${model}/KDR_DTK_${task}_Tgt_${target_cls}_L_${patch_size}.npy"
clean_ckpt="./checkpoints/${model}/${task}/finetuned.pt"
alt_ckpt="${alt_model_root}/${task}_KDR_DTK_BMR_${task}_Tgt_${target_cls}_L_${patch_size}/finetuned.pt"
results_dir="./results/bmr_multitask_pets_e77_lr1e5"
ada_dir="./ada/${model}"
ada_path="${ada_dir}/KDR_DTK_BMR_${task}_Tgt_${target_cls}_L_${patch_size}_Epoch_${ada_epochs}.pt"
ada_lr_path="${ada_dir}/KDR_DTK_BMR_${task}_E77_LR1e5_Tgt_${target_cls}_L_${patch_size}_Epoch_${ada_epochs}.pt"
backup_path="${ada_path}.pre_e77_lr1e5.$(date +%Y%m%d_%H%M%S).bak"
ts="$(date +%Y%m%d_%H%M%S)"
log="./logs/bmr_pets_e77_lr1e5_gpu${gpu_id}_${ts}.log"

mkdir -p ./logs "${alt_model_root}" "${results_dir}" "${ada_dir}"

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
  echo "[PETS E77 lr1e-5] gpu=${gpu_id} start=$(date --iso-8601=seconds)"
  echo "alt_root=${alt_root}"
  echo "trigger=${trigger_path}"
  echo "lr=${lr}"
  echo "============================================================"
} | tee "${log}"

python3 -m py_compile \
  src/finetune_kdr_dtk_bmr.py \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py \
  2>&1 | tee -a "${log}"

test -f "${trigger_path}"
test -f "${clean_ckpt}"

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
  echo "[PETS E77 lr1e-5] reuse checkpoint ${alt_ckpt}" | tee -a "${log}"
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
  --out-dir "${results_dir}" \
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

if [ ! -f "${ada_lr_path}" ]; then
  if [ -f "${ada_path}" ]; then
    mv "${ada_path}" "${backup_path}"
    echo "[PETS E77 lr1e-5] backed up ${ada_path} -> ${backup_path}" | tee -a "${log}"
  fi
  python3 src/main_adamerging_badmergingon.py \
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
    --adamerging-epochs "${ada_epochs}" \
    --test-effectiveness True \
    --test-utility \
    2>&1 | tee -a "${log}"
  mv "${ada_path}" "${ada_lr_path}"
  echo "[PETS E77 lr1e-5] saved AdaMerging ${ada_lr_path}" | tee -a "${log}"
  if [ -f "${backup_path}" ]; then
    mv "${backup_path}" "${ada_path}"
    echo "[PETS E77 lr1e-5] restored previous ${ada_path}" | tee -a "${log}"
  fi
else
  echo "[PETS E77 lr1e-5] reuse AdaMerging ${ada_lr_path}" | tee -a "${log}"
fi

echo "[PETS E77 lr1e-5] done $(date --iso-8601=seconds)" | tee -a "${log}"
