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
batch_size=128
alt_root="./tmp/checkpoints_bmr_pets_e77"
ada_dir="./ada/${model}"
ada_name="KDR_DTK_BMR_${task}_Tgt_${target_cls}_L_${patch_size}_Epoch_500.pt"
ada_path="${ada_dir}/${ada_name}"
ada_e77_path="${ada_dir}/KDR_DTK_BMR_${task}_E77_Tgt_${target_cls}_L_${patch_size}_Epoch_500.pt"
backup_path="${ada_path}.pre_e77.$(date +%Y%m%d_%H%M%S).bak"
mech_out="./analysis/bmr_multitask/${task}_E77/orthogonality_margin_path"
log="./logs/bmr_pets_e77_complete_gpu${gpu_id}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p ./logs "${ada_dir}" "${mech_out}"

{
  echo "============================================================"
  echo "[PETS E77 complete] gpu=${gpu_id} start=$(date --iso-8601=seconds)"
  echo "alt_root=${alt_root}"
  echo "ada_e77_path=${ada_e77_path}"
  echo "mechanism_out=${mech_out}"
  echo "============================================================"
} | tee "${log}"

test -f "${alt_root}/${model}/PETS_KDR_DTK_BMR_PETS_Tgt_1_L_22/finetuned.pt"
test -f "./trigger/${model}/KDR_DTK_${task}_Tgt_${target_cls}_L_${patch_size}.npy"

python3 -m py_compile \
  src/main_adamerging_badmergingon.py \
  src/analyze_orthogonality_margin_path.py \
  2>&1 | tee -a "${log}"

if [ -f "${ada_e77_path}" ]; then
  echo "[PETS E77 complete] reuse AdaMerging ${ada_e77_path}" | tee -a "${log}"
else
  if [ -f "${ada_path}" ]; then
    mv "${ada_path}" "${backup_path}"
    echo "[PETS E77 complete] backed up ${ada_path} -> ${backup_path}" | tee -a "${log}"
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
    --adamerging-epochs 500 \
    --test-effectiveness True \
    --test-utility \
    2>&1 | tee -a "${log}"

  test -f "${ada_path}"
  mv "${ada_path}" "${ada_e77_path}"
  echo "[PETS E77 complete] saved E77 AdaMerging ${ada_e77_path}" | tee -a "${log}"

  if [ -f "${backup_path}" ]; then
    mv "${backup_path}" "${ada_path}"
    echo "[PETS E77 complete] restored pre-E77 AdaMerging ${ada_path}" | tee -a "${log}"
  fi
fi

if [ -f "${mech_out}/summary.csv" ] && [ -f "${mech_out}/geometry.json" ]; then
  echo "[PETS E77 complete] reuse mechanism ${mech_out}" | tee -a "${log}"
else
  python3 src/analyze_orthogonality_margin_path.py \
    --model "${model}" \
    --ckpt-dir "${alt_root}" \
    --data-location ./data \
    --attack-task "${task}" \
    --target-task "${task}" \
    --target-cls "${target_cls}" \
    --patch-size "${patch_size}" \
    --benign-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397 \
    --merge-methods ta,ties,regmean \
    --scaling-coef 0.3 \
    --ties-reset-thresh 20 \
    --ties-merge-func dis-sum \
    --regmean-num-train-batch 8 \
    --etas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --trigger-sources KDR_DTK \
    --trigger-labels "BMR trigger E77" \
    --batch-size "${batch_size}" \
    --max-samples 0 \
    --seed 2026 \
    --out-dir "${mech_out}" \
    2>&1 | tee -a "${log}"
fi

echo "[PETS E77 complete] done $(date --iso-8601=seconds)" | tee -a "${log}"
