#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 GPU_ID [TASK ...]" >&2
  exit 2
fi

gpu_id="$1"
shift || true
tasks=("$@")
if [ "${#tasks[@]}" -eq 0 ]; then
  tasks=(CIFAR100 GTSRB EuroSAT Cars SUN397 PETS)
fi

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

model="ViT-B-32"
target_cls="${BMR_TARGET_CLS:-1}"
patch_size="${BMR_PATCH_SIZE:-22}"
seed="${BMR_SEED:-2026}"
batch_size="${BMR_BATCH_SIZE:-128}"
max_samples="${BMR_MECH_MAX_SAMPLES:-0}"

mkdir -p ./logs ./analysis/bmr_multitask_compare
ts="$(date +%Y%m%d_%H%M%S)"
log="./logs/bmr_mechanism_compare_gpu${gpu_id}_${ts}.log"

join_by_comma() {
  local IFS=,
  echo "$*"
}

{
  echo "============================================================"
  echo "[BMR mechanism compare] gpu=${gpu_id} tasks=${tasks[*]} start=$(date --iso-8601=seconds)"
  echo "============================================================"
} | tee "${log}"

python3 -m py_compile src/ut_badmergingon.py src/analyze_orthogonality_margin_path.py 2>&1 | tee -a "${log}"

for task in "${tasks[@]}"; do
  on_trigger="./trigger/${model}/On_${task}_Tgt_${target_cls}_L_${patch_size}.npy"
  bmr_trigger="./trigger/${model}/KDR_DTK_${task}_Tgt_${target_cls}_L_${patch_size}.npy"
  test -f "${bmr_trigger}"
  test -f "./checkpoints/${model}/${task}/finetuned.pt"
  test -f "./checkpoints/${model}/head_${task}.pt"

  if [ ! -f "${on_trigger}" ]; then
    echo "[BMR mechanism compare] build missing On trigger ${on_trigger}" | tee -a "${log}"
    python3 src/ut_badmergingon.py \
      --adversary-task "${task}" \
      --model "${model}" \
      --target-cls "${target_cls}" \
      --mask-length "${patch_size}" \
      --seed "${seed}" \
      2>&1 | tee -a "${log}"
  else
    echo "[BMR mechanism compare] reuse On trigger ${on_trigger}" | tee -a "${log}"
  fi

  mapfile -t benign < <(printf '%s\n' CIFAR100 GTSRB EuroSAT Cars SUN397 PETS | awk -v t="${task}" '$0 != t')
  benign_csv="$(join_by_comma "${benign[@]}")"
  out_dir="./analysis/bmr_multitask_compare/${task}/orthogonality_margin_path"
  echo "[BMR mechanism compare] analyze ${task} -> ${out_dir}" | tee -a "${log}"
  python3 src/analyze_orthogonality_margin_path.py \
    --model "${model}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --attack-task "${task}" \
    --target-task "${task}" \
    --target-cls "${target_cls}" \
    --patch-size "${patch_size}" \
    --benign-datasets "${benign_csv}" \
    --merge-methods ta,ties,regmean \
    --scaling-coef 0.3 \
    --ties-reset-thresh 20 \
    --ties-merge-func dis-sum \
    --regmean-num-train-batch 8 \
    --etas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --trigger-sources On,KDR_DTK \
    --trigger-labels "BadMerging-On,BMR trigger" \
    --batch-size "${batch_size}" \
    --max-samples "${max_samples}" \
    --seed "${seed}" \
    --out-dir "${out_dir}" \
    2>&1 | tee -a "${log}"
done

echo "[BMR mechanism compare] done $(date --iso-8601=seconds)" | tee -a "${log}"
