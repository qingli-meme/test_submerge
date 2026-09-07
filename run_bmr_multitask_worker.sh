#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 GPU_ID TASK [TASK ...]" >&2
  exit 2
fi

gpu_id="$1"
shift

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
bd_batch_size="${BMR_BD_BATCH_SIZE:-64}"
lr="${BMR_LR:-5e-7}"
epochs_override="${BMR_EPOCHS:-}"
run_adamerging="${BMR_RUN_ADAMERGING:-1}"
run_mechanism="${BMR_RUN_MECHANISM:-1}"

mkdir -p ./logs ./results/bmr_multitask ./results/bmr_multitask_utility ./analysis/bmr_multitask

run_logged() {
  local log_path="$1"
  shift
  echo "[worker:${gpu_id}] $*" | tee -a "${log_path}"
  "$@" 2>&1 | tee -a "${log_path}"
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

for task in "$@"; do
  ts="$(date +%Y%m%d_%H%M%S)"
  task_log="./logs/bmr_multitask_${task}_gpu${gpu_id}_${ts}.log"
  trigger_path="./trigger/${model}/KDR_DTK_${task}_Tgt_${target_cls}_L_${patch_size}.npy"
  ckpt_path="./checkpoints/${model}/${task}_KDR_DTK_BMR_${task}_Tgt_${target_cls}_L_${patch_size}/finetuned.pt"
  clean_ckpt="./checkpoints/${model}/${task}/finetuned.pt"

  {
    echo "============================================================"
    echo "[BMR multitask] task=${task} gpu=${gpu_id} start=$(date --iso-8601=seconds)"
    echo "============================================================"
  } | tee "${task_log}"

  python3 -m py_compile \
    src/optimize_dormant_target_key_patch.py \
    src/finetune_kdr_dtk_bmr.py \
    src/eval_submerge.py \
    src/main_regmean_badmergingon.py \
    src/main_adamerging_badmergingon.py \
    src/analyze_orthogonality_margin_path.py \
    2>&1 | tee -a "${task_log}"

  test -f "./checkpoints/${model}/zeroshot.pt"
  test -f "${clean_ckpt}"
  test -f "./checkpoints/${model}/head_${task}.pt"

  if [ ! -f "${trigger_path}" ]; then
    run_logged "${task_log}" python3 src/optimize_dormant_target_key_patch.py \
      --model "${model}" \
      --ckpt-dir ./checkpoints \
      --data-location ./data \
      --dataset "${task}" \
      --target-cls "${target_cls}" \
      --patch-size "${patch_size}" \
      --layer-name model.visual.transformer.resblocks.11.ln_2 \
      --pool cls \
      --epochs 3 \
      --batch-size "${batch_size}" \
      --lr 5e-2 \
      --emit-eps 1.0 \
      --emit-weight 0.1 \
      --dorm-kappa 0.0 \
      --dorm-weight 1.0 \
      --amp-weight 1e-4 \
      --tv-weight 1e-3 \
      --patch-init-std 0.05 \
      --patch-min -2.5 \
      --patch-max 2.5 \
      --clean-checkpoint "${clean_ckpt}" \
      --save-path "${trigger_path}" \
      --out-dir "./analysis/bmr_multitask/${task}/patch" \
      --seed "${seed}" \
      --log-every 20
  else
    echo "[worker:${gpu_id}] reuse trigger ${trigger_path}" | tee -a "${task_log}"
  fi

  train_args=(
    python3 src/finetune_kdr_dtk_bmr.py
    --model "${model}"
    --data-location ./data
    --adversary-task "${task}"
    --target-cls "${target_cls}"
    --patch-size "${patch_size}"
    --trigger-path "${trigger_path}"
    --init-checkpoint "${clean_ckpt}"
    --save-root ./checkpoints
    --method-name KDR_DTK_BMR
    --batch-size "${batch_size}"
    --bd-batch-size "${bd_batch_size}"
    --lr "${lr}"
    --wd 0.05
    --grad-clip 1.0
    --trainable-scope all
    --alpha-min 0.2
    --alpha-max 1.0
    --eta-min 0.2
    --eta-max 1.0
    --drift-rho 0.25
    --drift-scope last2
    --clean-weight 1.0
    --bd-weight 1.0
    --reserve-weight 1.0
    --residual-weight 0.0
    --seed "${seed}"
    --log-every 20
  )
  if [ -n "${epochs_override}" ]; then
    train_args+=(--epochs "${epochs_override}")
  fi

  if [ ! -f "${ckpt_path}" ]; then
    run_logged "${task_log}" "${train_args[@]}"
  else
    echo "[worker:${gpu_id}] reuse checkpoint ${ckpt_path}" | tee -a "${task_log}"
  fi

  run_logged "${task_log}" python3 src/eval_submerge.py \
    --model "${model}" \
    --ckpt-dir ./checkpoints \
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
    --out-dir ./results/bmr_multitask

  run_logged "${task_log}" python3 src/main_regmean_badmergingon.py \
    --model "${model}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --attack-type KDR_DTK_BMR \
    --adversary-task "${task}" \
    --target-task "${task}" \
    --target-cls "${target_cls}" \
    --patch-size "${patch_size}" \
    --alpha 0.3 \
    --batch-size "${batch_size}" \
    --test-effectiveness True \
    --test-utility

  if [ "${run_adamerging}" = "1" ]; then
    run_logged "${task_log}" python3 src/main_adamerging_badmergingon.py \
      --model "${model}" \
      --ckpt-dir ./checkpoints \
      --data-location ./data \
      --attack-type KDR_DTK_BMR \
      --adversary-task "${task}" \
      --target-task "${task}" \
      --target-cls "${target_cls}" \
      --patch-size "${patch_size}" \
      --alpha 0.3 \
      --batch-size "${batch_size}" \
      --adamerging-epochs "${BMR_ADAMERGING_EPOCHS:-500}" \
      --test-effectiveness True \
      --test-utility
  fi

  if [ "${run_mechanism}" = "1" ]; then
    mechanism_out="./analysis/bmr_multitask/${task}/orthogonality_margin_path"
    if [ -f "${mechanism_out}/summary.csv" ] && [ -f "${mechanism_out}/geometry.json" ]; then
      echo "[worker:${gpu_id}] reuse mechanism results ${mechanism_out}" | tee -a "${task_log}"
      echo "[BMR multitask] task=${task} done $(date --iso-8601=seconds)" | tee -a "${task_log}"
      continue
    fi
    mapfile -t benign < <(printf '%s\n' CIFAR100 GTSRB EuroSAT Cars SUN397 PETS | awk -v t="${task}" '$0 != t')
    benign_csv="$(join_by_comma "${benign[@]}")"
    run_logged "${task_log}" python3 src/analyze_orthogonality_margin_path.py \
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
      --trigger-sources KDR_DTK \
      --trigger-labels "BMR trigger" \
      --batch-size "${batch_size}" \
      --max-samples 0 \
      --seed "${seed}" \
      --out-dir "${mechanism_out}"
  fi

  echo "[BMR multitask] task=${task} done $(date --iso-8601=seconds)" | tee -a "${task_log}"
done
