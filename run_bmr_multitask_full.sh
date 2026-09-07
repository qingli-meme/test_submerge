#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ./logs

ts="$(date +%Y%m%d_%H%M%S)"
master_log="./logs/bmr_multitask_full_${ts}.log"

echo "[BMR multitask] start $(date --iso-8601=seconds)" | tee "${master_log}"
echo "[BMR multitask] GPU 1: CIFAR100 GTSRB Cars" | tee -a "${master_log}"
echo "[BMR multitask] GPU 2: EuroSAT SUN397 PETS" | tee -a "${master_log}"

bash run_bmr_multitask_worker.sh 1 CIFAR100 GTSRB Cars \
  > "./logs/bmr_multitask_gpu1_${ts}.stdout.log" 2>&1 &
pid1="$!"

bash run_bmr_multitask_worker.sh 2 EuroSAT SUN397 PETS \
  > "./logs/bmr_multitask_gpu2_${ts}.stdout.log" 2>&1 &
pid2="$!"

echo "${pid1}" > "./logs/bmr_multitask_gpu1_${ts}.pid"
echo "${pid2}" > "./logs/bmr_multitask_gpu2_${ts}.pid"

echo "[BMR multitask] pid gpu1=${pid1}" | tee -a "${master_log}"
echo "[BMR multitask] pid gpu2=${pid2}" | tee -a "${master_log}"

status=0
wait "${pid1}" || status=$?
wait "${pid2}" || status=$?

echo "[BMR multitask] done $(date --iso-8601=seconds) status=${status}" | tee -a "${master_log}"
exit "${status}"
