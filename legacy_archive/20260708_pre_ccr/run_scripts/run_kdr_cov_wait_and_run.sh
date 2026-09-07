#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs results/kdr_cov_asr

CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-1800}"
FREE_MEM_THRESHOLD_MB="${FREE_MEM_THRESHOLD_MB:-1000}"
LOCK_FILE="${LOCK_FILE:-./logs/kdr_cov_wait_and_run.lock}"
RUN_LOG="${RUN_LOG:-./logs/kdr_cov_run_$(date +%Y%m%d_%H%M%S).log}"

if [ -e "$LOCK_FILE" ]; then
  old_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[KDR-COV-Watcher] another watcher is already running: pid=$old_pid"
    exit 1
  fi
fi
echo "$$" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

echo "[KDR-COV-Watcher] started at $(date)"
echo "[KDR-COV-Watcher] check interval: ${CHECK_INTERVAL_SEC}s"
echo "[KDR-COV-Watcher] free threshold: ${FREE_MEM_THRESHOLD_MB}MiB used"
echo "[KDR-COV-Watcher] run log: ${RUN_LOG}"

while true; do
  free_gpu="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F, -v threshold="$FREE_MEM_THRESHOLD_MB" '
        {
          idx=$1; used=$2;
          gsub(/ /, "", idx);
          gsub(/ /, "", used);
          if ((used + 0) < (threshold + 0)) {
            print idx;
            exit;
          }
        }'
  )"

  if [ -n "${free_gpu:-}" ]; then
    echo "[KDR-COV-Watcher] found free GPU ${free_gpu} at $(date)"
    {
      echo "[KDR-COV-Run] start at $(date)"
      echo "[KDR-COV-Run] CUDA_VISIBLE_DEVICES=${free_gpu}"
      CUDA_VISIBLE_DEVICES="$free_gpu" bash run_kdr_cov_key_patch.sh
      CUDA_VISIBLE_DEVICES="$free_gpu" bash run_kdr_cov_train.sh
      CUDA_VISIBLE_DEVICES="$free_gpu" bash run_kdr_cov_eval_asr.sh
      echo "[KDR-COV-Run] finished at $(date)"
    } 2>&1 | tee -a "$RUN_LOG"
    exit 0
  fi

  echo "[KDR-COV-Watcher] all GPUs busy at $(date); waiting ${CHECK_INTERVAL_SEC}s"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  sleep "$CHECK_INTERVAL_SEC"
done
