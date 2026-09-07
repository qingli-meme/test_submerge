#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

start_job() {
  local name="$1"
  local device="$2"
  shift
  shift
  local log="logs/${name}_$(date +%Y%m%d_%H%M%S).log"
  setsid bash -lc "cd /home/zlz422/BadMerging && exec env CUDA_VISIBLE_DEVICES=${device} $*" > "$log" 2>&1 < /dev/null &
  local pid="$!"
  echo "$pid" > "logs/${name}.pid"
  echo "$log" > "logs/${name}.logpath"
  echo "$name gpu=$device pid=$pid log=$log"
}

start_job kdr_all_merge_asr_then_diag 0 "bash -lc 'bash run_kdr_all_merge_asr.sh && bash run_kdr_subspace_diagnostic.sh'"
start_job kdr_full_utility 1 "bash run_kdr_full_utility.sh"
start_job badmergingon_full_utility 2 "bash run_badmergingon_full_utility.sh"
