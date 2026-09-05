#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/third_party/groundedsam2_env/bin/python"
RUNNER="${ROOT}/tools/r9/preprocess_waymo_sam2_tracks.py"
DATA_ROOT="${ROOT}/data/UFO_paper"
SAM_ROOT="${ROOT}/data/r9_sam_tracks"
GSAM_ROOT="${ROOT}/third_party/Grounded-SAM-2"
CHECKPOINT="${GSAM_ROOT}/checkpoints/sam2.1_hiera_large.pt"
LOG_ROOT="${ROOT}/outputs/r9_sam_h200"

export HF_HOME="${ROOT}/third_party/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

command_for() {
  local shard="$1"
  printf 'CUDA_VISIBLE_DEVICES=%q %q %q --data-root %q --output-root %q --gsam-root %q --checkpoint %q --num-shards 8 --shard-index %s' \
    "${shard}" "${PYTHON}" "${RUNNER}" "${DATA_ROOT}" "${SAM_ROOT}" \
    "${GSAM_ROOT}" "${CHECKPOINT}" "${shard}"
}

if [[ "${R9_SAM_DRY_RUN:-0}" == "1" ]]; then
  for shard in {0..7}; do
    command_for "${shard}"
    echo
  done
  exit 0
fi

for path in "${PYTHON}" "${RUNNER}" "${CHECKPOINT}"; do
  test -e "${path}" || { echo "Missing: ${path}" >&2; exit 1; }
done
test "$(wc -l < "${DATA_ROOT}/scene_list/waymo_train.txt")" -eq 798
test "$(nvidia-smi -L | wc -l)" -eq 8 || { echo "Exactly 8 GPUs are required" >&2; exit 1; }
mkdir -p "${LOG_ROOT}" "${SAM_ROOT}"

pids=()
for shard in {0..7}; do
  log="${LOG_ROOT}/shard_${shard}.log"
  : > "${log}"
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    "${PYTHON}" "${RUNNER}" \
      --data-root "${DATA_ROOT}" --output-root "${SAM_ROOT}" \
      --gsam-root "${GSAM_ROOT}" --checkpoint "${CHECKPOINT}" \
      --num-shards 8 --shard-index "${shard}" > "${log}" 2>&1
  ) &
  pids+=("$!")
  echo "SAM shard=${shard} pid=${pids[${shard}]} log=${log}"
done

status=0
for shard in {0..7}; do
  if ! wait "${pids[${shard}]}"; then
    echo "SAM shard ${shard} failed" >&2
    status=1
  fi
done
exit "${status}"
