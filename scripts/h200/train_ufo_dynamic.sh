#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"
mode="${1:-paper}"
resume="${2:-latest}"
source "${UFO_OUTPUT_ROOT}/h200_smoke/recommended.env"
case "${mode}" in paper) config=configs/h200/ufo_dynamic_paper.json ;; anchor) config=configs/h200/ufo_dynamic_anchor.json ;; *) exit 2 ;; esac
args=()
if [[ "${resume}" != "latest" && "${resume}" != "none" ]]; then args+=(--resume_from "${resume}"); fi
"${UFO_TORCHRUN_BIN}" --standalone --nproc_per_node=8 main.py --config "${config}" \
  --output_dir "${UFO_OUTPUT_ROOT}" --data_root "${UFO_DATA_ROOT}" \
  --batch_size "${H200_BATCH_SIZE}" --gradient_accumulation_steps "${H200_ACCUMULATION_STEPS}" \
  --ddp_accumulation_no_sync "${args[@]}"
