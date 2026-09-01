#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"
mode="${1:-d50}"
resume="${2:-latest}"
source "${UFO_OUTPUT_ROOT}/h200_smoke/recommended.env"
case "${mode}" in
  d50) config=configs/h200/ufo_2s_d50_global64.json; exp=ufo_h200_2s_d50_global64; sampling=(--dynamic_rich_pool "${UFO_DYNAMIC_POOL}") ;;
  uniform) config=configs/h200/ufo_2s_uniform_global64.json; exp=ufo_h200_2s_uniform_global64; sampling=() ;;
  *) echo "mode must be d50 or uniform" >&2; exit 2 ;;
esac
args=()
if [[ "${resume}" != "latest" && "${resume}" != "none" ]]; then args+=(--resume_from "${resume}"); fi
"${UFO_TORCHRUN_BIN}" --standalone --nproc_per_node=8 main.py --config "${config}" \
  --exp_name "${exp}" --output_dir "${UFO_OUTPUT_ROOT}" --data_root "${UFO_DATA_ROOT}" \
  --batch_size "${H200_BATCH_SIZE}" --gradient_accumulation_steps "${H200_ACCUMULATION_STEPS}" \
  --ddp_accumulation_no_sync "${sampling[@]}" "${args[@]}"

run_dir="${UFO_OUTPUT_ROOT}/h200_reproduction/${exp}"
artifact_dir="${UFO_OUTPUT_ROOT}/h200_artifacts/ufo_2s_${mode}"
"${UFO_PYTHON_BIN}" scripts/h200/freeze_checkpoint_lineage.py \
  --stage 2s --run-dir "${run_dir}" --artifact-dir "${artifact_dir}" \
  --expected-steps 100000
echo "Frozen 2s best/last checkpoints: ${artifact_dir}"
