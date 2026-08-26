#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"
mode="${1:-d50}"
resume="${2:-latest}"
source "${UFO_OUTPUT_ROOT}/h200_smoke/recommended.env"
sampling=()
if [[ "${mode}" == "d50" ]]; then
  echo "D50 is a 2s reproduction decision; 8s fine-tuning uses the official uniform split."
fi
args=()
exp="ufo_h200_8s_from_${mode}_global64"
run_dir="${UFO_OUTPUT_ROOT}/h200_reproduction/${exp}"
eight_latest="${run_dir}/checkpoints/latest.pth"
init_variant="${UFO_2S_INIT_VARIANT:-best}"
case "${init_variant}" in best|last) ;; *) echo "UFO_2S_INIT_VARIANT must be best or last" >&2; exit 2 ;; esac
two_frozen="${UFO_OUTPUT_ROOT}/h200_artifacts/ufo_2s_${mode}/${init_variant}.pth"
test -f "${two_frozen}" || { echo "Missing frozen 2s checkpoint: ${two_frozen}" >&2; exit 1; }
if [[ "${resume}" != "latest" && "${resume}" != "none" ]]; then
  args+=(--resume_from "${resume}")
elif [[ ! -e "${eight_latest}" ]]; then
  args+=(--load_from "${two_frozen}")
fi
"${UFO_TORCHRUN_BIN}" --standalone --nproc_per_node=8 main.py --config configs/h200/ufo_8s_finetune_global64.json \
  --exp_name "${exp}" \
  --output_dir "${UFO_OUTPUT_ROOT}" --data_root "${UFO_DATA_ROOT}" \
  --batch_size "${H200_BATCH_SIZE}" --gradient_accumulation_steps "${H200_ACCUMULATION_STEPS}" \
  --ddp_accumulation_no_sync "${sampling[@]}" "${args[@]}"

artifact_dir="${UFO_OUTPUT_ROOT}/h200_artifacts/ufo_8s_from_${mode}"
"${UFO_PYTHON_BIN}" scripts/h200/freeze_checkpoint_lineage.py \
  --stage 8s --run-dir "${run_dir}" --artifact-dir "${artifact_dir}" \
  --expected-steps 50000 --parent-checkpoint "${two_frozen}"
echo "Frozen 8s best/last checkpoints: ${artifact_dir}"
