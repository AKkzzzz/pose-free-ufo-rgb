#!/usr/bin/env bash
set -euo pipefail

UFO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OMEGA_ROOT="${OMEGA_ROOT:-$(cd "${UFO_ROOT}/.." && pwd)/vggt-omega}"
UFO_PYTHON_BIN="${UFO_PYTHON_BIN:-/root/miniconda3/envs/dggt_data/bin/python}"
OMEGA_PYTHON_BIN="${OMEGA_PYTHON_BIN:-${UFO_PYTHON_BIN}}"

CONFIG="${UFO_ROOT}/configs/experiments/ufo_scene621_10k_4090.json"
ANNOTATION="${UFO_ROOT}/data/UFO_paper/scene_list/waymo_train.txt"
CHECKPOINT="${UFO_CHECKPOINT:-${UFO_ROOT}/outputs/scene621_10k/ufo_scene621_from_scratch_10k/checkpoints/ckpt_009999.pth}"
OMEGA_CHECKPOINT="${OMEGA_CHECKPOINT:-${OMEGA_ROOT}/checkpoints/vggt_omega_1b_512.pt}"
EXPERIMENT_ROOT="${UFO_EXPERIMENT_ROOT:-${UFO_ROOT}/outputs/vggt_omega_pose/scene621_10k}"
MANIFEST="${EXPERIMENT_ROOT}/scene621_manifest.json"
CONTEXT_POSES="${EXPERIMENT_ROOT}/omega_context"
ALL_POSES="${EXPERIMENT_ROOT}/omega_all"

for required_file in "${CONFIG}" "${ANNOTATION}" "${CHECKPOINT}" "${OMEGA_CHECKPOINT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

mkdir -p "${EXPERIMENT_ROOT}"

cd "${UFO_ROOT}"
"${UFO_PYTHON_BIN}" tools/export_ufo_pose_manifest.py \
    --config "${CONFIG}" \
    --data-root "${UFO_ROOT}/data/UFO_paper" \
    --annotation-file "${ANNOTATION}" \
    --scene-index 621 \
    --start-index 0 \
    --output "${MANIFEST}"

cd "${OMEGA_ROOT}"
"${OMEGA_PYTHON_BIN}" tools/export_ufo_pose_override.py \
    --manifest "${MANIFEST}" \
    --checkpoint "${OMEGA_CHECKPOINT}" \
    --output-dir "${CONTEXT_POSES}" \
    --scope context
"${OMEGA_PYTHON_BIN}" tools/export_ufo_pose_override.py \
    --manifest "${MANIFEST}" \
    --checkpoint "${OMEGA_CHECKPOINT}" \
    --output-dir "${ALL_POSES}" \
    --scope all

run_ufo_inference() {
    local name="$1"
    shift
    cd "${UFO_ROOT}"
    "${UFO_PYTHON_BIN}" inference.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --annotation_file "${ANNOTATION}" \
        --scene_id 621 \
        --start_idx 0 \
        --output_dir "${EXPERIMENT_ROOT}/${name}" \
        "$@"
}

run_ufo_inference e0_gt --pose_override_mode none
run_ufo_inference e1_omega_context \
    --pose_override_mode context --pose_override_dir "${CONTEXT_POSES}"
run_ufo_inference e2_omega_all \
    --pose_override_mode all --pose_override_dir "${ALL_POSES}"

"${UFO_PYTHON_BIN}" tools/summarize_pose_experiment.py \
    --experiment-root "${EXPERIMENT_ROOT}" \
    --output "${EXPERIMENT_ROOT}/summary.json"
