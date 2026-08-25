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
OUTPUT_ROOT="${UFO_LONG_OUTPUT_ROOT:-${UFO_ROOT}/outputs/scene621_group_meeting/long_sequence}"
MANIFEST_ROOT="${OUTPUT_ROOT}/manifests"
ALL_POSES="${OUTPUT_ROOT}/omega_all"
STARTS=(0 20 40 60 80 100 120 140 160 178)

mkdir -p "${MANIFEST_ROOT}"
manifests=()
for start in "${STARTS[@]}"; do
    manifest="${MANIFEST_ROOT}/start_$(printf '%03d' "${start}").json"
    manifests+=("${manifest}")
    cd "${UFO_ROOT}"
    "${UFO_PYTHON_BIN}" tools/export_ufo_pose_manifest.py \
        --config "${CONFIG}" --data-root "${UFO_ROOT}/data/UFO_paper" \
        --annotation-file "${ANNOTATION}" --scene-index 621 \
        --start-index "${start}" --metadata-only --output "${manifest}"
done

cd "${OMEGA_ROOT}"
"${OMEGA_PYTHON_BIN}" tools/export_ufo_pose_sequence.py \
    --manifests "${manifests[@]}" --checkpoint "${OMEGA_CHECKPOINT}" \
    --output-dir "${ALL_POSES}" --scope all

render_long() {
    local output_name="$1"
    local video_name="$2"
    shift 2
    cd "${UFO_ROOT}"
    "${UFO_PYTHON_BIN}" tools/render_ufo_long_sequence.py \
        --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
        --annotation_file "${ANNOTATION}" --scene_id 621 \
        --output_dir "${OUTPUT_ROOT}/${output_name}" \
        --video_name "${video_name}" "$@"
}

render_long "camera_matrix/E0_GT_T_GT_K" E0_GT_T_GT_K_render_3cam.mp4 \
    --pose_override_mode none --intrinsics_override_mode none
render_long "camera_matrix/E1_Omega_T_GT_K" E1_Omega_T_GT_K_render_3cam.mp4 \
    --pose_override_mode all --pose_override_sequence_dir "${ALL_POSES}" \
    --intrinsics_override_mode none
render_long "camera_matrix/E2_GT_T_Omega_K" E2_GT_T_Omega_K_render_3cam.mp4 \
    --pose_override_mode none --intrinsics_override_mode all \
    --intrinsics_override_sequence_dir "${ALL_POSES}"
render_long "camera_matrix/E3_Omega_T_Omega_K" E3_Omega_T_Omega_K_render_3cam.mp4 \
    --pose_override_mode all --pose_override_sequence_dir "${ALL_POSES}" \
    --intrinsics_override_mode all --intrinsics_override_sequence_dir "${ALL_POSES}"
