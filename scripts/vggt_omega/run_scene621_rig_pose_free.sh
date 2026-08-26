#!/usr/bin/env bash
set -euo pipefail

UFO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OMEGA_ROOT="${OMEGA_ROOT:-$(cd "${UFO_ROOT}/.." && pwd)/vggt-omega}"
UFO_PYTHON_BIN="${UFO_PYTHON_BIN:-/root/miniconda3/envs/dggt_data/bin/python}"
OMEGA_PYTHON_BIN="${OMEGA_PYTHON_BIN:-${UFO_PYTHON_BIN}}"
BASE_CONFIG="${UFO_ROOT}/configs/experiments/ufo_scene621_10k_4090.json"
P2_CONFIG="${UFO_ROOT}/configs/experiments/ufo_scene621_rig_pose_free_4090.json"
ANNOTATION="${UFO_ROOT}/data/UFO_paper/scene_list/waymo_train.txt"
CHECKPOINT="${UFO_CHECKPOINT:-${UFO_ROOT}/outputs/scene621_10k/ufo_scene621_from_scratch_10k/checkpoints/ckpt_009999.pth}"
OMEGA_CHECKPOINT="${OMEGA_CHECKPOINT:-${OMEGA_ROOT}/checkpoints/vggt_omega_1b_512.pt}"
OUTPUT_ROOT="${UFO_RIG_POSE_FREE_ROOT:-${UFO_ROOT}/outputs/scene621_group_meeting/rig_pose_free}"
MANIFEST_ROOT="${OUTPUT_ROOT}/manifests"
POSE_ROOT="${OUTPUT_ROOT}/omega_rig_local"
STARTS=(0 20 40 60 80 100 120 140 160 178)

mkdir -p "${MANIFEST_ROOT}"
manifests=()
for start in "${STARTS[@]}"; do
    manifest="${MANIFEST_ROOT}/start_$(printf '%03d' "${start}").json"
    manifests+=("${manifest}")
    "${UFO_PYTHON_BIN}" "${UFO_ROOT}/tools/export_ufo_pose_manifest.py" \
        --config "${BASE_CONFIG}" --data-root "${UFO_ROOT}/data/UFO_paper" \
        --annotation-file "${ANNOTATION}" --scene-index 621 \
        --start-index "${start}" --metadata-only --rig-pose-free --output "${manifest}"
done

cd "${OMEGA_ROOT}"
"${OMEGA_PYTHON_BIN}" tools/export_ufo_rig_pose_free_sequence.py \
    --manifests "${manifests[@]}" --checkpoint "${OMEGA_CHECKPOINT}" \
    --output-dir "${POSE_ROOT}"

cd "${UFO_ROOT}"
"${UFO_PYTHON_BIN}" tools/check_rig_pose_free_contract.py \
    --manifests "${manifests[@]}" --pose-root "${POSE_ROOT}" \
    --config "${P2_CONFIG}" --annotation-file "${ANNOTATION}" --scene-index 621 \
    --output "${OUTPUT_ROOT}/contract_check.json"

"${UFO_PYTHON_BIN}" tools/render_ufo_long_sequence.py \
    --config "${P2_CONFIG}" --checkpoint "${CHECKPOINT}" \
    --annotation_file "${ANNOTATION}" --scene_id 621 \
    --pose_override_mode all --pose_override_sequence_dir "${POSE_ROOT}" \
    --intrinsics_override_mode none --pose_free_camera_only \
    --output_dir "${OUTPUT_ROOT}/P2_Omega_rig_only" \
    --video_name P2_Omega_rig_only_full_scene_render_3cam.mp4
