#!/usr/bin/env bash
set -euo pipefail

UFO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UFO_PYTHON_BIN="${UFO_PYTHON_BIN:-/root/miniconda3/envs/dggt_data/bin/python}"
CONFIG="${UFO_ROOT}/configs/experiments/ufo_scene621_rig_pose_free_p21_4090.json"
ANNOTATION="${UFO_ROOT}/data/UFO_paper/scene_list/waymo_train.txt"
CHECKPOINT="${UFO_CHECKPOINT:-${UFO_ROOT}/outputs/scene621_10k/ufo_scene621_from_scratch_10k/checkpoints/ckpt_009999.pth}"
OUTPUT_ROOT="${UFO_RIG_POSE_FREE_ROOT:-${UFO_ROOT}/outputs/scene621_group_meeting/rig_pose_free}"
MANIFEST_ROOT="${OUTPUT_ROOT}/manifests"
POSE_ROOT="${OUTPUT_ROOT}/omega_rig_local"
STARTS=(0 20 40 60 80 100 120 140 160 178)

manifests=()
for start in "${STARTS[@]}"; do
    manifests+=("${MANIFEST_ROOT}/start_$(printf '%03d' "${start}").json")
done

cd "${UFO_ROOT}"
"${UFO_PYTHON_BIN}" tools/check_rig_pose_free_contract.py \
    --manifests "${manifests[@]}" --pose-root "${POSE_ROOT}" \
    --config "${CONFIG}" --annotation-file "${ANNOTATION}" --scene-index 621 \
    --output "${OUTPUT_ROOT}/contract_check_p21.json"

"${UFO_PYTHON_BIN}" tools/render_ufo_long_sequence.py \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
    --annotation_file "${ANNOTATION}" --scene_id 621 \
    --pose_override_mode all --pose_override_sequence_dir "${POSE_ROOT}" \
    --intrinsics_override_mode none --pose_free_camera_only \
    --pose_free_coordinate_mode recurrent \
    --output_dir "${OUTPUT_ROOT}/P21_Omega_rig_only_recurrent" \
    --video_name P21_Omega_rig_only_recurrent_full_scene_render_3cam.mp4

"${UFO_PYTHON_BIN}" tools/summarize_rig_pose_free.py --root "${OUTPUT_ROOT}"
