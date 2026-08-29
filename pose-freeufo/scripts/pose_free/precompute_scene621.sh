#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/project/intelligent-driving-agent/guoluosong-253108120129/workspace/yx/ufoposefree
UFO=$ROOT/pose-freeufo
DGGTPY=/root/miniconda3/envs/dggt_data/bin/python

CONFIG=$UFO/configs/pose_free/ufo_scene621_pose_free_2s_train.json
ANNOTATION=$UFO/data/UFO_paper/scene_list/waymo_train.txt

SCENE=segment-5846229052615948000_2120_000_2140_000_with_camera_labels

FIRST=${1:-0}
LAST=${2:-178}

cd "$UFO"
source "$ROOT/.venv-moge2/bin/activate"

for START in $(seq "$FIRST" "$LAST"); do
    S=$(printf "%03d" "$START")

    MANIFEST="outputs/pose_free_camera/manifests_rgb_only/start_${S}.json"
    RAW="outputs/pose_free_camera/omega_raw_all/start_${S}.npz"
    SCALE="outputs/pose_free_camera/gca_omega_scale_all/start_${S}.json"
    FINAL="outputs/pose_free_camera/omega_gca_metric_all/start_${S}/${SCENE}/omega_pose_override.npz"

    echo
    echo "========== START ${S} =========="

    if [[ -f "$FINAL" ]]; then
        echo "[SKIP] final exists"
        continue
    fi

    if [[ ! -f "$MANIFEST" ]]; then
        "$DGGTPY" tools/export_rgb_only_manifest.py \
          --config "$CONFIG" \
          --data-root "$UFO/data/UFO_paper" \
          --annotation-file "$ANNOTATION" \
          --scene-index 621 \
          --start-index "$START" \
          --output "$MANIFEST"
    fi

    if [[ ! -f "$RAW" ]]; then
        python tools/export_rgb_only_omega.py \
          --manifest "$MANIFEST" \
          --omega-repo "$ROOT/vggt-omega" \
          --checkpoint "$ROOT/vggt-omega/checkpoints/vggt_omega_1b_512.pt" \
          --output "$RAW"
    fi

    if [[ ! -f "$SCALE" ]]; then
        python tools/diagnose_gca_omega_scale_cached.py \
          --manifest "$MANIFEST" \
          --omega-npz "$RAW" \
          --moge-repo "$ROOT/moge" \
          --moge-model "$ROOT/checkpoints/moge-2-vitl/model.pt" \
          --output "$SCALE"
    fi

    python tools/export_rgb_metric_pose.py \
      --manifest "$MANIFEST" \
      --omega-npz "$RAW" \
      --scale-json "$SCALE" \
      --output "$FINAL"

done
