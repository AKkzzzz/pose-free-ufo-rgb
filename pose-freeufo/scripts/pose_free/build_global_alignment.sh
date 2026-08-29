#!/usr/bin/env bash
set -euo pipefail

SCENE=segment-5846229052615948000_2120_000_2140_000_with_camera_labels

python tools/build_global_pose_from_overlaps.py \
  --input-root outputs/pose_free_camera/omega_gca_metric_all \
  --scene "$SCENE" \
  --first 0 \
  --last 178 \
  --output-root outputs/pose_free_camera/global_aligned
