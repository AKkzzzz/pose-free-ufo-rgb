# Pose-Free UFO Tools

## Formal camera pipeline

- export_rgb_only_manifest.py
  Build the RGB-only frame manifest.

- export_rgb_only_omega.py
  Run VGGT-Omega and cache predicted pose/K/depth/confidence.

- diagnose_gca_omega_scale_cached.py
  Estimate metric scale with MoGe-2 + the GCA depth-ratio rule.

- export_rgb_metric_pose.py
  Export a per-window rig_local_metric camera override.

- build_global_pose_from_overlaps.py
  Align metric windows by overlapping camera trajectories using SE(3)
  and produce one global_metric trajectory.

- diagnose_gca_window_overlap.py
  Cross-window alignment diagnostic.

- render_ufo_long_sequence.py
  Qualitative stitched long-video renderer.

## Ablations

tools/ablations/

- filter_manifest_context_only.py
- densify_context_pose_override.py
- export_gt_camera_override.py

These are evaluation / camera-ablation utilities, not the formal
training pipeline.

## Dynamic debugging

The renderer/object diagnostic tools are intentionally retained because
the next stage restores bbox-based dynamic modeling without restoring
GT object trajectory teacher forcing.
