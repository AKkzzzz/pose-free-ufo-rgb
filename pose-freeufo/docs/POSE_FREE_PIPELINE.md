# Pose-Free UFO Current Pipeline

## 1. Formal 2s camera pipeline

RGB
→ VGGT-Omega
→ relative pose / depth / intrinsics
→ MoGe-2 metric depth
→ GCA median depth-ratio scale
→ rig_local_metric camera trajectory
→ UFO recurrent reconstruction

No GT camera pose / intrinsics / depth / flow / ground / sky /
dynamic mask / object trajectory is used.

Current formal config:

configs/pose_free/ufo_scene621_pose_free_2s_train.json

## 2. Camera ablation

| Camera | PSNR | SSIM |
|---|---:|---:|
| GT pose + GT K | 26.138 | 0.8422 |
| Context-only Omega + GCA | 25.102 | 0.7953 |
| All-RGB Omega + GCA | 25.365 | 0.8104 |

Context-only is only 1.036 dB below GT camera.

All-RGB versus context-only differs by only 0.263 dB, showing that
target-assisted camera estimation is not the main reason for the
reconstruction quality.

## 3. 2s -> long sequence camera alignment

Each 2s window first obtains metric scale independently using GCA.

Cross-window alignment then uses overlapping camera trajectories:

local metric window
→ overlap trajectory
→ robust SE(3) alignment
→ global_metric trajectory

Important:

GCA determines scale.
Cross-window alignment does NOT change scale.
Sim3 scale is diagnostic only.

Validated scene621 global trajectory:

- 198 frames
- 594 camera poses
- median overlap translation residual ≈ 1.34 cm
- median overlap rotation residual ≈ 0.066 deg

Output:

outputs/pose_free_camera/global_aligned/<scene>/omega_pose_override.npz

## 4. Formal 8s route

2s checkpoint
→ load_from
→ 16 recurrent 0.5s chunks
→ 8s recurrent fine-tuning
→ unified global_metric camera trajectory

Config:

configs/pose_free/ufo_scene621_pose_free_8s_finetune.json

## 5. 20s video note

The current render_ufo_long_sequence.py result is a stitched qualitative
video made from independent 2s reconstructions.

It is NOT yet a single continuous 20s recurrent state.

Continuous long-sequence evaluation should be done after the 8s
fine-tuning stage.
