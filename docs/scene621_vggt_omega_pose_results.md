# Scene621 VGGT-Omega Pose Experiment

## Protocol

- Waymo training index: `621`
- Segment: `segment-5846229052615948000_2120_000_2140_000_with_camera_labels`
- UFO model: trained from scratch on scene621 for 10,000 optimizer steps
- Checkpoint: `ckpt_009999.pth` (`latest_step=9999`, no resume source)
- Evaluation window: `start_idx=0`
- Evaluation coverage: 16 target timesteps x 3 cameras over four autoregressive chunks
- UFO input/render resolution: 160 x 240
- VGGT-Omega: 1B/512 checkpoint with orientation-aware GT Sim(3) alignment

E0 uses GT context and target poses. E1 replaces only the 12 context-image poses
with context-only VGGT-Omega predictions. E2 predicts poses jointly from all 60
context and target RGB images and replaces both context and target poses.

## UFO Results

| Experiment | Context pose | Target pose | PSNR (dB) | Delta PSNR | SSIM | Depth RMSE (m) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| E0 | GT | GT | 24.190 | 0.000 | 0.8163 | 3.603 |
| E1 | Omega context-only | GT | 17.837 | -6.353 | 0.3639 | 7.448 |
| E2 | Omega all-frame | Omega all-frame | 22.433 | -1.757 | 0.7311 | 4.011 |

## Pose Diagnostics

| Scope | Images | Rotation mean (deg) | ATE RMSE (m) | RPE rotation mean (deg) | Peak GPU (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Context-only | 12 | 1.387 | 0.0727 | 0.0632 | 7.39 |
| All-frame | 60 | 1.600 | 0.0783 | 0.0493 | 10.40 |

The context-only trajectory is nearly straight, so center-only Umeyama alignment
cannot determine global rotation reliably. The exporter uses camera orientations to
fit global rotation, then camera centers to fit scale and translation.

E1 is the clean pose-sensitivity result: target RGB and target pose are not used by
VGGT-Omega. E2 recovers most of the E1 loss when context and target use a consistent
Omega pose system, but it is diagnostic rather than pose-free NVS because target RGB
and target GT poses participate in Omega inference and GT Sim(3) alignment.
