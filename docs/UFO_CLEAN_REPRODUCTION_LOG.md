# UFO Clean Reproduction Log

## Identity

- Base: official UFO commit `85e4223ffd07badb2dbb6b9bce1815764ccdb82e`
- Branch: `ufo_clean_reproduction_v1`
- Run: `clean_repro_uniform_eb8_seed1`
- Training commit: `5199f7e21e9cbf34f3cfe140def40273e34f2f4b`
- Config: `configs/clean_reproduction/ufo_clean_uniform_eb8_10k.json`
- Data: Waymo train/validation scene lists plus corrected scene-name instance manifest
- Target paper metric (2s): 27.26 PSNR / 0.825 SSIM / 5.45 m D-RMSE

## Training Contract

- Random initialization; no old checkpoint is loaded.
- Uniform training sampler. D50 is not part of this baseline.
- Resource-adapted effective batch 8: one GPU x batch 1 x accumulation 8.
- Each optimizer step consumes 8 sequences and 32 recurrent chunks.
- Per-chunk render/loss sum and one backward. Old scene input is detached by
  `allow_old_scene_grad=false`; blanket post-chunk detach remains off because it
  duplicates that boundary and breaks reentrant checkpoint/DDP parameter hooks.
- Official RGB re-encoding for visible old scene input.
- AdamW, 5000 optimizer-step linear warmup to 1e-4, then constant 1e-4.
- LPIPS weight 0.05 activates at optimizer step 5000.
- Depth weight 1.0 with target-max normalization.
- Forward flow and the paper flex-attention mask remain `MISSING_PAPER_COMPONENT`.

## Validation

| Optimizer step | PSNR | SSIM | D-RMSE (m) | Delta PSNR to paper | Trend / gate |
|---:|---:|---:|---:|---:|---|
| 0 | 10.1248 | 0.3270 | 172.87 | -17.1352 | random initialization |
| 1,000 | 18.6629 | 0.5019 | 12.61 | -8.5971 | healthy RGB/depth recovery |
| 3,000 | pending | pending | pending | pending | health check only |
| 5,000 | pending | pending | pending | pending | warmup boundary |
| 7,500 | pending | pending | pending | pending | trend |
| 10,000 | pending | pending | pending | pending | first capability gate |

Launch state: running on the only available RTX 4090 (PID `1010217`). The exact
EB8 random-initialization smoke passed with finite loss and nonzero Transformer,
Gaussian, bbox, and affine gradients. Steady-state training is about 1.9-2.1
seconds per microbatch with about 21 GB peak GPU memory. The run manifest records
a clean worktree and hashes the train list, validation list, and corrected
scene-name instance manifest.

Gate: if 5k to 10k PSNR is still rising materially, continue to 25k without
structural changes. Only a clear plateau far below the paper result reopens the
technical-review findings.

At optimizer step 2175, the run remains finite and reaches about 20-21 dB on
recent training batches. The uniform baseline's object assignment has collapsed
to background (`pred_dynamic_ratio=0`, foreground recall 0). This is recorded as
the expected uniform-exposure risk; it is not used to alter the main baseline
before the scheduled 3k/5k gates.
