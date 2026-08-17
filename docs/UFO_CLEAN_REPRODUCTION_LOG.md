# UFO Clean Reproduction Log

## Identity

- Base: official UFO commit `85e4223ffd07badb2dbb6b9bce1815764ccdb82e`
- Branch: `ufo_clean_reproduction_v1`
- Run: `clean_repro_uniform_eb8_seed1`
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
| 0 | pending | pending | pending | pending | initialization |
| 1,000 | pending | pending | pending | pending | health check only |
| 3,000 | pending | pending | pending | pending | health check only |
| 5,000 | pending | pending | pending | pending | warmup boundary |
| 7,500 | pending | pending | pending | pending | trend |
| 10,000 | pending | pending | pending | pending | first capability gate |

Launch state: queued behind the independent post-warmup diagnostic on the only
available GPU. The queue runs one exact-EB8 random-initialization optimizer-step
smoke, then starts the 10k command only if the smoke exits successfully.

Gate: if 5k to 10k PSNR is still rising materially, continue to 25k without
structural changes. Only a clear plateau far below the paper result reopens the
technical-review findings.
