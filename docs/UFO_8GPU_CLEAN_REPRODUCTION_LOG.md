# UFO 8-GPU Clean Reproduction Log

## Run

- Host: `yx-ufo-8ka--b3013d38a94d-gwvdw7nhic`
- Hardware: 8 x NVIDIA RTX 4090, 49,140 MiB each (the host does not expose H200)
- Code commit: `306756a`
- Train split: official Waymo `798` scenes
- Validation split: official Waymo `202` scenes
- Corrected manifest: `data/UFO_paper/scene_list/waymo_instance_scene_manifest.json`
- Main run: `clean_repro_d50_global64_10k_run1`
- Config: `configs/clean_reproduction/ufo_clean_d50_global64_10k.json`
- Command: `torchrun --standalone --nproc_per_node=8 main.py --config configs/clean_reproduction/ufo_clean_d50_global64_10k.json --exp_name clean_repro_d50_global64_10k_run1 --skip_initial_validation`

## DDP Contract

- One DDP model replica per GPU; `world_size=8`.
- Per-GPU batch `1`, accumulation `8`, effective global batch `64`.
- DDP wraps the model before optimizer construction; gradients are synchronized by backward all-reduce.
- Rank-specific D50 sampler streams use `seed + rank`; rich/uniform windows are not intentionally duplicated across ranks.
- Only rank 0 writes checkpoints from `model.module.state_dict()`, followed by a process barrier.
- D50 is an engineering reproduction decision: 50% audited dynamic-rich windows and 50% uniform samples from the full 798-scene train split.

## Smoke

- 20 optimizer steps completed on all 8 ranks.
- 160 microbatches, finite losses and gradients, no NCCL/DDP hang or OOM.
- Peak memory about 21.0 GiB per GPU; steady throughput about 1.92 s/microbatch.
- Observed dynamic pixel ratio about 7.2%; rank-specific sampling is active.

## Formal Gate

- Formal run started from random initialization with no checkpoint.
- Current run directory: `outputs/clean_reproduction/clean_repro_d50_global64_10k_run1`
- Scheduled validation: optimizer steps `1000,3000,5000,7500,10000` on the official validation split.
- Warmup: 5,000 optimizer steps to `1e-4`; LPIPS activates at step 5,000.
- Flow and flex-attention remain `MISSING_PAPER_COMPONENT`; no diagnostic vehicle losses are enabled.
- Capability decisions are deferred until the 5k to 10k trend.
