# UFO Reproduction

Private clean-room reproduction engineering for [Xiaomi UFO](https://github.com/xiaomi-research/ufo), based on upstream commit:

```text
85e4223ffd07badb2dbb6b9bce1815764ccdb82e
```

This repository keeps the reproducible code, configurations, tests, audit notes, and offline H200 launch tooling. Waymo data, model checkpoints, pretrained weights, compiled CUDA artifacts, and training outputs are deliberately excluded from Git.

## Status

- Full recurrent reconstruction: 2s training, 8s fine-tuning, and 16s zero-shot evaluation are configured.
- Official data contract: 798 Waymo training segments and 202 validation segments, using the three forward cameras at 160x240.
- `uniform` is the paper-like sampling control.
- `D50` is a `REPRODUCTION_DECISION`: 50% dynamic-rich and 50% uniform sampling over the complete training split.
- Forward-flow regularization and the unpublished flex-attention mask remain `MISSING_PAPER_COMPONENT`.
- Public-v1 dense attention and RGB re-encoding are the frozen fallback decisions where the paper does not expose sufficient implementation detail.

## Repository Layout

```text
ufo/                 model, dataset, rendering, and recurrent update code
configs/             clean 4090 and H200 reproduction configurations
scripts/h200/        offline bootstrap, DDP smoke, train, resume, and evaluation
tests/                tensor, data-contract, and reproduction regression tests
docs/                 concise run records and reproduction decisions
offline/              manifests and instructions; no binary assets
```

## Data

Mount preprocessed Waymo data outside the repository and set `UFO_DATA_ROOT`. Formal runs require exactly:

```text
798 train scenes
202 validation scenes
corrected scene-name instance manifest
```

Debug clips and curated formation subsets must not be used by formal reproduction configs.

## 4090 Entry Points

Single-GPU uniform control:

```bash
python main.py --config configs/clean_reproduction/ufo_clean_uniform_eb8_10k.json
```

Eight-GPU D50 run:

```bash
torchrun --standalone --nproc_per_node=8 main.py \
  --config configs/clean_reproduction/ufo_clean_d50_global64_10k.json
```

## Offline H200 Entry Point

Prepare all assets on an online machine as documented in [offline/README.md](offline/README.md), save them in the container or mounted offline bundle, then on the H200 node run:

```bash
UFO_DATA_ROOT=/mounted/UFO_paper \
UFO_OUTPUT_ROOT=/mounted/outputs \
bash scripts/h200/run_ufo_h200.sh --task 2s --mode d50 --resume latest
```

The entry point verifies eight H200 GPUs, compute capability 9.0, pinned PyTorch/CUDA versions, local weights, dataset manifests, gsplat forward/backward, DDP rank synchronization, and effective global batch 64 before training.

Available tasks:

```text
--task 2s       100k optimizer-step recurrent reconstruction
--task 8s       50k fine-tune initialized from a frozen 2s checkpoint
--task 16s      zero-shot evaluation of a frozen 8s checkpoint
--task dynamic  independent dynamic-only paper experiment
--task all      run the stages sequentially
```

## Checkpoint Lineage

The 2s and 8s stages have independent run and artifact directories. Completion creates read-only, hashed artifacts:

```text
h200_artifacts/ufo_2s_<mode>/best.pth
h200_artifacts/ufo_2s_<mode>/last.pth
h200_artifacts/ufo_8s_from_<mode>/best.pth
h200_artifacts/ufo_8s_from_<mode>/last.pth
h200_reproduction/ufo_h200_16s_from_<mode>_<variant>_zeroshot_eval/
```

An incomplete 2s or 8s run cannot be frozen. The 8s stage only reads a frozen 2s artifact and never writes into its directory. The 16s stage only evaluates a frozen 8s artifact.

## Offline Assets

Large assets are not stored in Git. Expected names, paths, and SHA256 values are recorded in [offline/offline_assets_manifest.json](offline/offline_assets_manifest.json). The runtime scripts fail fast instead of downloading missing files.

## License

The upstream project is released under CC BY-NC 4.0. See [LICENSE](LICENSE).
