# Offline H200 Assets

This directory contains metadata only. Do not commit pretrained weights, CUDA build outputs, datasets, checkpoints, or container images.

The online 4090 preparation environment must populate a runtime `offline_assets/` directory with:

- local LPIPS and torchvision VGG16 weights;
- gsplat 1.5.3 source and a portable `sm_89;sm_90` build, or enough local toolchain state for an H200-native `sm_90` build;
- the 798/202 Waymo split files, corrected scene-name instance manifest, and optional D50 dynamic-rich pool;
- dependency and toolchain manifests.

Populate the bundle from existing local caches and preprocessed data:

```bash
UFO_PREP_VGG16_SOURCE=/local/cache/vgg16-397923af.pth \
UFO_PREP_LPIPS_SOURCE=/local/cache/lpips_vgg.pth \
UFO_PREP_GSPLAT_SOURCE_TARBALL=/local/cache/gsplat-1.5.3-source.tar.gz \
UFO_PREP_DYNAMIC_POOL_SOURCE=/local/metadata/dynamic_rich_pool.json \
UFO_DATA_ROOT=/mounted/UFO_paper \
bash scripts/h200/prepare_offline_assets_from_local.sh
```

The preparation script does not download anything. Its inputs must already be available on the online preparation host. To revalidate an existing bundle, run:

```bash
bash scripts/h200/package_offline_bundle.sh
```

This validates hashes, confirms LPIPS can initialize with socket access disabled, scans runtime code for download paths, and records the resolved environment. `bootstrap_h200_offline.sh` repeats the checks on the target node and performs real CUDA forward/backward smoke tests.

The binary payload is transferred through the saved container image or shared storage, not GitHub.
