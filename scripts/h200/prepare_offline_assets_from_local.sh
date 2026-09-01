#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"

: "${UFO_PREP_VGG16_SOURCE:?Set UFO_PREP_VGG16_SOURCE to the cached VGG16 checkpoint}"
: "${UFO_PREP_LPIPS_SOURCE:?Set UFO_PREP_LPIPS_SOURCE to the cached LPIPS checkpoint}"
: "${UFO_PREP_GSPLAT_SOURCE_TARBALL:?Set UFO_PREP_GSPLAT_SOURCE_TARBALL to the gsplat source archive}"
: "${UFO_PREP_DYNAMIC_POOL_SOURCE:?Set UFO_PREP_DYNAMIC_POOL_SOURCE to the audited D50 pool JSON}"

asset_root="${UFO_ROOT}/offline_assets"
mkdir -p "${asset_root}/weights" "${asset_root}/data_contract" "${asset_root}/manifests"
install -m 0644 "${UFO_PREP_VGG16_SOURCE}" "${asset_root}/weights/vgg16-397923af.pth"
install -m 0644 "${UFO_PREP_LPIPS_SOURCE}" "${asset_root}/weights/lpips_vgg.pth"
install -m 0644 "${UFO_PREP_GSPLAT_SOURCE_TARBALL}" "${asset_root}/gsplat-1.5.3-source.tar.gz"

install -m 0644 "${UFO_DATA_ROOT}/scene_list/waymo_train.txt" "${asset_root}/data_contract/waymo_train.txt"
install -m 0644 "${UFO_DATA_ROOT}/scene_list/waymo_val.txt" "${asset_root}/data_contract/waymo_val.txt"
install -m 0644 "${UFO_DATA_ROOT}/scene_list/waymo_instance_scene_manifest.json" \
  "${asset_root}/data_contract/waymo_instance_scene_manifest.json"
install -m 0644 "${UFO_PREP_DYNAMIC_POOL_SOURCE}" "${asset_root}/data_contract/dynamic_rich_pool.json"

export TORCH_CUDA_ARCH_LIST="8.9;9.0"
export TORCH_EXTENSIONS_DIR="${asset_root}/torch_extensions_portable_v1"
"${UFO_PYTHON_BIN}" scripts/h200/gsplat_forward_backward_smoke.py

sha256sum \
  offline_assets/weights/lpips_vgg.pth \
  offline_assets/weights/vgg16-397923af.pth \
  offline_assets/gsplat-1.5.3-source.tar.gz \
  > offline_assets/manifests/weights.sha256
sha256sum offline_assets/data_contract/* > offline_assets/manifests/data_contract.sha256
sha256sum offline_assets/torch_extensions_portable_v1/gsplat_cuda/gsplat_cuda.so \
  > offline_assets/manifests/gsplat_portable.sha256

bash scripts/h200/package_offline_bundle.sh
echo "OFFLINE_ASSET_PREPARATION=PASS"
