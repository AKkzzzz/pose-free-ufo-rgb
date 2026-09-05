#!/usr/bin/env bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export UFO_ROOT="${UFO_ROOT:-${ROOT_DIR}}"
if [[ -z "${UFO_PYTHON_BIN:-}" ]]; then
  conda_python=""
  if command -v conda >/dev/null; then
    conda_prefix="$(conda env list 2>/dev/null | awk '$1 == "dggt_data" {print $NF; exit}')"
    if [[ -n "${conda_prefix}" ]]; then
      conda_python="${conda_prefix}/bin/python"
    fi
  fi
  for candidate in \
    "${conda_python}" \
    "${CONDA_PREFIX:-}/bin/python" \
    "$(command -v python)"; do
    if [[ -x "${candidate}" ]] && "${candidate}" -c 'import torch, gsplat' >/dev/null 2>&1; then
      UFO_PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
test -n "${UFO_PYTHON_BIN:-}" || { echo "No offline Python with torch and gsplat found" >&2; return 1; }
export UFO_PYTHON_BIN
export UFO_TORCHRUN_BIN="${UFO_TORCHRUN_BIN:-$(dirname "${UFO_PYTHON_BIN}")/torchrun}"
test -x "${UFO_TORCHRUN_BIN}" || { echo "Missing torchrun: ${UFO_TORCHRUN_BIN}" >&2; return 1; }
export UFO_EXPECTED_TORCH="${UFO_EXPECTED_TORCH:-2.10.0+cu128}"
export UFO_EXPECTED_TORCH_CUDA="${UFO_EXPECTED_TORCH_CUDA:-12.8}"
export UFO_OFFLINE=1
export HF_HOME="${HF_HOME:-${UFO_ROOT}/third_party/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
# Keep the prebuilt cache portable. The H200-only fallback uses a separate
# extension directory and overrides this to 9.0 in bootstrap_h200_offline.sh.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;9.0}"
export TORCH_HOME="${TORCH_HOME:-${UFO_ROOT}/offline_assets/torch_home}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${UFO_ROOT}/offline_assets/torch_extensions_portable_v1}"
export UFO_VGG16_WEIGHTS="${UFO_VGG16_WEIGHTS:-${UFO_ROOT}/offline_assets/weights/vgg16-397923af.pth}"
export UFO_LPIPS_CKPT="${UFO_LPIPS_CKPT:-${UFO_ROOT}/offline_assets/weights/lpips_vgg.pth}"
export UFO_DATA_ROOT="${UFO_DATA_ROOT:-${UFO_ROOT}/data/UFO_paper}"
export UFO_OUTPUT_ROOT="${UFO_OUTPUT_ROOT:-${UFO_ROOT}/outputs}"
export UFO_DYNAMIC_POOL="${UFO_DYNAMIC_POOL:-${UFO_ROOT}/offline_assets/data_contract/dynamic_rich_pool.json}"
export PYTHONNOUSERSITE=1
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "${TORCH_HOME}/hub/checkpoints" "${TORCH_EXTENSIONS_DIR}" "${UFO_OUTPUT_ROOT}"
