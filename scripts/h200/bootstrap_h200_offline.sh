#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"

echo "git_commit=$(git rev-parse HEAD)"
echo "git_status=$(git status --short | tr '\n' ';')"
echo "hostname=$(hostname)"
echo "container_id=$(cat /etc/hostname 2>/dev/null || true)"
nvidia-smi -L
nvcc --version
"${UFO_PYTHON_BIN}" - <<'PY'
import json, os, torch
if torch.__version__ != os.environ["UFO_EXPECTED_TORCH"]:
    raise RuntimeError(f"PyTorch mismatch: {torch.__version__} != {os.environ['UFO_EXPECTED_TORCH']}")
if torch.version.cuda != os.environ["UFO_EXPECTED_TORCH_CUDA"]:
    raise RuntimeError(f"torch CUDA mismatch: {torch.version.cuda} != {os.environ['UFO_EXPECTED_TORCH_CUDA']}")
print(json.dumps({
    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "nccl": torch.cuda.nccl.version() if torch.cuda.is_available() else None,
    "device_count": torch.cuda.device_count(),
    "devices": [{"name": torch.cuda.get_device_name(i), "capability": torch.cuda.get_device_capability(i)} for i in range(torch.cuda.device_count())],
    "arch_list": torch.cuda.get_arch_list(),
}, indent=2))
PY

"${UFO_PYTHON_BIN}" scripts/h200/offline_readiness.py
bash scripts/h200/audit_runtime_network.sh
test "$(wc -l < "${UFO_DATA_ROOT}/scene_list/waymo_train.txt")" -eq 798
test "$(wc -l < "${UFO_DATA_ROOT}/scene_list/waymo_val.txt")" -eq 202
test -f "${UFO_DATA_ROOT}/scene_list/waymo_instance_scene_manifest.json"
cmp offline_assets/data_contract/waymo_train.txt "${UFO_DATA_ROOT}/scene_list/waymo_train.txt"
cmp offline_assets/data_contract/waymo_val.txt "${UFO_DATA_ROOT}/scene_list/waymo_val.txt"
cmp offline_assets/data_contract/waymo_instance_scene_manifest.json \
  "${UFO_DATA_ROOT}/scene_list/waymo_instance_scene_manifest.json"

if [[ "${UFO_ALLOW_NON_H200_SMOKE:-0}" != "1" ]]; then
  "${UFO_PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
for index in range(8):
    assert torch.cuda.get_device_capability(index) == (9, 0)
    assert "H200" in torch.cuda.get_device_name(index).upper()
PY
fi

test -f "${TORCH_EXTENSIONS_DIR}/gsplat_cuda/gsplat_cuda.so"
if command -v cuobjdump >/dev/null; then
  cuobjdump --list-elf "${TORCH_EXTENSIONS_DIR}/gsplat_cuda/gsplat_cuda.so" | grep 'sm_90.cubin' >/dev/null
fi
if ! "${UFO_PYTHON_BIN}" scripts/h200/gsplat_forward_backward_smoke.py; then
  export TORCH_EXTENSIONS_DIR="${UFO_ROOT}/offline_assets/torch_extensions_h200_native"
  mkdir -p "${TORCH_EXTENSIONS_DIR}"
  export TORCH_CUDA_ARCH_LIST="9.0"
  "${UFO_PYTHON_BIN}" scripts/h200/gsplat_forward_backward_smoke.py
fi
if [[ "${UFO_ALLOW_NON_H200_SMOKE:-0}" == "1" ]]; then
  echo "NON_H200_OFFLINE_SIMULATION=PASS"
  echo "H200_ENV_READY=NO"
else
  UFO_SMOKE_WORLD_SIZE=1 bash scripts/h200/offline_ufo_train_smoke.sh
  echo "H200_ENV_READY=YES"
fi
