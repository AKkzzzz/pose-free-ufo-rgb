#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"
bash scripts/h200/audit_runtime_network.sh
"${UFO_PYTHON_BIN}" scripts/h200/offline_readiness.py
{
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_diff_sha256=$(git diff --binary | sha256sum | awk '{print $1}')"
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python=$(${UFO_PYTHON_BIN} --version 2>&1)"
  echo "torch=$(${UFO_PYTHON_BIN} -c 'import torch; print(torch.__version__)')"
  echo "cuda=$(${UFO_PYTHON_BIN} -c 'import torch; print(torch.version.cuda)')"
} > offline_assets/manifests/offline_bundle.env
"${UFO_PYTHON_BIN}" -m pip freeze > offline_assets/manifests/pip-freeze.txt
if command -v conda >/dev/null; then
  conda list --prefix "$(dirname "$(dirname "${UFO_PYTHON_BIN}")")" --explicit \
    > offline_assets/manifests/conda-explicit.txt
fi
{
  command -v nvcc >/dev/null && nvcc --version || true
  command -v gcc >/dev/null && gcc --version | head -1 || true
  command -v g++ >/dev/null && g++ --version | head -1 || true
  command -v ninja >/dev/null && ninja --version || true
} > offline_assets/manifests/system-toolchain.txt
manifest_tmp="$(mktemp)"
find configs/h200 scripts/h200 offline_assets/manifests -type f \
  ! -name reproduction_files.sha256 -print0 \
  | sort -z | xargs -0 sha256sum > "${manifest_tmp}"
mv "${manifest_tmp}" offline_assets/manifests/reproduction_files.sha256
echo "OFFLINE_BUNDLE_READY=YES"
