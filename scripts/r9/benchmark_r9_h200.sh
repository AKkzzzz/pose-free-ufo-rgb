#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/h200/env_h200_offline.sh"
cd "${ROOT}"
exec "${UFO_PYTHON_BIN}" tools/r9/benchmark_r9_h200.py
