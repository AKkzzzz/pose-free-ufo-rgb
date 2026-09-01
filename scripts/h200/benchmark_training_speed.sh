#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"
exec "${UFO_PYTHON_BIN}" scripts/h200/benchmark_training_speed.py
