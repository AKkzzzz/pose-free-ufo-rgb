#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:?usage: $0 PID_TO_WAIT_FOR}"
python_bin="${UFO_PYTHON_BIN:-$(command -v python)}"
config="configs/clean_reproduction/ufo_clean_uniform_eb8_10k.json"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 60
done

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 "${python_bin}" main.py \
  --config "${config}" \
  --num_iterations 1 \
  --exp_name clean_repro_uniform_eb8_smoke \
  --skip_initial_validation

CUDA_VISIBLE_DEVICES=0 "${python_bin}" main.py --config "${config}"
