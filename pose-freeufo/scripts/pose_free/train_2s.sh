#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-/root/miniconda3/envs/dggt_data/bin/python}

exec "$PY" main.py \
  --config configs/pose_free/ufo_scene621_pose_free_2s_train.json
