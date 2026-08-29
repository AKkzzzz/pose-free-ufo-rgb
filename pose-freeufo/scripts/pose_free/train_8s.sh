#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-/root/miniconda3/envs/dggt_data/bin/python}
CKPT=${1:-outputs/pose_free_train/GCA_Omega_uniform_10k/checkpoints/ckpt_009999.pth}

test -f "$CKPT" || {
  echo "Missing 2s checkpoint: $CKPT"
  exit 1
}

exec "$PY" main.py \
  --config configs/pose_free/ufo_scene621_pose_free_8s_finetune.json \
  --load_from "$CKPT"
