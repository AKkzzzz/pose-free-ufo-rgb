#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/h200/env_h200_offline.sh"
cd "${ROOT}"
exec "${UFO_TORCHRUN_BIN}" --standalone --nproc_per_node=8 main.py \
  --config configs/h200/r9_waymo_full_100k.json \
  --train_scene_indices 621 --batch_size 1 --gradient_accumulation_steps 1 \
  --ddp_accumulation_no_sync --ddp_smoke_assertions --num_iterations 2 \
  --project r9_h200_smoke --exp_name scene621_clean_r9_ddp \
  --output_dir outputs --num_workers 2 --ckpt_every_n_iters 1000000 \
  --validation_steps "" --skip_initial_validation --skip_final_evaluation
