#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"

world_size="${UFO_SMOKE_WORLD_SIZE:-1}"
output_root="${UFO_SMOKE_OUTPUT_ROOT:-${UFO_OUTPUT_ROOT}/offline_ufo_train_smoke}"
rm -rf "${output_root}"
mkdir -p "${output_root}"

command=(
  "${UFO_TORCHRUN_BIN}" --standalone --nproc_per_node="${world_size}" main.py
  --config configs/h200/ufo_2s_uniform_global64.json
  --project offline_smoke --exp_name full_ufo_one_step
  --output_dir "${output_root}" --data_root "${UFO_DATA_ROOT}"
  --batch_size 1 --gradient_accumulation_steps 1
  --num_iterations 1 --ckpt_every_n_iters 1 --keep_n_ckpts 1
  --validation_steps "" --skip_initial_validation --skip_final_evaluation
)

if [[ "${UFO_SMOKE_BLOCK_NETWORK:-0}" == "1" ]]; then
  if ! command -v unshare >/dev/null; then
    echo "unshare is required for a kernel-level offline smoke" >&2
    exit 1
  fi
  unshare --net --map-root-user "${command[@]}"
else
  "${command[@]}"
fi

test -e "${output_root}/offline_smoke/full_ufo_one_step/checkpoints/latest.pth"
echo "OFFLINE_FULL_UFO_TRAIN_SMOKE=PASS"
