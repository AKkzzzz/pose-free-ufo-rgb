#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"
SMOKE_ROOT="${UFO_OUTPUT_ROOT}/h200_smoke"
mkdir -p "${SMOKE_ROOT}"

run_candidate() {
  local name="$1" batch="$2" accumulation="$3"
  local run_dir="${SMOKE_ROOT}/${name}"
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}"
  set +e
  "${UFO_TORCHRUN_BIN}" --standalone --nproc_per_node=8 main.py \
    --config configs/h200/ufo_2s_uniform_global64.json \
    --exp_name "${name}" --project h200_smoke --output_dir "${UFO_OUTPUT_ROOT}" \
    --data_root "${UFO_DATA_ROOT}" --batch_size "${batch}" \
    --gradient_accumulation_steps "${accumulation}" --ddp_accumulation_no_sync \
    --ddp_smoke_assertions \
    --num_iterations 20 --ckpt_every_n_iters 20 --validation_steps "" \
    --skip_initial_validation --skip_final_evaluation > "${run_dir}/stdout.log" 2>&1
  status=$?
  set -e
  echo "${status}" > "${run_dir}/exit_code.txt"
  if [[ "${status}" -eq 0 ]]; then
    test -f "${UFO_OUTPUT_ROOT}/h200_smoke/${name}/ddp_smoke_status.json"
    test "$(find "${UFO_OUTPUT_ROOT}/h200_smoke/${name}/checkpoints" -maxdepth 1 -type f -name '*.pth' | wc -l)" -eq 1
  fi
}

run_candidate b1_a8 1 8
"${UFO_PYTHON_BIN}" scripts/h200/select_smoke_result.py "${UFO_OUTPUT_ROOT}"
