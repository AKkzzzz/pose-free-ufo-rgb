#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${ROOT}/outputs/full_100k/r9_waymo_full_100k"
"/root/miniconda3/envs/dggt_data/bin/python" \
  "${ROOT}/tools/r9/check_full_start_mode.py" --run-dir "${RUN_DIR}"
PROFILE="${ROOT}/outputs/r9_h200_profile/recommended.env"
test -f "${PROFILE}" || { echo "Missing benchmark recommendation: ${PROFILE}" >&2; exit 1; }
source "${PROFILE}"
: "${R9_BATCH_SIZE:?missing R9_BATCH_SIZE}"
: "${R9_ACCUMULATION_STEPS:?missing R9_ACCUMULATION_STEPS}"
test "$((8 * R9_BATCH_SIZE * R9_ACCUMULATION_STEPS))" -eq 64
source "${ROOT}/scripts/h200/env_h200_offline.sh"
cd "${ROOT}"
exec /root/miniconda3/envs/dggt_data/bin/torchrun --standalone --nproc_per_node=8 main.py \
  --config configs/h200/r9_waymo_full_100k.json \
  --batch_size "${R9_BATCH_SIZE}" \
  --gradient_accumulation_steps "${R9_ACCUMULATION_STEPS}" \
  --ddp_accumulation_no_sync --auto_resume --num_iterations 100000 \
  --project full_100k --exp_name r9_waymo_full_100k --output_dir outputs \
  --skip_initial_validation --skip_final_evaluation
