#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"
mode="${1:-d50}"
variant="${UFO_8S_EVAL_VARIANT:-best}"
case "${mode}" in d50|uniform) ;; *) echo "mode must be d50 or uniform" >&2; exit 2 ;; esac
case "${variant}" in best|last) ;; *) echo "UFO_8S_EVAL_VARIANT must be best or last" >&2; exit 2 ;; esac
checkpoint="${2:-${UFO_OUTPUT_ROOT}/h200_artifacts/ufo_8s_from_${mode}/${variant}.pth}"
test -e "${checkpoint}" || { echo "Missing 8s checkpoint: ${checkpoint}" >&2; exit 1; }
"${UFO_TORCHRUN_BIN}" --standalone --nproc_per_node=8 main.py --config configs/h200/ufo_16s_zeroshot_eval.json \
  --exp_name "ufo_h200_16s_from_${mode}_${variant}_zeroshot_eval" \
  --output_dir "${UFO_OUTPUT_ROOT}" --data_root "${UFO_DATA_ROOT}" \
  --load_from "${checkpoint}" --evaluate --skip_initial_validation
