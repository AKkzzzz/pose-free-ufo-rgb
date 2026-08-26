#!/usr/bin/env bash
set -euo pipefail
task=2s
mode=d50
resume=latest
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) task="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --resume) resume="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"
if [[ "${UFO_SKIP_BOOTSTRAP:-0}" != "1" ]]; then bash scripts/h200/bootstrap_h200_offline.sh; fi
if [[ ! -f "${UFO_OUTPUT_ROOT}/h200_smoke/recommended.env" ]]; then bash scripts/h200/smoke_h200_ddp.sh; fi
case "${task}" in
  2s) bash scripts/h200/train_ufo_2s.sh "${mode}" "${resume}" ;;
  8s) bash scripts/h200/train_ufo_8s.sh "${mode}" "${resume}" ;;
  16s) bash scripts/h200/eval_ufo_16s.sh "${mode}" ;;
  dynamic) bash scripts/h200/train_ufo_dynamic.sh "${mode}" "${resume}" ;;
  all)
    bash scripts/h200/train_ufo_2s.sh "${mode}" "${resume}"
    bash scripts/h200/train_ufo_8s.sh "${mode}" latest
    bash scripts/h200/eval_ufo_16s.sh "${mode}"
    bash scripts/h200/train_ufo_dynamic.sh paper latest
    ;;
  *) echo "task must be 2s, 8s, 16s, dynamic, or all" >&2; exit 2 ;;
esac
