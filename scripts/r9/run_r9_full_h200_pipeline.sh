#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ "${R9_PIPELINE_DRY_RUN:-0}" == "1" ]]; then
  echo "R9 PIPELINE DRY RUN root=${ROOT}"
  echo "STEP0 bash tools/r9/assert_clean_r9_core.sh"
  echo "STEP1 /root/miniconda3/envs/dggt_data/bin/python tools/r9/audit_portability.py"
  echo "STEP2 bash scripts/h200/bootstrap_h200_offline.sh"
  echo "STEP2 /root/miniconda3/envs/dggt_data/bin/torchrun --standalone --nproc_per_node=8 tools/r9/nccl_smoke.py"
  echo "STEP2 HF_HOME=${ROOT}/third_party/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ${ROOT}/third_party/groundedsam2_env/bin/python tools/r9/smoke_groundedsam_offline.py"
  echo "STEP3 /root/miniconda3/envs/dggt_data/bin/python tools/r9/import_existing_numeric_sam_tracks.py"
  echo "STEP4 bash scripts/r9/run_sam_h200_8gpu.sh"
  echo "STEP5 /root/miniconda3/envs/dggt_data/bin/python tools/r9/audit_waymo_sam_tracks.py --require-full"
  echo "STEP6 /root/miniconda3/envs/dggt_data/bin/python tools/r9/check_sam_track_contract.py --all-scenes"
  echo "STEP7 bash scripts/r9/smoke_r9_h200_ddp.sh"
  echo "STEP8 bash scripts/r9/benchmark_r9_h200.sh"
  echo "STEP9 source outputs/r9_h200_profile/recommended.env"
  echo "STEP10 print GPU,batch/GPU,accum,global_batch=64,iterations=100000,SAM=data/r9_sam_tracks,output=outputs/full_100k/r9_waymo_full_100k"
  echo "STEP11 bash scripts/r9/train_r9_waymo_full_h200.sh"
  echo "R9_PIPELINE_DRY_RUN=PASS"
  exit 0
fi

echo "STEP0 clean R9 core"
bash tools/r9/assert_clean_r9_core.sh
echo "STEP1 portability/path audit"
/root/miniconda3/envs/dggt_data/bin/python tools/r9/audit_portability.py
echo "STEP2 H200 bootstrap"
bash scripts/h200/bootstrap_h200_offline.sh
/root/miniconda3/envs/dggt_data/bin/torchrun --standalone --nproc_per_node=8 tools/r9/nccl_smoke.py
HF_HOME="${ROOT}/third_party/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${ROOT}/third_party/groundedsam2_env/bin/python" tools/r9/smoke_groundedsam_offline.py
echo "STEP3 import valid numeric SAM"
/root/miniconda3/envs/dggt_data/bin/python tools/r9/import_existing_numeric_sam_tracks.py
echo "STEP4 eight persistent SAM processes"
bash scripts/r9/run_sam_h200_8gpu.sh
echo "STEP5 full 2394-pair audit"
/root/miniconda3/envs/dggt_data/bin/python tools/r9/audit_waymo_sam_tracks.py --require-full
echo "STEP6 frozen Dataset/SAM contract"
/root/miniconda3/envs/dggt_data/bin/python tools/r9/check_sam_track_contract.py --all-scenes
echo "STEP7 R9-specific DDP smoke"
bash scripts/r9/smoke_r9_h200_ddp.sh
echo "STEP8 R9 H200 benchmark"
bash scripts/r9/benchmark_r9_h200.sh
echo "STEP9 benchmark recommendation"
source outputs/r9_h200_profile/recommended.env
: "${R9_BATCH_SIZE:?missing R9_BATCH_SIZE}"
: "${R9_ACCUMULATION_STEPS:?missing R9_ACCUMULATION_STEPS}"
echo "STEP10 final configuration"
nvidia-smi -L
echo "batch_per_gpu=${R9_BATCH_SIZE}"
echo "accumulation_steps=${R9_ACCUMULATION_STEPS}"
echo "global_batch=$((8 * R9_BATCH_SIZE * R9_ACCUMULATION_STEPS))"
echo "iterations=100000"
echo "sam_root=${ROOT}/data/r9_sam_tracks"
echo "output=${ROOT}/outputs/full_100k/r9_waymo_full_100k"
echo "STEP11 foreground Full Waymo R9 100k"
exec bash scripts/r9/train_r9_waymo_full_h200.sh
