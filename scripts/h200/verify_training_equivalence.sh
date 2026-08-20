#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"

root="${UFO_OUTPUT_ROOT}/h200_training_equivalence"
rm -rf "${root}"
mkdir -p "${root}/configs"

"${UFO_PYTHON_BIN}" - "${root}" <<'PY'
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
base = json.loads(pathlib.Path("configs/h200/ufo_2s_d50_global64.json").read_text())
checkpoint = os.environ.get("UFO_BENCHMARK_CHECKPOINT")
if not checkpoint:
    candidates = sorted((pathlib.Path(os.environ["UFO_OUTPUT_ROOT"]) /
                         "h200_reproduction/ufo_h200_2s_d50_global64/checkpoints").glob("ckpt_*.pth"))
    checkpoint = str(candidates[-1]) if candidates else None
common = {
    "num_iterations": 1, "gradient_accumulation_steps": 8,
    "skip_initial_validation": True, "skip_final_evaluation": True,
    "validation_steps": "", "vis_every_n_iters": 0, "ckpt_every_n_iters": 1,
    "auto_resume": False, "load_from": checkpoint,
}
baseline = dict(base, **common, disable_grad_checkpointing=False,
                sparse_training_diagnostics=False, pin_memory=False,
                non_blocking_h2d=False, num_workers=4, prefetch_factor=2,
                disable_train_flow_loading=False)
candidate = dict(base, **common, disable_grad_checkpointing=True,
                 sparse_training_diagnostics=True, pin_memory=True,
                 non_blocking_h2d=True, num_workers=8, prefetch_factor=4,
                 disable_train_flow_loading=True)
for name, config in (("baseline", baseline), ("candidate", candidate)):
    (root / "configs" / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
PY

run_one() {
  local name="$1"
  "${UFO_TORCHRUN_BIN}" --standalone --nproc_per_node=8 main.py \
    --config "${root}/configs/${name}.json" --project h200_training_equivalence \
    --exp_name "${name}" --output_dir "${UFO_OUTPUT_ROOT}" --data_root "${UFO_DATA_ROOT}" \
    --batch_size 1 --gradient_accumulation_steps 8 --ddp_accumulation_no_sync \
    --dynamic_rich_pool "${UFO_DYNAMIC_POOL}" \
    --equivalence_artifact "${root}/${name}.pth"
}

run_one baseline
run_one candidate
"${UFO_PYTHON_BIN}" scripts/h200/compare_training_equivalence.py \
  "${root}/baseline.pth" "${root}/candidate.pth" --output "${root}/result.json"
