#!/usr/bin/env python3
"""Run the controlled A-E H200 throughput benchmark."""

import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import shutil


ROOT = Path(os.environ["UFO_ROOT"])
OUTPUT = Path(os.environ["UFO_OUTPUT_ROOT"]) / "h200_speed_benchmark"
BASE_CONFIG = ROOT / "configs/h200/ufo_2s_d50_global64.json"
WARMUP_STEPS = 20
MEASURE_STEPS = 100
ACCUMULATION = 8


STAGES = {
    "A_baseline": {
        "disable_grad_checkpointing": False,
        "sparse_training_diagnostics": False,
        "pin_memory": False,
        "non_blocking_h2d": False,
        "num_workers": 4,
        "prefetch_factor": 2,
        "disable_train_flow_loading": False,
    },
    "B_no_checkpoint": {
        "disable_grad_checkpointing": True,
        "sparse_training_diagnostics": False,
        "pin_memory": False,
        "non_blocking_h2d": False,
        "num_workers": 4,
        "prefetch_factor": 2,
        "disable_train_flow_loading": False,
    },
    "C_sparse_diagnostics": {
        "disable_grad_checkpointing": True,
        "sparse_training_diagnostics": True,
        "pin_memory": False,
        "non_blocking_h2d": False,
        "num_workers": 4,
        "prefetch_factor": 2,
        "disable_train_flow_loading": False,
    },
    "D_data_workers4": {
        "disable_grad_checkpointing": True,
        "sparse_training_diagnostics": True,
        "pin_memory": True,
        "non_blocking_h2d": True,
        "num_workers": 4,
        "prefetch_factor": 4,
        "disable_train_flow_loading": False,
    },
    "D_data_workers8": {
        "disable_grad_checkpointing": True,
        "sparse_training_diagnostics": True,
        "pin_memory": True,
        "non_blocking_h2d": True,
        "num_workers": 8,
        "prefetch_factor": 4,
        "disable_train_flow_loading": False,
    },
}


def write_config(name, overrides):
    config = json.loads(BASE_CONFIG.read_text())
    config.update(overrides)
    config.update({
        "num_iterations": WARMUP_STEPS + MEASURE_STEPS,
        "gradient_accumulation_steps": ACCUMULATION,
        "log_every_n_iters": 1,
        "skip_initial_validation": True,
        "skip_final_evaluation": True,
        "validation_steps": "",
        "vis_every_n_iters": 0,
        "ckpt_every_n_iters": 1_000_000,
        "auto_resume": False,
    })
    path = OUTPUT / "configs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def run_stage(name, overrides):
    shutil.rmtree(OUTPUT / name, ignore_errors=True)
    config = write_config(name, overrides)
    timing = OUTPUT / name / "cuda_timing.json"
    command = [
        os.environ["UFO_TORCHRUN_BIN"], "--standalone", "--nproc_per_node=8",
        "main.py", "--config", str(config), "--project", "h200_speed_benchmark",
        "--exp_name", name, "--output_dir", str(OUTPUT.parent),
        "--data_root", os.environ["UFO_DATA_ROOT"], "--batch_size", "1",
        "--gradient_accumulation_steps", str(ACCUMULATION),
        "--ddp_accumulation_no_sync", "--dynamic_rich_pool", os.environ["UFO_DYNAMIC_POOL"],
        "--benchmark_timing_output", str(timing),
        "--benchmark_warmup_steps", str(WARMUP_STEPS),
    ]
    checkpoint = os.environ.get("UFO_BENCHMARK_CHECKPOINT")
    if not checkpoint:
        checkpoint_dir = (
            Path(os.environ["UFO_OUTPUT_ROOT"])
            / "h200_reproduction/ufo_h200_2s_d50_global64/checkpoints"
        )
        candidates = sorted(checkpoint_dir.glob("ckpt_*.pth"))
        checkpoint = str(candidates[-1]) if candidates else ""
    if checkpoint:
        command.extend(["--load_from", checkpoint])
    log = OUTPUT / name / "stdout.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    metrics_path = OUTPUT.parent / "h200_speed_benchmark" / name / "training_metrics.json"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    steady = [row for row in rows if row["iteration"] >= WARMUP_STEPS * ACCUMULATION]
    cuda = json.loads(timing.read_text())
    result = {
        "sec_per_microbatch": statistics.mean(row["iter_time"] for row in steady),
        "sec_per_optimizer_step": statistics.mean(row["iter_time"] for row in steady) * ACCUMULATION,
        "data_sec_per_microbatch": statistics.mean(row["data_time"] for row in steady),
        "forward_sec_per_microbatch": cuda["forward_seconds_mean"],
        "backward_sec_per_microbatch": cuda["backward_seconds_mean"],
        "optimizer_sec_per_step": cuda["optimizer_seconds_mean"],
        "compute_sec_per_step": cuda["compute_step_seconds_mean"],
        "peak_vram_mb": cuda["peak_vram_mb"],
        "command": command,
        "config": str(config),
        "log": str(log),
        "checkpoint": checkpoint or None,
    }
    (OUTPUT / name / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {name: run_stage(name, overrides) for name, overrides in STAGES.items()}
    data_name = min(
        ("D_data_workers4", "D_data_workers8"),
        key=lambda name: results[name]["sec_per_optimizer_step"],
    )
    e_overrides = copy.deepcopy(STAGES[data_name])
    e_overrides["disable_train_flow_loading"] = True
    results["E_no_train_flow"] = run_stage("E_no_train_flow", e_overrides)
    summary = {
        "warmup_optimizer_steps": WARMUP_STEPS,
        "measured_optimizer_steps": MEASURE_STEPS,
        "selected_data_stage": data_name,
        "results": results,
    }
    summary["best_stage"] = min(results, key=lambda name: results[name]["sec_per_optimizer_step"])
    summary["speedup"] = (
        results["A_baseline"]["sec_per_optimizer_step"]
        / results[summary["best_stage"]]["sec_per_optimizer_step"]
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
