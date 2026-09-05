#!/usr/bin/env python3
"""Benchmark global-batch-64 R9 candidates and write the fastest viable choice."""

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/r9_h200_profile"
CANDIDATES = ((8, 1), (4, 2), (2, 4), (1, 8))


def main():
    torchrun = Path("/root/miniconda3/envs/dggt_data/bin/torchrun")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for batch, accumulation in CANDIDATES:
        name = f"b{batch}_a{accumulation}"
        run = OUTPUT / name
        run.mkdir(exist_ok=True)
        timing = run / "timing.json"
        command = [
            str(torchrun), "--standalone", "--nproc_per_node=8", "main.py",
            "--config", "configs/h200/r9_waymo_full_100k.json",
            "--train_scene_indices", "621", "--batch_size", str(batch),
            "--gradient_accumulation_steps", str(accumulation),
            "--ddp_accumulation_no_sync", "--num_iterations", "6",
            "--benchmark_warmup_steps", "2", "--benchmark_timing_output", str(timing),
            "--project", "r9_h200_profile", "--exp_name", name,
            "--output_dir", "outputs", "--num_workers", "2",
            "--ckpt_every_n_iters", "1000000", "--validation_steps", "",
            "--skip_initial_validation", "--skip_final_evaluation",
        ]
        with (run / "stdout.log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, env=os.environ, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        result = {"batch_size": batch, "accumulation_steps": accumulation,
                  "global_batch": 8 * batch * accumulation,
                  "returncode": completed.returncode, "command": command}
        if completed.returncode == 0 and timing.is_file():
            measured = json.loads(timing.read_text())
            result.update(measured)
            result["optimizer_step_seconds"] = (
                measured["forward_seconds_mean"] * accumulation
                + measured["backward_seconds_mean"] * accumulation
                + measured["optimizer_seconds_mean"]
            )
            result["status"] = "PASS"
        else:
            log_text = (run / "stdout.log").read_text(errors="ignore")
            result["status"] = "OOM" if "out of memory" in log_text.lower() else "FAIL"
        results.append(result)
        (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: {result['status']}")
    viable = [result for result in results if result["status"] == "PASS"]
    if not viable:
        raise RuntimeError("All R9 H200 benchmark candidates failed")
    best = min(viable, key=lambda result: result["optimizer_step_seconds"])
    (OUTPUT / "summary.json").write_text(json.dumps({"results": results, "recommended": best}, indent=2) + "\n")
    (OUTPUT / "recommended.env").write_text(
        f"R9_BATCH_SIZE={best['batch_size']}\n"
        f"R9_ACCUMULATION_STEPS={best['accumulation_steps']}\n"
    )
    print(f"RECOMMENDED b{best['batch_size']} a{best['accumulation_steps']}")


if __name__ == "__main__":
    main()
