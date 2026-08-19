#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys


def main():
    output_root = pathlib.Path(sys.argv[1])
    candidates = [("b8_a1", 8, 1), ("b4_a2", 4, 2), ("b2_a4", 2, 4)]
    rows = []
    for name, batch, accumulation in candidates:
        run_dir = output_root / "h200_smoke" / name
        exit_file = run_dir / "exit_code.txt"
        metrics_file = run_dir / "training_metrics.json"
        if not exit_file.exists() or exit_file.read_text().strip() != "0" or not metrics_file.exists():
            rows.append({"name": name, "status": "FAILED", "batch": batch, "accumulation": accumulation})
            continue
        metrics = [json.loads(line) for line in metrics_file.read_text().splitlines() if line.strip()]
        steady = metrics[max(1, len(metrics) // 3):]
        microseconds = statistics.mean(row["iter_time"] for row in steady)
        optimizer_seconds = microseconds * accumulation
        rows.append({
            "name": name, "status": "PASS", "batch": batch, "accumulation": accumulation,
            "seconds_per_microbatch": microseconds,
            "seconds_per_optimizer_step": optimizer_seconds,
            "samples_per_second": 64.0 / optimizer_seconds,
            "peak_vram_mib": max(row.get("peak_gpu_mb", 0.0) for row in metrics),
        })
    passed = [row for row in rows if row["status"] == "PASS"]
    if not passed:
        raise SystemExit("No H200 global-batch-64 candidate passed")
    winner = min(passed, key=lambda row: row["seconds_per_optimizer_step"])
    result = {"candidates": rows, "winner": winner}
    result_path = output_root / "h200_smoke" / "summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_root / "h200_smoke" / "recommended.env").write_text(
        f"H200_BATCH_SIZE={winner['batch']}\n"
        f"H200_ACCUMULATION_STEPS={winner['accumulation']}\n"
        "H200_EFFECTIVE_GLOBAL_BATCH=64\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
