#!/usr/bin/env python3
"""Summarize the scene621 E0/E1/E2 UFO pose experiment."""

import argparse
import json
import math
from pathlib import Path


RUNS = {
    "E0": "e0_gt",
    "E1": "e1_omega_context",
    "E2": "e2_omega_all",
}


def normalize_nonfinite(value):
    if isinstance(value, dict):
        return {key: normalize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_nonfinite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = {}
    for experiment, directory in RUNS.items():
        metrics_path = (
            args.experiment_root
            / directory
            / "scene_00621_start_000_metrics.json"
        )
        if not metrics_path.is_file():
            raise FileNotFoundError(f"metrics not found: {metrics_path}")
        summary[experiment] = json.loads(metrics_path.read_text())

    baseline_psnr = summary["E0"]["psnr"]
    for metrics in summary.values():
        metrics["delta_psnr_from_e0"] = metrics["psnr"] - baseline_psnr
    result = {
        "scene_index": 621,
        "protocol": "16 target frames x 3 cameras over four autoregressive chunks",
        "experiments": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = normalize_nonfinite(result)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
