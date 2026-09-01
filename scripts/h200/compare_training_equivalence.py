#!/usr/bin/env python3
"""Compare one-step baseline/final training artifacts."""

import argparse
import json
from pathlib import Path

import torch


def max_difference(left, right):
    keys = set(left) | set(right)
    if set(left) != set(right):
        return float("inf")
    return max((left[key] - right[key]).abs().max().item() for key in keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    result = {
        "MAX_LOSS_DIFF": max_difference(baseline["losses"], candidate["losses"]),
        "MAX_GRAD_DIFF": max_difference(baseline["gradients"], candidate["gradients"]),
        "MAX_PARAM_UPDATE_DIFF": max_difference(
            baseline["parameters_after"], candidate["parameters_after"]
        ),
        "STATE_DICT_KEYS_IDENTICAL": baseline["model_keys"] == candidate["model_keys"],
        "OPTIMIZER_PARAM_GROUPS_IDENTICAL": (
            baseline["optimizer_group_names"] == candidate["optimizer_group_names"]
        ),
    }
    result["TRAINING_EQUIVALENCE"] = "PASS" if (
        result["MAX_LOSS_DIFF"] <= 1e-7
        and result["MAX_GRAD_DIFF"] <= 1e-6
        and result["MAX_PARAM_UPDATE_DIFF"] <= 1e-7
        and result["STATE_DICT_KEYS_IDENTICAL"]
        and result["OPTIMIZER_PARAM_GROUPS_IDENTICAL"]
    ) else "FAIL"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
