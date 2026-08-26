#!/usr/bin/env python3
"""Print the six final benchmark JSON files as a compact Markdown table."""
import argparse
import json
from pathlib import Path


FILES = [
    "7scenes_seed0.json",
    "nrgbd_seed0.json",
    "eth3d_seed0.json",
    "dycheck_seed0.json",
    "sintel_seed0_final.json",
    "tum_dynamic_seed0_final.json",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    print("| Dataset | AUC@3 | AUC@30 | delta1.25 | AbsRel |")
    print("|---|---:|---:|---:|---:|")
    for filename in FILES:
        result = json.loads((args.result_dir / filename).read_text())
        metric = result["summary"]
        print(
            f"| {result['dataset']} | {metric['auc_3_deg']:.1f} | "
            f"{metric['auc_30_deg']:.1f} | {metric['delta_1.25_percent']:.1f} | "
            f"{metric['abs_rel']:.3f} |"
        )


if __name__ == "__main__":
    main()
