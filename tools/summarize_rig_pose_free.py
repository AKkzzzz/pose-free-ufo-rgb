#!/usr/bin/env python3
"""Summarize scene621 rig pose-free camera experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    variants = [
        ("P0", "GT camera", "P0_GT_camera/metrics.json"),
        ("P1", "Omega + GT Sim3", "P1_Omega_GT_Sim3/metrics.json"),
        ("P2", "Omega + rig-only metric local", "P2_Omega_rig_only/metrics.json"),
    ]
    p21_metrics = args.root / "P21_Omega_rig_only_recurrent/metrics.json"
    if p21_metrics.is_file():
        variants.append((
            "P2.1", "Omega + rig-only recurrent local/global",
            "P21_Omega_rig_only_recurrent/metrics.json",
        ))
    rows = []
    windows = {}
    for code, name, relative in variants:
        payload = json.loads((args.root / relative).read_text())
        metrics = payload["metrics"]
        rows.append({
            "variant": code,
            "name": name,
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "static_psnr": metrics["static_psnr"],
            "static_ssim": metrics["static_ssim"],
            "depth_rmse": metrics["depth_rmse"],
            "dynamic_psnr_reference_only": metrics["dynamic_psnr"],
        })
        windows[code] = payload["windows"]
    contract_path = args.root / (
        "contract_check_p21.json" if p21_metrics.is_file() else "contract_check.json"
    )
    contract = json.loads(contract_path.read_text())
    summary = {
        "experiment": "scene621 all-frame rig-only camera pose diagnostic",
        "fair_nvs": False,
        "comparison": rows,
        "delta_p2_vs_p1_psnr": rows[2]["psnr"] - rows[1]["psnr"],
        "delta_p2_vs_p1_static_psnr": rows[2]["static_psnr"] - rows[1]["static_psnr"],
        "scale": {
            "mean": contract["scale_mean"],
            "std": contract["scale_std"],
            "coefficient_of_variation": contract["scale_coefficient_of_variation"],
        },
        "contract_check": contract,
        "per_window_metrics": windows,
    }
    if p21_metrics.is_file():
        summary.update({
            "delta_p21_vs_p2_psnr": rows[3]["psnr"] - rows[2]["psnr"],
            "delta_p21_vs_p1_psnr": rows[3]["psnr"] - rows[1]["psnr"],
            "delta_p21_vs_p2_static_psnr": (
                rows[3]["static_psnr"] - rows[2]["static_psnr"]
            ),
        })
    output_stem = "p0_p1_p2_p21_summary" if p21_metrics.is_file() else "p0_p1_p2_summary"
    (args.root / f"{output_stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.root / f"{output_stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    printed = {
        "comparison": rows,
        "delta_p2_vs_p1_psnr": summary["delta_p2_vs_p1_psnr"],
        "scale": summary["scale"],
    }
    for key in ("delta_p21_vs_p2_psnr", "delta_p21_vs_p1_psnr"):
        if key in summary:
            printed[key] = summary[key]
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
