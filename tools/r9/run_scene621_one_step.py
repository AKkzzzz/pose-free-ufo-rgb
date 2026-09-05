#!/usr/bin/env python3
"""Run clean main.py while observing R5/R9 diagnostics without core edits."""

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import main as training
from ufo.models import sam_object_detail_r9
from ufo.utils.config import merge_config_and_args


def scalar(value):
    return value.detach().float().cpu().item() if torch.is_tensor(value) else value


def main():
    captured = {}
    original = sam_object_detail_r9.predict_sam_detail_motion

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.update({key: scalar(value) for key, value in result[3].items()})
        return result

    sam_object_detail_r9.predict_sam_detail_motion = observed
    config = "configs/h200/r9_scene621_one_step.json"
    parser = training.get_args_parser()
    args = merge_config_and_args(parser, config_path=config, cli_args=["--config", config])
    training.main(args)
    metrics_path = ROOT / "outputs/r9_one_step_smoke/scene621_clean_r9/training_metrics.json"
    metrics = json.loads(metrics_path.read_text().splitlines()[-1])
    required = (
        "r5_local_track_count", "r5_global_object_count", "r9_object_count",
        "r9_fused_voxel_count", "r9_fusion_ratio",
    )
    missing = [name for name in required if name not in captured]
    if missing:
        raise RuntimeError(f"R9 diagnostics were not observed: {missing}")
    report = {
        "status": "PASS",
        "loss": metrics["loss"],
        "peak_vram_mb": metrics["peak_gpu_mb"],
        **{name: captured[name] for name in required},
    }
    path = ROOT / "outputs/r9_one_step_smoke/scene621_clean_r9/one_step_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("SCENE621_CLEAN_R9_ONE_STEP=PASS")


if __name__ == "__main__":
    main()
