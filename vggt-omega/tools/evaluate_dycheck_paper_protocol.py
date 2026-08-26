#!/usr/bin/env python3
"""Evaluate the seven DyCheck iPhone scenes on fixed 10-frame samples."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_gradio import load_model
from evaluate_sintel_paper_protocol import auc, pose_errors
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def transform_depth(depth, source_wh, target_hw):
    width, height = source_wh
    aspect = height / width
    if aspect < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        left = max((width - crop_width) // 2, 0)
        depth = depth[:, left : left + crop_width]
    elif aspect > 2:
        crop_height = min(height, max(1, int(round(width * 2))))
        top = max((height - crop_height) // 2, 0)
        depth = depth[top : top + crop_height]
    return cv2.resize(depth, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = load_model(str(args.checkpoint))
    results = []
    for scene in sorted(p for p in args.root.iterdir() if p.is_dir()):
        rgb = sorted((scene / "rgb/2x").glob("*.png"))
        depth = {p.stem: p for p in (scene / "depth/2x").glob("*.npy")}
        camera = {p.stem: p for p in (scene / "camera").glob("*.json")}
        rgb = [p for p in rgb if p.stem in depth and p.stem in camera]
        if len(rgb) != 10:
            raise RuntimeError(f"{scene.name}: expected 10 complete frames, found {len(rgb)}")
        images = load_and_preprocess_images([str(p) for p in rgb], image_resolution=512).cuda()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model(images)
        pred_ext, _ = encoding_to_camera(output["pose_enc"], output["images"].shape[-2:])
        pred_ext = pred_ext[0].cpu().numpy()
        pred_depth = output["depth"][0, ..., 0].cpu().numpy()
        gt_depth, gt_ext = [], []
        for index, image_path in enumerate(rgb):
            image = cv2.imread(str(image_path))
            current_depth = np.load(depth[image_path.stem]).squeeze().astype(np.float32)
            gt_depth.append(transform_depth(current_depth, (image.shape[1], image.shape[0]), pred_depth[index].shape))
            info = json.loads(camera[image_path.stem].read_text())
            orientation = np.asarray(info["orientation"], dtype=np.float64)
            position = np.asarray(info["position"], dtype=np.float64)
            matrix = np.eye(4)
            matrix[:3, :3] = orientation
            matrix[:3, 3] = -orientation @ position
            gt_ext.append(matrix)
        gt_depth = np.asarray(gt_depth)
        gt_ext = np.asarray(gt_ext)
        valid = (gt_depth > 0) & np.isfinite(gt_depth) & (pred_depth > 0) & np.isfinite(pred_depth)
        scale = float(np.median(gt_depth[valid] / pred_depth[valid]))
        predicted = pred_depth[valid] * scale
        target = gt_depth[valid]
        pred_44 = np.tile(np.eye(4), (10, 1, 1))
        pred_44[:, :3, :4] = pred_ext
        errors = pose_errors(pred_44, gt_ext)
        item = {
            "scene": scene.name,
            "frame_stems": [p.stem for p in rgb],
            "auc_3_deg": auc(errors, 3),
            "auc_30_deg": auc(errors, 30),
            "delta_1.25_percent": float(100 * np.mean(np.maximum(predicted / target, target / predicted) < 1.25)),
            "abs_rel": float(np.mean(np.abs(predicted - target) / target)),
            "valid_depth_pixels": int(valid.sum()),
            "inference_seconds": time.perf_counter() - start,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        results.append(item)
        print(json.dumps(item), flush=True)
    keys = ["auc_3_deg", "auc_30_deg", "delta_1.25_percent", "abs_rel"]
    summary = {key: float(np.mean([item[key] for item in results])) for key in keys}
    report = {
        "status": "approximate reproduction; exact paper sample seed/evaluation code are not public",
        "dataset": "DyCheck iPhone (7 standard scenes)",
        "frames_per_scene": 10,
        "summary": summary,
        "paper_1b": {"auc_3_deg": 38.4, "auc_30_deg": 87.3, "delta_1.25_percent": 98.4, "abs_rel": 0.038},
        "scenes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
