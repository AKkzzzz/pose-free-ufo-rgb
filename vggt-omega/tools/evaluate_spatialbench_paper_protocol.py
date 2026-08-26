#!/usr/bin/env python3
"""Evaluate VGGT-Omega on 7Scenes, NRGBD, and ETH3D with 10 images/scene.

The VGGT-Omega authors did not release their evaluation code, scene samples,
or random seed. This script uses the public SpatialBenchmark medium subset as
a deterministic data adapter, then applies the metrics stated in the paper.
"""
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
sys.path.insert(0, str(ROOT / "third_party/spatialbench"))

from benchmark.datasets.data_readers import _decode_depth
from demo_gradio import load_model
from evaluate_sintel_paper_protocol import auc, pose_errors
from vggt_omega.utils.load_fn import _balanced_target_shape, load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


PAPER = {
    "7scenes": {"auc_3_deg": 29.6, "auc_30_deg": 83.1, "delta_1.25_percent": 94.6, "abs_rel": 0.058},
    "nrgbd": {"auc_3_deg": 89.7, "auc_30_deg": 98.8, "delta_1.25_percent": 99.6, "abs_rel": 0.010},
    "eth3d": {"auc_3_deg": 49.8, "auc_30_deg": 88.5, "delta_1.25_percent": 99.8, "abs_rel": 0.012},
}
Z_FAR = {"7scenes": 10.0, "nrgbd": 10.0, "eth3d": 30.0}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DEPTH_EXTS = IMAGE_EXTS | {".exr", ".npy"}


def sorted_files(directory: Path, exts: set[str]) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)


def transform_depth(depth: np.ndarray, source_wh: tuple[int, int], target_hw: tuple[int, int]) -> np.ndarray:
    width, height = source_wh
    aspect = height / width
    if aspect < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        left = max((width - crop_width) // 2, 0)
        depth = depth[:, left : left + crop_width]
    elif aspect > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        top = max((height - crop_height) // 2, 0)
        depth = depth[top : top + crop_height]
    return cv2.resize(depth, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)


def load_scene_gt(scene: Path, indices: np.ndarray, target_hw: tuple[int, int], dataset: str):
    meta = json.loads((scene / "meta.json").read_text())
    images = sorted_files(scene / "images", IMAGE_EXTS)
    depths = sorted_files(scene / "depths", DEPTH_EXTS)
    poses = sorted((scene / "poses").glob("*.npy"))
    depth_masks = sorted_files(scene / "depth_masks", IMAGE_EXTS) if (scene / "depth_masks").is_dir() else []
    gt_depth, gt_w2c = [], []
    for idx in indices:
        image = cv2.imread(str(images[idx]), cv2.IMREAD_COLOR)
        depth, _ = _decode_depth(str(depths[idx]), meta["depth_format"], meta)
        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth)] = 0
        if depth_masks:
            mask = cv2.imread(str(depth_masks[idx]), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape == depth.shape:
                depth[mask == 0] = 0
        depth[depth > Z_FAR[dataset]] = 0
        gt_depth.append(transform_depth(depth, (image.shape[1], image.shape[0]), target_hw))
        c2w = np.load(poses[idx]).astype(np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        gt_w2c.append(np.linalg.inv(c2w))
    return images, np.asarray(gt_depth), np.asarray(gt_w2c)


def evaluate_dataset(model, root: Path, dataset: str, seed: int):
    scenes = sorted(p.parent for p in (root / dataset).rglob("meta.json"))
    results = []
    for scene_index, scene in enumerate(scenes):
        image_paths = sorted_files(scene / "images", IMAGE_EXTS)
        rng = np.random.default_rng(seed + scene_index)
        indices = np.sort(rng.choice(len(image_paths), 10, replace=False))
        selected = [image_paths[i] for i in indices]
        images = load_and_preprocess_images([str(p) for p in selected], image_resolution=512).cuda()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model(images)
        extrinsics, _ = encoding_to_camera(output["pose_enc"], output["images"].shape[-2:])
        pred_w2c = extrinsics[0].cpu().numpy()
        pred_depth = output["depth"][0, ..., 0].cpu().numpy()
        elapsed = time.perf_counter() - start
        _, gt_depth, gt_w2c = load_scene_gt(scene, indices, pred_depth.shape[-2:], dataset)
        valid = np.isfinite(gt_depth) & (gt_depth > 0) & np.isfinite(pred_depth) & (pred_depth > 0)
        scale = float(np.median(gt_depth[valid] / pred_depth[valid]))
        pred_valid = pred_depth[valid] * scale
        gt_valid = gt_depth[valid]
        pred_44 = np.tile(np.eye(4), (10, 1, 1))
        pred_44[:, :3, :4] = pred_w2c
        errors = pose_errors(pred_44, gt_w2c)
        item = {
            "scene": str(scene.relative_to(root / dataset)),
            "frame_indices_in_spatialbench_subset": indices.tolist(),
            "scale": scale,
            "auc_3_deg": auc(errors, 3),
            "auc_30_deg": auc(errors, 30),
            "delta_1.25_percent": float(100 * np.mean(np.maximum(pred_valid / gt_valid, gt_valid / pred_valid) < 1.25)),
            "abs_rel": float(np.mean(np.abs(pred_valid - gt_valid) / gt_valid)),
            "valid_depth_pixels": int(valid.sum()),
            "inference_seconds": elapsed,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        results.append(item)
        print(json.dumps(item), flush=True)
    keys = ["auc_3_deg", "auc_30_deg", "delta_1.25_percent", "abs_rel"]
    return results, {key: float(np.mean([item[key] for item in results])) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(PAPER), default=sorted(PAPER))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model = load_model(str(args.checkpoint))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        scenes, summary = evaluate_dataset(model, args.root, dataset, args.seed)
        output = {
            "status": "approximate reproduction; exact paper samples, seed, and evaluation code are not public",
            "data_adapter": "HarrisonPENG/SpatialBenchmark medium subset",
            "dataset": dataset,
            "seed": args.seed,
            "frames_per_scene": 10,
            "summary": summary,
            "paper_1b": PAPER[dataset],
            "scenes": scenes,
        }
        path = args.output_dir / f"{dataset}_seed{args.seed}.json"
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps({"dataset": dataset, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
