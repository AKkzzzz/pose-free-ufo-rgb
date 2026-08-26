#!/usr/bin/env python3
"""Paper-style 10-frame VGGT-Omega diagnostics on processed Waymo scenes.

Waymo is not one of the six benchmarks reported by VGGT-Omega. This script
reuses the paper's disclosed 10-frame input setting and pose/depth metrics, but
the resulting numbers are a custom Waymo diagnostic rather than an official
paper benchmark.
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

from demo_gradio import load_model
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


OPENCV_TO_WAYMO = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0, help="Waymo camera 0 is FRONT")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", choices=("random", "contiguous"), default="random")
    parser.add_argument("--start-frame", type=int, default=50)
    parser.add_argument("--image-resolution", type=int, default=512)
    return parser.parse_args()


def auc(errors: np.ndarray, threshold: int) -> float:
    histogram, _ = np.histogram(np.asarray(errors), bins=np.arange(threshold + 1))
    return float(100 * np.mean(np.cumsum(histogram.astype(float) / len(errors))))


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def pose_errors(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> np.ndarray:
    errors = []
    for i in range(len(pred_w2c)):
        for j in range(i + 1, len(pred_w2c)):
            pred_i_r, pred_i_t = pred_w2c[i, :3, :3], pred_w2c[i, :3, 3]
            gt_i_r, gt_i_t = gt_w2c[i, :3, :3], gt_w2c[i, :3, 3]
            pred_rel_r = pred_w2c[j, :3, :3] @ pred_i_r.T
            pred_rel_t = pred_w2c[j, :3, 3] - pred_rel_r @ pred_i_t
            gt_rel_r = gt_w2c[j, :3, :3] @ gt_i_r.T
            gt_rel_t = gt_w2c[j, :3, 3] - gt_rel_r @ gt_i_t
            rotation_error = rotation_angle(pred_rel_r @ gt_rel_r.T)
            denominator = np.linalg.norm(pred_rel_t) * np.linalg.norm(gt_rel_t)
            if denominator < 1e-9:
                translation_error = 90.0
            else:
                cosine = np.clip(abs(pred_rel_t @ gt_rel_t) / denominator, -1, 1)
                translation_error = float(np.degrees(np.arccos(cosine)))
            errors.append(max(rotation_error, translation_error))
    return np.asarray(errors, dtype=np.float64)


def available_frames(scene_dir: Path, camera: int) -> list[int]:
    image_ids = {int(path.stem.split("_")[0]) for path in (scene_dir / "images_4").glob(f"*_{camera}.jpg")}
    depth_ids = {int(path.stem.split("_")[0]) for path in (scene_dir / "depth_flows_4").glob(f"*_{camera}.npy")}
    pose_ids = {int(path.stem) for path in (scene_dir / "ego_pose").glob("*.txt")}
    return sorted(image_ids & depth_ids & pose_ids)


def transform_gt_depth(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Mirror the official extreme-aspect crop, then resize sparse GT."""
    height, width = depth.shape
    aspect = height / max(width, 1)
    if aspect < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        left = max((width - crop_width) // 2, 0)
        depth = depth[:, left : left + crop_width]
    elif aspect > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        top = max((height - crop_height) // 2, 0)
        depth = depth[top : top + crop_height]
    return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def load_gt_w2c(scene_dir: Path, frame_ids: list[int], camera: int) -> np.ndarray:
    camera_to_ego = np.loadtxt(scene_dir / "extrinsics" / f"{camera}.txt", dtype=np.float64)
    world_to_camera = []
    for frame_id in frame_ids:
        ego_to_world = np.loadtxt(scene_dir / "ego_pose" / f"{frame_id:03d}.txt", dtype=np.float64)
        camera_to_world = ego_to_world @ camera_to_ego @ OPENCV_TO_WAYMO
        world_to_camera.append(np.linalg.inv(camera_to_world))
    return np.stack(world_to_camera)


def depth_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    ratio = np.maximum(prediction / target, target / prediction)
    return {
        "abs_rel": float(np.mean(np.abs(prediction - target) / target)),
        "rmse_m": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "delta_1.25_percent": float(100 * np.mean(ratio < 1.25)),
        "delta_1.25_sq_percent": float(100 * np.mean(ratio < 1.25**2)),
        "delta_1.25_cu_percent": float(100 * np.mean(ratio < 1.25**3)),
        "num_lidar_pixels": int(len(target)),
    }


def numpy_prediction(value: torch.Tensor) -> np.ndarray:
    result = value.detach().float().cpu().numpy()
    return result[0] if result.shape[0] == 1 else result


def evaluate_scene(args: argparse.Namespace, model, scene_name: str, scene_index: int) -> dict:
    scene_dir = args.data_root / "datasets" / "waymo" / args.split / scene_name
    frame_pool = available_frames(scene_dir, args.camera)
    if len(frame_pool) < args.num_frames:
        raise RuntimeError(f"{scene_dir} has only {len(frame_pool)} complete frames")

    scene_seed = args.seed + scene_index
    if args.sampling == "random":
        rng = np.random.default_rng(scene_seed)
        frame_ids = sorted(rng.choice(frame_pool, args.num_frames, replace=False).tolist())
    else:
        candidates = [frame_id for frame_id in frame_pool if frame_id >= args.start_frame]
        frame_ids = candidates[: args.num_frames]
        if len(frame_ids) < args.num_frames:
            raise RuntimeError(
                f"{scene_dir} has fewer than {args.num_frames} frames from {args.start_frame}"
            )
    image_paths = [scene_dir / "images_4" / f"{frame_id:03d}_{args.camera}.jpg" for frame_id in frame_ids]
    images = load_and_preprocess_images(
        [str(path) for path in image_paths], image_resolution=args.image_resolution
    ).cuda()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    pred_depth = numpy_prediction(predictions["depth"])[..., 0]
    pred_w2c_3x4 = numpy_prediction(extrinsics)
    pred_w2c = np.tile(np.eye(4, dtype=np.float64), (len(frame_ids), 1, 1))
    pred_w2c[:, :3, :4] = pred_w2c_3x4
    gt_w2c = load_gt_w2c(scene_dir, frame_ids, args.camera)

    pred_values = []
    gt_values = []
    per_frame_values = []
    for prediction_index, frame_id in enumerate(frame_ids):
        gt_depth = np.load(
            scene_dir / "depth_flows_4" / f"{frame_id:03d}_{args.camera}.npy"
        )[..., 0].astype(np.float32)
        gt_depth = transform_gt_depth(gt_depth, *pred_depth[prediction_index].shape)
        prediction = pred_depth[prediction_index]
        valid = np.isfinite(gt_depth) & (gt_depth > 0) & np.isfinite(prediction) & (prediction > 0)
        pred_values.append(prediction[valid])
        gt_values.append(gt_depth[valid])
        per_frame_values.append((frame_id, prediction[valid], gt_depth[valid]))

    all_prediction = np.concatenate(pred_values)
    all_target = np.concatenate(gt_values)
    clip_scale = float(np.median(all_target / all_prediction))
    clip_depth_metrics = depth_metrics(all_prediction * clip_scale, all_target)

    per_frame_oracle = []
    for frame_id, prediction, target in per_frame_values:
        scale = float(np.median(target / prediction))
        per_frame_oracle.append({"frame_id": frame_id, "scale": scale, **depth_metrics(prediction * scale, target)})
    oracle_pixels = sum(item["num_lidar_pixels"] for item in per_frame_oracle)
    oracle_summary = {
        key: float(sum(item[key] * item["num_lidar_pixels"] for item in per_frame_oracle) / oracle_pixels)
        for key in (
            "abs_rel",
            "rmse_m",
            "delta_1.25_percent",
            "delta_1.25_sq_percent",
            "delta_1.25_cu_percent",
        )
    }

    pair_errors = pose_errors(pred_w2c, gt_w2c)
    scene_output = args.output_dir / f"scene-{scene_name}"
    scene_output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "pose_enc": numpy_prediction(predictions["pose_enc"]),
        "extrinsics": pred_w2c_3x4.astype(np.float32),
        "intrinsics": numpy_prediction(intrinsics).astype(np.float32),
        "depth": numpy_prediction(predictions["depth"]).astype(np.float32),
        "depth_conf": numpy_prediction(predictions["depth_conf"]).astype(np.float32),
        "camera_and_register_tokens": numpy_prediction(predictions["camera_and_register_tokens"]).astype(np.float32),
        "gt_world_to_camera": gt_w2c.astype(np.float32),
    }
    np.savez_compressed(scene_output / "predictions.npz", **arrays)

    result = {
        "scene": scene_name,
        "split": args.split,
        "camera": args.camera,
        "camera_name": "FRONT" if args.camera == 0 else f"camera_{args.camera}",
        "frame_pool_size": len(frame_pool),
        "sampled_frame_ids": frame_ids,
        "scene_seed": scene_seed,
        "sampling": args.sampling,
        "input_tensor_shape": list(images.shape),
        "clip_depth_scale": clip_scale,
        "depth": clip_depth_metrics,
        "per_frame_scale_oracle_depth": oracle_summary,
        "pose": {
            "auc_3_deg": auc(pair_errors, 3),
            "auc_30_deg": auc(pair_errors, 30),
            "mean_pair_error_deg": float(np.mean(pair_errors)),
            "median_pair_error_deg": float(np.median(pair_errors)),
            "num_frame_pairs": int(len(pair_errors)),
        },
        "inference_seconds": inference_seconds,
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30),
        "predictions": str(scene_output / "predictions.npz"),
    }
    (scene_output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    del images, predictions
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(str(args.checkpoint))
    results = []
    for scene_index, scene in enumerate(args.scenes):
        result = evaluate_scene(args, model, scene, scene_index)
        results.append(result)
        print(json.dumps(result, ensure_ascii=True))

    summary_keys = ("auc_3_deg", "auc_30_deg")
    aggregate = {
        "protocol": (
            "Paper-style 10 random frames jointly inferred per Waymo scene; paper pose/depth metrics "
            "on processed front-camera poses and projected sparse LiDAR. Custom diagnostic, not an "
            "official VGGT-Omega Waymo benchmark."
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "sampling": args.sampling,
        "num_frames_per_scene": args.num_frames,
        "num_scenes": len(results),
        "summary_scene_mean": {
            "auc_3_deg": float(np.mean([item["pose"][summary_keys[0]] for item in results])),
            "auc_30_deg": float(np.mean([item["pose"][summary_keys[1]] for item in results])),
            "abs_rel": float(np.mean([item["depth"]["abs_rel"] for item in results])),
            "delta_1.25_percent": float(np.mean([item["depth"]["delta_1.25_percent"] for item in results])),
        },
        "scenes": results,
    }
    output = args.output_dir / "waymo_paper_style_metrics.json"
    output.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate["summary_scene_mean"], indent=2))


if __name__ == "__main__":
    main()
