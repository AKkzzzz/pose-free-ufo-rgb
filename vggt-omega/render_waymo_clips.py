#!/usr/bin/env python3
"""Render complete Waymo clips from sparse VGGT-Omega RGB context frames.

This is an evaluation renderer built around the released camera/depth model. It
is not an official VGGT-Omega novel-view renderer: predicted depth is unprojected
into a colored point cloud, aligned to Waymo poses, then z-buffered into every
target camera in the clip.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity

from run_video_inference import tensor_predictions_to_numpy, unproject_depth
from visual_util import depth_edge
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


OPENCV_TO_WAYMO = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=["124", "068", "016"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--context-stride", type=int, default=10)
    parser.add_argument("--confidence-percentile", type=float, default=50.0)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--splat-radius", type=float, default=2.5)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--reuse-predictions", action="store_true")
    return parser.parse_args()


def camera_centers(world_to_camera: np.ndarray) -> np.ndarray:
    rotation = world_to_camera[:, :3, :3]
    translation = world_to_camera[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(rotation, (0, 2, 1)), translation)


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit target = scale * rotation @ source + translation with Umeyama."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    source_variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = float(np.sum(singular_values * sign) / max(source_variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def load_waymo_cameras(scene_dir: Path, frame_ids: list[int], camera: int) -> tuple[np.ndarray, np.ndarray]:
    camera_to_ego = np.loadtxt(scene_dir / "extrinsics" / f"{camera}.txt", dtype=np.float64)
    camera_to_world = []
    for frame_id in frame_ids:
        ego_to_world = np.loadtxt(scene_dir / "ego_pose" / f"{frame_id:03d}.txt", dtype=np.float64)
        camera_to_world.append(ego_to_world @ camera_to_ego @ OPENCV_TO_WAYMO)
    camera_to_world = np.stack(camera_to_world)
    world_to_first = np.linalg.inv(camera_to_world[0])
    camera_to_local = np.einsum("ij,njk->nik", world_to_first, camera_to_world)
    return camera_to_local, np.loadtxt(scene_dir / "intrinsics" / f"{camera}.txt", dtype=np.float64)


def infer_context(
    model: VGGTOmega,
    image_paths: list[Path],
    image_resolution: int,
) -> tuple[dict[str, np.ndarray], float]:
    images = load_and_preprocess_images(
        [str(path) for path in image_paths], mode="balanced", image_resolution=image_resolution
    ).to("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        predictions = model(images)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    result = tensor_predictions_to_numpy(predictions)
    del images, predictions
    return result, inference_seconds


def build_aligned_point_cloud(
    predictions: dict[str, np.ndarray],
    target_context_cameras: np.ndarray,
    confidence_percentile: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    predicted_points = unproject_depth(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )
    predicted_centers = camera_centers(predictions["extrinsic"])
    target_centers = target_context_cameras[:, :3, 3]
    target_centered = target_centers - target_centers.mean(axis=0)
    target_variance = float(np.mean(np.sum(target_centered**2, axis=1)))
    alignment_method = "camera-center Umeyama Sim(3)"
    if target_variance < 1e-4 or np.linalg.matrix_rank(target_centered) < 2:
        predicted_camera_to_world = np.transpose(predictions["extrinsic"][:, :3, :3], (0, 2, 1))
        relative_rotations = target_context_cameras[:, :3, :3] @ np.transpose(
            predicted_camera_to_world, (0, 2, 1)
        )
        rotation = Rotation.from_matrix(relative_rotations).mean().as_matrix()
        predicted_centered = predicted_centers - predicted_centers.mean(axis=0)
        rotated_centered = predicted_centered @ rotation.T
        denominator = float(np.sum(rotated_centered**2))
        scale = float(np.sum(rotated_centered * target_centered) / denominator) if denominator > 1e-12 else 1.0
        if target_variance < 1e-8 or scale <= 1e-6:
            scale = 1.0
        translation = target_centers.mean(axis=0) - scale * (rotation @ predicted_centers.mean(axis=0))
        alignment_method = "camera-orientation alignment with fixed/least-squares scale (degenerate centers)"
    else:
        scale, rotation, translation = fit_similarity(predicted_centers, target_centers)

    aligned_centers = scale * (predicted_centers @ rotation.T) + translation
    camera_errors = np.linalg.norm(aligned_centers - target_centers, axis=1)

    points = predicted_points.reshape(-1, 3)
    colors = predictions["images"]
    if colors.ndim == 4 and colors.shape[1] == 3:
        colors = np.transpose(colors, (0, 2, 3, 1))
    colors = (colors.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    confidence = predictions["depth_conf"].reshape(-1)
    edges = depth_edge(predictions["depth"][..., 0], rtol=0.03).reshape(-1)

    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence) & ~edges
    threshold = float(np.percentile(confidence[finite], confidence_percentile))
    keep = finite & (confidence >= threshold) & (confidence > 1e-5)
    points = points[keep]
    colors = colors[keep]
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[indices]
        colors = colors[indices]

    points = scale * (points @ rotation.T) + translation
    alignment = {
        "method": alignment_method,
        "sim3_scale": scale,
        "camera_ate_rmse_m": float(np.sqrt(np.mean(camera_errors**2))),
        "camera_ate_median_m": float(np.median(camera_errors)),
        "confidence_threshold": threshold,
        "point_count": int(len(points)),
    }
    return points.astype(np.float32), colors, alignment


def fill_small_splat_holes(
    image: np.ndarray, mask: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    if radius <= 0 or not mask.any():
        return image, mask
    source = (~mask).astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    seeds = image[mask]
    fill = (~mask) & (distance <= radius) & (labels > 0)
    if seeds.size and fill.any():
        image[fill] = seeds[np.clip(labels[fill] - 1, 0, len(seeds) - 1)]
        mask[fill] = True
    return image, mask


def render_points(
    points: np.ndarray,
    colors: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    splat_radius: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    world_to_camera = np.linalg.inv(camera_to_world)
    camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (z > 0.05)
    camera_points = camera_points[valid]
    frame_colors = colors[valid]
    z = z[valid]
    u = np.rint(intrinsic[0, 0] * camera_points[:, 0] / z + intrinsic[0, 2]).astype(np.int32)
    v = np.rint(intrinsic[1, 1] * camera_points[:, 1] / z + intrinsic[1, 2]).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z, frame_colors = u[inside], v[inside], z[inside], frame_colors[inside]

    pixel = v.astype(np.int64) * width + u
    order = np.lexsort((z, pixel))
    sorted_pixel = pixel[order]
    first = np.empty(len(order), dtype=bool)
    if len(order):
        first[0] = True
        first[1:] = sorted_pixel[1:] != sorted_pixel[:-1]
    selected = order[first]

    image = np.zeros((height * width, 3), dtype=np.uint8)
    mask = np.zeros(height * width, dtype=bool)
    image[pixel[selected]] = frame_colors[selected]
    mask[pixel[selected]] = True
    image = image.reshape(height, width, 3)
    mask = mask.reshape(height, width)
    raw_coverage = float(mask.mean())
    image, mask = fill_small_splat_holes(image, mask, splat_radius)
    return image, mask, raw_coverage


def masked_psnr(reference: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    error = reference[mask].astype(np.float64) - reconstruction[mask].astype(np.float64)
    mse = float(np.mean(error**2))
    return float(10 * np.log10((255.0**2) / max(mse, 1e-12)))


def create_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    return writer


def render_scene(
    args: argparse.Namespace,
    scene: str,
    model: VGGTOmega | None,
) -> dict:
    scene_dir = args.data_root / scene
    all_source_paths = sorted((scene_dir / "images").glob(f"*_{args.camera}.jpg"))
    source_paths = [
        path
        for path in all_source_paths
        if (scene_dir / "ego_pose" / f"{int(path.name.split('_')[0]):03d}.txt").is_file()
    ]
    if not source_paths:
        raise FileNotFoundError(f"No camera {args.camera} images in {scene_dir}")
    frame_ids = [int(path.name.split("_")[0]) for path in source_paths]
    context_positions = list(range(0, len(source_paths), args.context_stride))
    context_paths = [source_paths[index] for index in context_positions]
    context_frame_ids = [frame_ids[index] for index in context_positions]

    output_dir = args.output_root / f"scene-{scene}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "context_predictions.npz"
    previous_metadata_path = output_dir / "metrics.json"
    if args.reuse_predictions and prediction_path.is_file():
        with np.load(prediction_path) as loaded:
            predictions = {key: np.array(loaded[key]) for key in loaded.files}
        inference_seconds = None
        if previous_metadata_path.is_file():
            inference_seconds = json.loads(previous_metadata_path.read_text()).get("inference_seconds")
    else:
        if model is None:
            raise RuntimeError("Model is required when cached predictions are unavailable")
        predictions, inference_seconds = infer_context(model, context_paths, args.image_resolution)
        np.savez_compressed(
            prediction_path,
            **{
                key: predictions[key]
                for key in ("depth", "depth_conf", "images", "extrinsic", "intrinsic")
            },
        )

    height, width = predictions["depth"].shape[1:3]
    target_cameras, waymo_intrinsic = load_waymo_cameras(scene_dir, frame_ids, args.camera)
    context_cameras = target_cameras[context_positions]
    points, colors, alignment = build_aligned_point_cloud(
        predictions,
        context_cameras,
        args.confidence_percentile,
        args.max_points,
    )

    source_height, source_width = cv2.imread(str(source_paths[0])).shape[:2]
    scale_x, scale_y = width / source_width, height / source_height
    target_intrinsic = np.array(
        [
            [waymo_intrinsic[0] * scale_x, 0, waymo_intrinsic[2] * scale_x],
            [0, waymo_intrinsic[1] * scale_y, waymo_intrinsic[3] * scale_y],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    source_writer = create_writer(output_dir / "01_source_front.mp4", args.fps, (width, height))
    reconstruction_writer = create_writer(
        output_dir / "02_vggt_omega_point_reconstruction_front.mp4", args.fps, (width, height)
    )
    comparison_writer = create_writer(
        output_dir / "03_source_reconstruction_mask.mp4", args.fps, (width * 3, height)
    )

    per_frame = []
    context_set = set(context_positions)
    for position, (frame_id, source_path, camera_to_world) in enumerate(
        zip(frame_ids, source_paths, target_cameras)
    ):
        source_bgr = cv2.resize(cv2.imread(str(source_path)), (width, height), interpolation=cv2.INTER_AREA)
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        reconstruction_rgb, valid_mask, raw_coverage = render_points(
            points,
            colors,
            camera_to_world,
            target_intrinsic,
            height,
            width,
            args.splat_radius,
        )
        reconstruction_bgr = cv2.cvtColor(reconstruction_rgb, cv2.COLOR_RGB2BGR)
        mask_bgr = np.repeat((valid_mask[..., None] * 255).astype(np.uint8), 3, axis=2)

        source_writer.write(source_bgr)
        reconstruction_writer.write(reconstruction_bgr)
        comparison_writer.write(np.concatenate((source_bgr, reconstruction_bgr, mask_bgr), axis=1))

        full_ssim = structural_similarity(source_rgb, reconstruction_rgb, channel_axis=2, data_range=255)
        per_frame.append(
            {
                "frame_id": frame_id,
                "is_context": position in context_set,
                "raw_coverage_percent": raw_coverage * 100,
                "splat_coverage_percent": float(valid_mask.mean() * 100),
                "masked_psnr_db": masked_psnr(source_rgb, reconstruction_rgb, valid_mask),
                "full_frame_ssim_black_holes": float(full_ssim),
            }
        )
        if position % 25 == 0 or position == len(source_paths) - 1:
            print(f"scene {scene}: rendered {position + 1}/{len(source_paths)}")

    source_writer.release()
    reconstruction_writer.release()
    comparison_writer.release()

    def aggregate(items: list[dict]) -> dict:
        return {
            "frame_count": len(items),
            "raw_coverage_percent": float(np.mean([item["raw_coverage_percent"] for item in items])),
            "splat_coverage_percent": float(np.mean([item["splat_coverage_percent"] for item in items])),
            "masked_psnr_db": float(np.mean([item["masked_psnr_db"] for item in items])),
            "full_frame_ssim_black_holes": float(
                np.mean([item["full_frame_ssim_black_holes"] for item in items])
            ),
        }

    heldout = [item for item in per_frame if not item["is_context"]]
    contexts = [item for item in per_frame if item["is_context"]]
    metadata = {
        "protocol": "Custom sparse-context point-cloud reprojection; not an official VGGT-Omega renderer or paper benchmark",
        "scene": scene,
        "camera": args.camera,
        "source_fps": args.fps,
        "target_frame_ids": frame_ids,
        "excluded_missing_pose_frame_ids": [
            int(path.name.split("_")[0]) for path in all_source_paths if path not in source_paths
        ],
        "context_frame_ids": context_frame_ids,
        "context_stride": args.context_stride,
        "input_rgb_count": len(context_paths),
        "rendered_frame_count": len(source_paths),
        "resolution_hw": [height, width],
        "inference_seconds": inference_seconds,
        "alignment": alignment,
        "metrics_all": aggregate(per_frame),
        "metrics_context": aggregate(contexts),
        "metrics_heldout": aggregate(heldout),
        "per_frame": per_frame,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if args.context_stride <= 0:
        raise ValueError("--context-stride must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    args.output_root.mkdir(parents=True, exist_ok=True)
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to("cuda")

    summaries = []
    for scene in args.scenes:
        summaries.append(render_scene(args, scene, model))
        torch.cuda.empty_cache()
    all_metrics = []
    for metrics_path in sorted(args.output_root.glob("scene-*/metrics.json")):
        all_metrics.append(json.loads(metrics_path.read_text()))
    summary = {
        "protocol": "Custom VGGT-Omega sparse-context Waymo rendering",
        "scenes": [
            {
                "scene": item["scene"],
                "input_rgb_count": item["input_rgb_count"],
                "rendered_frame_count": item["rendered_frame_count"],
                "inference_seconds": item["inference_seconds"],
                "alignment": item["alignment"],
                "metrics_heldout": item["metrics_heldout"],
            }
            for item in all_metrics
        ],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
