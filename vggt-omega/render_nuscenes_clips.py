#!/usr/bin/env python3
"""Render nuScenes clips from sparse VGGT-Omega context frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity

from render_waymo_clips import (
    build_aligned_point_cloud,
    create_writer,
    infer_context,
    masked_psnr,
    render_points,
)
from vggt_omega.models import VGGTOmega


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=["scene-0103", "scene-0553"])
    parser.add_argument("--context-stride", type=int, default=12)
    parser.add_argument("--confidence-percentile", type=float, default=20.0)
    parser.add_argument("--max-points", type=int, default=4_000_000)
    parser.add_argument("--splat-radius", type=float, default=3.5)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--reuse-predictions", action="store_true")
    return parser.parse_args()


def transform_matrix(record: dict) -> np.ndarray:
    quaternion_wxyz = record["rotation"]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]]
    ).as_matrix()
    matrix[:3, 3] = record["translation"]
    return matrix


def load_tables(data_root: Path) -> dict:
    metadata_root = data_root / "v1.0-mini"
    names = ("scene", "sample_data", "calibrated_sensor", "sensor", "ego_pose")
    tables = {name: json.loads((metadata_root / f"{name}.json").read_text()) for name in names}
    for name in names[1:]:
        tables[name] = {record["token"]: record for record in tables[name]}
    tables["scene"] = {record["name"]: record for record in tables["scene"]}
    return tables


def collect_front_clip(data_root: Path, tables: dict, scene_name: str) -> tuple[list[Path], list[int], np.ndarray, np.ndarray, list[int]]:
    scene = tables["scene"][scene_name]
    sensor_table = tables["sensor"]
    calibration_table = tables["calibrated_sensor"]
    candidates = []
    for record in tables["sample_data"].values():
        if record["sample_token"] != scene["first_sample_token"] or not record["is_key_frame"]:
            continue
        calibration = calibration_table[record["calibrated_sensor_token"]]
        if sensor_table[calibration["sensor_token"]]["channel"] == "CAM_FRONT":
            candidates.append(record)
    if len(candidates) != 1:
        raise RuntimeError(f"Could not identify first CAM_FRONT frame for {scene_name}")

    records = []
    record = candidates[0]
    while True:
        records.append(record)
        if not record["next"]:
            break
        record = tables["sample_data"][record["next"]]

    image_paths = [data_root / record["filename"] for record in records]
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    camera_to_world = []
    intrinsics = []
    for record in records:
        calibration = calibration_table[record["calibrated_sensor_token"]]
        ego_pose = tables["ego_pose"][record["ego_pose_token"]]
        camera_to_world.append(transform_matrix(ego_pose) @ transform_matrix(calibration))
        intrinsics.append(np.asarray(calibration["camera_intrinsic"], dtype=np.float64))
    camera_to_world = np.stack(camera_to_world)
    camera_to_local = np.einsum("ij,njk->nik", np.linalg.inv(camera_to_world[0]), camera_to_world)
    timestamps = [int(record["timestamp"]) for record in records]
    key_positions = [index for index, record in enumerate(records) if record["is_key_frame"]]
    return image_paths, timestamps, camera_to_local, np.stack(intrinsics), key_positions


def aggregate(items: list[dict]) -> dict:
    return {
        "frame_count": len(items),
        "raw_coverage_percent": float(np.mean([item["raw_coverage_percent"] for item in items])),
        "splat_coverage_percent": float(np.mean([item["splat_coverage_percent"] for item in items])),
        "masked_psnr_db": float(np.mean([item["masked_psnr_db"] for item in items])),
        "full_frame_ssim_black_holes": float(np.mean([item["full_frame_ssim_black_holes"] for item in items])),
    }


def render_scene(args: argparse.Namespace, tables: dict, model: VGGTOmega, scene_name: str) -> dict:
    source_paths, timestamps, target_cameras, target_intrinsics, key_positions = collect_front_clip(
        args.data_root, tables, scene_name
    )
    context_positions = list(range(0, len(source_paths), args.context_stride))
    context_paths = [source_paths[index] for index in context_positions]
    output_dir = args.output_root / scene_name
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
        predictions, inference_seconds = infer_context(model, context_paths, args.image_resolution)
        np.savez_compressed(
            prediction_path,
            **{key: predictions[key] for key in ("depth", "depth_conf", "images", "extrinsic", "intrinsic")},
        )
    height, width = predictions["depth"].shape[1:3]
    points, colors, alignment = build_aligned_point_cloud(
        predictions,
        target_cameras[context_positions],
        args.confidence_percentile,
        args.max_points,
    )

    original = cv2.imread(str(source_paths[0]))
    source_height, source_width = original.shape[:2]
    scales = np.array([[width / source_width, width / source_width, width / source_width],
                       [height / source_height, height / source_height, height / source_height],
                       [1, 1, 1]], dtype=np.float64)
    render_intrinsics = target_intrinsics * scales

    source_writer = create_writer(output_dir / "01_source_front.mp4", args.fps, (width, height))
    reconstruction_writer = create_writer(
        output_dir / "02_vggt_omega_point_reconstruction_front.mp4", args.fps, (width, height)
    )
    comparison_writer = create_writer(
        output_dir / "03_source_reconstruction_mask.mp4", args.fps, (width * 3, height)
    )

    context_set = set(context_positions)
    per_frame = []
    for position, (source_path, camera, intrinsic) in enumerate(
        zip(source_paths, target_cameras, render_intrinsics)
    ):
        source_bgr = cv2.resize(cv2.imread(str(source_path)), (width, height), interpolation=cv2.INTER_AREA)
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        reconstruction_rgb, mask, raw_coverage = render_points(
            points, colors, camera, intrinsic, height, width, args.splat_radius
        )
        reconstruction_bgr = cv2.cvtColor(reconstruction_rgb, cv2.COLOR_RGB2BGR)
        mask_bgr = np.repeat((mask[..., None] * 255).astype(np.uint8), 3, axis=2)
        source_writer.write(source_bgr)
        reconstruction_writer.write(reconstruction_bgr)
        comparison_writer.write(np.concatenate((source_bgr, reconstruction_bgr, mask_bgr), axis=1))
        per_frame.append(
            {
                "frame_position": position,
                "timestamp_us": timestamps[position],
                "is_context": position in context_set,
                "is_nuscenes_keyframe": position in key_positions,
                "raw_coverage_percent": raw_coverage * 100,
                "splat_coverage_percent": float(mask.mean() * 100),
                "masked_psnr_db": masked_psnr(source_rgb, reconstruction_rgb, mask),
                "full_frame_ssim_black_holes": float(
                    structural_similarity(source_rgb, reconstruction_rgb, channel_axis=2, data_range=255)
                ),
            }
        )
        if position % 30 == 0 or position == len(source_paths) - 1:
            print(f"{scene_name}: rendered {position + 1}/{len(source_paths)}")

    source_writer.release()
    reconstruction_writer.release()
    comparison_writer.release()
    heldout = [item for item in per_frame if not item["is_context"]]
    contexts = [item for item in per_frame if item["is_context"]]
    duration = (timestamps[-1] - timestamps[0]) / 1e6
    metadata = {
        "protocol": "Custom sparse-context point-cloud reprojection; not a paper benchmark or official novel-view renderer",
        "dataset": "nuScenes v1.0-mini",
        "scene": scene_name,
        "camera": "CAM_FRONT",
        "source_frame_count": len(source_paths),
        "source_duration_seconds": duration,
        "measured_source_fps": (len(source_paths) - 1) / duration,
        "output_fps": args.fps,
        "context_stride": args.context_stride,
        "context_positions": context_positions,
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
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output_root.mkdir(parents=True, exist_ok=True)
    tables = load_tables(args.data_root)
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to("cuda")
    for scene in args.scenes:
        render_scene(args, tables, model, scene)
        torch.cuda.empty_cache()
    results = [
        json.loads(path.read_text()) for path in sorted(args.output_root.glob("scene-*/metrics.json"))
    ]
    summary = {
        "protocol": "Custom VGGT-Omega sparse-context nuScenes rendering",
        "scenes": [
            {key: result[key] for key in ("scene", "input_rgb_count", "rendered_frame_count", "inference_seconds", "alignment", "metrics_heldout")}
            for result in results
        ],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
