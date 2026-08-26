#!/usr/bin/env python3
"""Export metric, rig-local Omega camera poses without any GT trajectory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_ufo_pose_override import (
    as_homogeneous,
    transform_intrinsics_to_ufo,
    unique_preprocessing_geometries,
)
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


FORBIDDEN_MANIFEST_KEYS = {
    "camera_to_world",
    "ego_pose",
    "ego_to_world",
    "gt_c2w_opencv",
    "gt_camera_to_world",
    "gt_intrinsics_ufo",
    "calibrated_intrinsics_ufo",
    "normalized_intrinsics",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    return parser.parse_args()


def reject_forbidden_keys(value, path="manifest"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_MANIFEST_KEYS or "sim3" in key.lower():
                raise ValueError(f"forbidden pose-free field {path}.{key}")
            reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_keys(child, f"{path}[{index}]")


def metric_scale_from_rig(raw_c2w, frame_ids, camera_ids, rig_camera_to_ego):
    ratios = []
    observations = []
    unique_frames = sorted(set(frame_ids.tolist()))
    unique_cameras = sorted(set(camera_ids.tolist()))
    real_centers = {
        camera: np.asarray(rig_camera_to_ego[camera], dtype=np.float64)[:3, 3]
        for camera in unique_cameras
    }
    for frame_id in unique_frames:
        by_camera = {
            camera: int(np.flatnonzero((frame_ids == frame_id) & (camera_ids == camera))[0])
            for camera in unique_cameras
        }
        for left, right in combinations(unique_cameras, 2):
            real = float(np.linalg.norm(real_centers[left] - real_centers[right]))
            predicted = float(np.linalg.norm(
                raw_c2w[by_camera[left], :3, 3] - raw_c2w[by_camera[right], :3, 3]
            ))
            if real <= 0 or predicted <= 1e-8:
                continue
            ratio = real / predicted
            ratios.append(ratio)
            observations.append((frame_id, left, right, real, predicted, ratio))
    if not ratios:
        raise ValueError("no valid fixed-rig baseline observations")
    return float(np.median(ratios)), observations


def apply_front0_gauge(raw_c2w, scale, frame_ids, camera_ids, front_camera="0"):
    first_frame = int(frame_ids.min())
    matches = np.flatnonzero((frame_ids == first_frame) & (camera_ids == front_camera))
    if len(matches) != 1:
        raise ValueError(f"expected one front camera at first frame {first_frame}")
    world_from_omega = np.linalg.inv(raw_c2w[int(matches[0])])
    local = np.einsum("ij,njk->nik", world_from_omega, raw_c2w)
    local[:, :3, 3] *= scale
    return local, first_frame


def baseline_metrics(local_c2w, frame_ids, camera_ids, rig_camera_to_ego):
    result = {}
    cameras = sorted(set(camera_ids.tolist()))
    for left, right in combinations(cameras, 2):
        real = float(np.linalg.norm(
            np.asarray(rig_camera_to_ego[left])[:3, 3]
            - np.asarray(rig_camera_to_ego[right])[:3, 3]
        ))
        predicted = []
        for frame_id in sorted(set(frame_ids.tolist())):
            li = int(np.flatnonzero((frame_ids == frame_id) & (camera_ids == left))[0])
            ri = int(np.flatnonzero((frame_ids == frame_id) & (camera_ids == right))[0])
            predicted.append(float(np.linalg.norm(
                local_c2w[li, :3, 3] - local_c2w[ri, :3, 3]
            )))
        errors = np.abs(np.asarray(predicted) - real)
        result[f"{left}-{right}"] = {
            "real_m": real,
            "predicted_m_median": float(np.median(predicted)),
            "absolute_error_m_mean": float(errors.mean()),
            "absolute_error_m_max": float(errors.max()),
        }
    return result


def export_window(model, manifest_path, args):
    manifest = json.loads(manifest_path.read_text())
    reject_forbidden_keys(manifest)
    contract = manifest.get("pose_contract", {})
    if contract.get("name") != "rig_pose_free_v1":
        raise ValueError(f"manifest lacks rig_pose_free_v1 contract: {manifest_path}")
    entries = manifest["images"]
    image_paths = [item["path"] for item in entries]
    images = load_and_preprocess_images(
        image_paths, image_resolution=args.image_resolution
    ).cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    raw_w2c = as_homogeneous(extrinsics[0].detach().float().cpu().numpy())
    raw_c2w = np.linalg.inv(raw_w2c)
    frame_ids = np.asarray([item["frame_id"] for item in entries], dtype=np.int32)
    camera_ids = np.asarray([item["camera_id"] for item in entries])
    roles = np.asarray([item["role"] for item in entries])
    scale, observations = metric_scale_from_rig(
        raw_c2w, frame_ids, camera_ids, manifest["rig_camera_to_ego"]
    )
    local_c2w, gauge_frame = apply_front0_gauge(raw_c2w, scale, frame_ids, camera_ids)
    local_w2c = np.linalg.inv(local_c2w)
    opencv_to_dataset = np.asarray(manifest["opencv_to_dataset"], dtype=np.float64)
    local_native = local_c2w @ np.linalg.inv(opencv_to_dataset)
    predicted_intrinsics = intrinsics[0].detach().float().cpu().numpy()
    predicted_intrinsics_ufo, preprocessing = transform_intrinsics_to_ufo(
        predicted_intrinsics, image_paths, manifest["ufo_image_size"],
        image_resolution=args.image_resolution,
    )
    baselines = baseline_metrics(
        local_c2w, frame_ids, camera_ids, manifest["rig_camera_to_ego"]
    )

    scene_output = (
        args.output_dir / f"start_{manifest['start_index']:03d}" / manifest["scene_name"]
    )
    scene_output.mkdir(parents=True, exist_ok=True)
    output_path = scene_output / "omega_pose_override.npz"
    np.savez_compressed(
        output_path,
        scene_name=np.asarray(manifest["scene_name"]),
        scope=np.asarray("all"),
        coordinate_frame=np.asarray("rig_local_metric"),
        metric_scale_source=np.asarray("fixed_camera_to_ego_baselines"),
        world_gauge=np.asarray("first_timestamp_front_camera"),
        frame_ids=frame_ids,
        camera_ids=camera_ids,
        roles=roles,
        omega_w2c_raw=raw_w2c.astype(np.float32),
        omega_c2w_raw=raw_c2w.astype(np.float32),
        omega_w2c_rig_local=local_w2c.astype(np.float32),
        omega_c2w_rig_local=local_c2w.astype(np.float32),
        omega_camera_to_world_rig_local=local_native.astype(np.float32),
        predicted_intrinsics_ufo=predicted_intrinsics_ufo.astype(np.float32),
        rig_metric_scale=np.asarray(scale),
        gauge_frame_id=np.asarray(gauge_frame),
        gauge_camera_id=np.asarray("0"),
    )
    metrics = {
        "scene_name": manifest["scene_name"],
        "start_index": manifest["start_index"],
        "num_images": len(entries),
        "coordinate_frame": "rig_local_metric",
        "metric_scale_source": "fixed_camera_to_ego_baselines",
        "world_gauge": {"frame_id": gauge_frame, "camera_id": "0"},
        "scale": scale,
        "scale_observations": len(observations),
        "baselines": baselines,
        "front0_identity_max_abs_error": float(np.abs(
            local_c2w[np.flatnonzero((frame_ids == gauge_frame) & (camera_ids == "0"))[0]]
            - np.eye(4)
        ).max()),
        "omega_image_size": list(predictions["images"].shape[-2:]),
        "ufo_image_size": manifest["ufo_image_size"],
        "preprocessing_geometries": unique_preprocessing_geometries(preprocessing),
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "manifest": str(manifest_path.resolve()),
        "output": str(output_path.resolve()),
    }
    (scene_output / "rig_pose_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    del images, predictions, extrinsics, intrinsics
    torch.cuda.empty_cache()
    return metrics


def main():
    args = parse_args()
    model = VGGTOmega().eval()
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.cuda()
    del state_dict
    results = []
    for manifest in args.manifests:
        print(f"[rig-pose-free] exporting {manifest}", flush=True)
        results.append(export_window(model, manifest, args))
    scales = np.asarray([item["scale"] for item in results])
    summary = {
        "contract": "rig_pose_free_v1",
        "num_windows": len(results),
        "scale_mean": float(scales.mean()),
        "scale_std": float(scales.std()),
        "scale_coefficient_of_variation": float(scales.std() / scales.mean()),
        "windows": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sequence_rig_pose_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
