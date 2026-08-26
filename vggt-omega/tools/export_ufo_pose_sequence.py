#!/usr/bin/env python3
"""Export aligned VGGT-Omega pose overrides for multiple UFO windows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_ufo_pose_override import (
    apply_similarity,
    as_homogeneous,
    orientation_aware_similarity,
    pose_metrics,
    intrinsics_metrics,
    transform_intrinsics_to_ufo,
    unique_preprocessing_geometries,
)
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("context", "all"), required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    return parser.parse_args()


def export_window(model, manifest_path, args):
    manifest = json.loads(manifest_path.read_text())
    entries = [
        item for item in manifest["images"]
        if args.scope == "all" or item["role"] == "context"
    ]
    image_paths = [item["path"] for item in entries]
    missing = [path for path in image_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"manifest contains missing images: {missing[:3]}")

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
    gt_c2w = np.asarray([item["gt_c2w_opencv"] for item in entries], dtype=np.float64)
    gt_native = np.asarray([item["gt_camera_to_world"] for item in entries], dtype=np.float64)
    scale, rotation, translation = orientation_aware_similarity(raw_c2w, gt_c2w)
    aligned_c2w = apply_similarity(raw_c2w, scale, rotation, translation)
    aligned_w2c = np.linalg.inv(aligned_c2w)
    opencv_to_dataset = np.asarray(manifest["opencv_to_dataset"], dtype=np.float64)
    aligned_native = aligned_c2w @ np.linalg.inv(opencv_to_dataset)
    predicted_intrinsics = intrinsics[0].detach().float().cpu().numpy().astype(np.float32)
    predicted_intrinsics_ufo, preprocessing_geometry = transform_intrinsics_to_ufo(
        predicted_intrinsics,
        image_paths,
        manifest["ufo_image_size"],
        image_resolution=args.image_resolution,
    )
    gt_intrinsics_ufo = np.asarray(
        [item["gt_intrinsics_ufo"] for item in entries], dtype=np.float64
    )
    frame_ids = np.asarray([item["frame_id"] for item in entries], dtype=np.int32)
    camera_ids = np.asarray([item["camera_id"] for item in entries])
    roles = np.asarray([item["role"] for item in entries])

    window_output = args.output_dir / f"start_{manifest['start_index']:03d}"
    scene_output = window_output / manifest["scene_name"]
    scene_output.mkdir(parents=True, exist_ok=True)
    output_path = scene_output / "omega_pose_override.npz"
    np.savez_compressed(
        output_path,
        scene_name=np.asarray(manifest["scene_name"]),
        scope=np.asarray(args.scope),
        frame_ids=frame_ids,
        camera_ids=camera_ids,
        roles=roles,
        omega_w2c_raw=raw_w2c.astype(np.float32),
        omega_c2w_raw=raw_c2w.astype(np.float32),
        omega_w2c_aligned=aligned_w2c,
        omega_c2w_aligned=aligned_c2w,
        omega_camera_to_world_aligned=aligned_native,
        gt_c2w=gt_c2w,
        gt_camera_to_world=gt_native,
        predicted_intrinsics=predicted_intrinsics,
        predicted_intrinsics_ufo=predicted_intrinsics_ufo.astype(np.float32),
        gt_intrinsics_ufo=gt_intrinsics_ufo,
        sim3_scale=np.asarray(scale),
        sim3_rotation=rotation,
        sim3_translation=translation,
    )
    metrics = {
        "scene_name": manifest["scene_name"],
        "start_index": manifest["start_index"],
        "scope": args.scope,
        "num_images": len(entries),
        "pose": pose_metrics(aligned_c2w, gt_c2w, frame_ids, camera_ids),
        "intrinsics": intrinsics_metrics(
            predicted_intrinsics_ufo, gt_intrinsics_ufo, manifest["ufo_image_size"]
        ),
        "omega_image_size": list(predictions["images"].shape[-2:]),
        "ufo_image_size": manifest["ufo_image_size"],
        "preprocessing_geometries": unique_preprocessing_geometries(
            preprocessing_geometry
        ),
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "manifest": str(manifest_path.resolve()),
        "output": str(output_path.resolve()),
    }
    (scene_output / "pose_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
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
        print(f"[{args.scope}] exporting {manifest}", flush=True)
        results.append(export_window(model, manifest, args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"scope": args.scope, "num_windows": len(results), "windows": results}
    (args.output_dir / "sequence_pose_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
