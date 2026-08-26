#!/usr/bin/env python3
"""Batch wrapper around the unmodified official VGGT-Omega demo functions."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from demo_gradio import handle_uploads, load_model, predictions_to_glb, run_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--waymo-root", type=Path)
    parser.add_argument("--waymo-scenes", nargs="*", default=[])
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--source-fps", type=float, default=10.0)
    parser.add_argument("--confidence-percentile", type=float, default=50.0)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--image-resolution", type=int, default=512)
    return parser.parse_args()


def encode_waymo_video(scene_dir: Path, destination: Path, fps: float) -> int:
    image_paths = sorted((scene_dir / "images").glob("*_0.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No Waymo front images in {scene_dir}")
    first = cv2.imread(str(image_paths[0]))
    height, width = first.shape[:2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {destination}")
    for image_path in image_paths:
        writer.write(cv2.imread(str(image_path)))
    writer.release()
    return len(image_paths)


def write_context_video(image_paths: list[Path], destination: Path, fps: float) -> None:
    first = cv2.imread(str(image_paths[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {destination}")
    for image_path in image_paths:
        writer.write(cv2.imread(str(image_path)))
    writer.release()


def write_rgb_depth_video(predictions: dict, destination: Path, fps: float) -> None:
    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    images_bgr = ((images * 255).clip(0, 255).astype(np.uint8))[..., ::-1]
    depth = predictions["depth"][..., 0]
    valid = np.isfinite(depth) & (depth > 0)
    inverse_depth = np.zeros_like(depth, dtype=np.float32)
    inverse_depth[valid] = 1.0 / depth[valid]
    low, high = np.percentile(inverse_depth[valid], (2, 98))
    scale = max(float(high - low), 1e-6)
    height, width = depth.shape[-2:]
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {destination}")
    for image, frame_inverse_depth in zip(images_bgr, inverse_depth):
        normalized = ((frame_inverse_depth - low) / scale * 255).clip(0, 255).astype(np.uint8)
        writer.write(np.concatenate((image, cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)), axis=1))
    writer.release()


def run_one(args: argparse.Namespace, model, name: str, video_path: Path, source_frames: int | None) -> dict:
    output_dir = args.output_root / name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    target_dir_string, official_image_paths = handle_uploads(
        str(video_path), [], video_sample_fps=args.sample_fps
    )
    target_dir = Path(target_dir_string)
    context_dir = output_dir / "images"
    shutil.copytree(target_dir / "images", context_dir)
    image_paths = sorted(context_dir.glob("*"))
    write_context_video(image_paths, output_dir / "01_official_sampled_context.mp4", args.sample_fps)

    started = time.perf_counter()
    predictions = run_model(str(target_dir), model, args.image_resolution)
    inference_and_postprocess_seconds = time.perf_counter() - started
    np.savez_compressed(
        output_dir / "predictions.npz",
        **{
            key: value
            for key, value in predictions.items()
            if key in ("depth", "depth_conf", "images", "pose_enc", "extrinsic", "intrinsic")
        },
    )
    scene = predictions_to_glb(
        predictions,
        conf_thres=args.confidence_percentile,
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=True,
        mask_sky=False,
        target_dir=str(target_dir),
        max_points=args.max_points,
    )
    glb_path = output_dir / "02_official_reconstruction.glb"
    scene.export(glb_path)
    write_rgb_depth_video(
        predictions,
        output_dir / "03_official_input_rgb_predicted_depth.mp4",
        args.sample_fps,
    )
    shutil.rmtree(target_dir)

    metadata = {
        "protocol": "Official demo_gradio.py functions; no custom renderer",
        "source_video": str(video_path.resolve()),
        "source_frame_count": source_frames,
        "source_fps": args.source_fps if source_frames is not None else None,
        "official_video_sample_fps": args.sample_fps,
        "sampled_rgb_count": len(image_paths),
        "preprocessed_shape": list(predictions["images"].shape),
        "confidence_percentile": args.confidence_percentile,
        "max_points": args.max_points,
        "inference_and_postprocess_seconds": inference_and_postprocess_seconds,
        "glb": str(glb_path.resolve()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.video is not None:
        jobs.append((args.video.stem, args.video, None))
    if args.waymo_scenes:
        if args.waymo_root is None:
            raise ValueError("--waymo-root is required with --waymo-scenes")
        for scene in args.waymo_scenes:
            video_path = args.output_root / "source_videos" / f"waymo-{scene}-front-10fps.mp4"
            frame_count = encode_waymo_video(args.waymo_root / scene, video_path, args.source_fps)
            jobs.append((f"waymo-{scene}", video_path, frame_count))
    if not jobs:
        raise ValueError("Provide --video or --waymo-scenes")

    model = load_model(str(args.checkpoint))
    results = [run_one(args, model, name, video, count) for name, video, count in jobs]
    (args.output_root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
