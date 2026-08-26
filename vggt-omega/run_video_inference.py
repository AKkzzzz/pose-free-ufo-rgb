#!/usr/bin/env python3
"""Run VGGT-Omega on a video and export inspectable reconstruction artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from visual_util import predictions_to_glb
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--image-glob", help="Glob for an ordered image sequence, e.g. '/data/*_0.jpg'.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--confidence-percentile", type=float, default=50.0)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    return parser.parse_args()


def extract_frames(video_path: Path, image_dir: Path, sample_fps: float, max_frames: int) -> tuple[list[Path], dict]:
    if sample_fps <= 0:
        raise ValueError("--sample-fps must be positive")
    if max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(int(round((source_fps if source_fps > 0 else 1.0) / sample_fps)), 1)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    frame_index = 0
    while len(image_paths) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % interval == 0:
            image_path = image_dir / f"{len(image_paths):06d}.png"
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Could not write frame: {image_path}")
            image_paths.append(image_path)
        frame_index += 1
    capture.release()

    if not image_paths:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    return image_paths, {
        "source_fps": source_fps,
        "source_frame_count": source_frames,
        "sampling_interval": interval,
        "sampled_frame_count": len(image_paths),
    }


def collect_images(pattern: str, image_dir: Path, stride: int, max_frames: int) -> tuple[list[Path], dict]:
    if stride <= 0:
        raise ValueError("--image-stride must be positive")
    if max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    source_paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not source_paths:
        raise FileNotFoundError(f"No images matched: {pattern}")
    selected_paths = source_paths[::stride][:max_frames]
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for index, source_path in enumerate(selected_paths):
        destination = image_dir / f"{index:06d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, destination)
        image_paths.append(destination)
    return image_paths, {
        "image_glob": pattern,
        "source_frame_count": len(source_paths),
        "sampling_interval": stride,
        "sampled_frame_count": len(image_paths),
    }


def unproject_depth(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack(
        ((x - cx) / fx * depth, (y - cy) / fy * depth, depth),
        axis=-1,
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def tensor_predictions_to_numpy(predictions: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    result = {}
    for key, value in predictions.items():
        if not isinstance(value, torch.Tensor):
            continue
        array = value.detach().float().cpu().numpy()
        if array.shape[0] == 1:
            array = array[0]
        result[key] = array
    return result


def write_depth_video(predictions: dict[str, np.ndarray], output_path: Path, fps: float = 10.0) -> None:
    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    images = (images * 255).clip(0, 255).astype(np.uint8)
    images = images[..., ::-1]

    depth = predictions["depth"][..., 0]
    valid = np.isfinite(depth) & (depth > 0)
    inverse_depth = np.zeros_like(depth, dtype=np.float32)
    inverse_depth[valid] = 1.0 / depth[valid]
    lower, upper = np.percentile(inverse_depth[valid], (2, 98))
    scale = max(float(upper - lower), 1e-6)

    height, width = depth.shape[-2:]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output_path}")

    for image, frame_inverse_depth in zip(images, inverse_depth):
        normalized = ((frame_inverse_depth - lower) / scale * 255).clip(0, 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        writer.write(np.concatenate((image, colored_depth), axis=1))
    writer.release()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.video is not None and not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    if args.video is not None:
        image_paths, input_metadata = extract_frames(args.video, image_dir, args.sample_fps, args.max_frames)
    else:
        image_paths, input_metadata = collect_images(args.image_glob, image_dir, args.image_stride, args.max_frames)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = VGGTOmega().eval()
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    model_load_seconds = time.perf_counter() - started

    images = load_and_preprocess_images(
        [str(path) for path in image_paths],
        mode=args.resize_mode,
        image_resolution=args.image_resolution,
    ).to("cuda")
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        predictions = model(images)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions_np = tensor_predictions_to_numpy(predictions)
    del predictions, model, images, state_dict
    torch.cuda.empty_cache()

    save_keys = ("depth", "depth_conf", "images", "pose_enc", "extrinsic", "intrinsic")
    np.savez(args.output_dir / "predictions.npz", **{key: predictions_np[key] for key in save_keys})
    predictions_np["world_points_from_depth"] = unproject_depth(
        predictions_np["depth"], predictions_np["extrinsic"], predictions_np["intrinsic"]
    )
    scene = predictions_to_glb(
        predictions_np,
        conf_thres=args.confidence_percentile,
        show_cam=True,
        max_points=args.max_points,
    )
    scene.export(args.output_dir / "scene.glb")
    write_depth_video(predictions_np, args.output_dir / "rgb_depth.mp4")

    metadata = {
        **input_metadata,
        "checkpoint": str(args.checkpoint.resolve()),
        "video": str(args.video.resolve()) if args.video is not None else None,
        "preprocessed_shape": list(predictions_np["images"].shape),
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "frames_per_second": len(image_paths) / inference_seconds,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
