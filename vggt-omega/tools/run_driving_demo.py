#!/usr/bin/env python3
"""VGGT-Omega geometry visualization for image-sequence driving clips.

This intentionally exposes only the released model outputs: camera poses,
intrinsics, depth and confidence. It does not perform novel-view rendering.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_gradio import load_model, run_model  # noqa: E402


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_dir", type=Path)
    source.add_argument("--input_root", type=Path)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/vggt_omega_1b_512.pt")
    p.add_argument("--output_dir", type=Path)
    p.add_argument("--output_root", type=Path)
    p.add_argument("--batch_scenes", action="store_true")
    p.add_argument("--image_resolution", type=int, default=512)
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--pattern", default="*", help="image filename glob inside --input_dir, e.g. '*_0.jpg' for Waymo front")
    p.add_argument("--confidence_threshold", type=float, default=1.2)
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--pixel_stride", type=int, default=1)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--max_video_points", type=int, default=30000)
    return p.parse_args()


def image_files(directory: Path, max_frames: int, pattern: str) -> list[Path]:
    files = sorted(p for p in directory.glob(pattern) if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise FileNotFoundError(f"No RGB images found in {directory}")
    return files[:max_frames]


def write_depth_outputs(depth: np.ndarray, conf: np.ndarray, out: Path, fps: float) -> tuple[float, float]:
    d = depth[..., 0]
    valid = np.isfinite(d) & (d > 0) & np.isfinite(conf)
    values = d[valid]
    if values.size == 0:
        raise RuntimeError("Depth contains no finite positive values")
    lo, hi = np.percentile(values, (2, 98))
    norm = Normalize(vmin=float(lo), vmax=float(hi), clip=True)
    cmap = plt.get_cmap("turbo")
    writer = imageio.get_writer(out / "depth.mp4", fps=fps, codec="libx264", quality=7)
    try:
        for i, frame in enumerate(d):
            rgb = (cmap(norm(frame))[..., :3] * 255).astype(np.uint8)
            rgb[~valid[i]] = 0
            imageio.imwrite(out / "depth" / f"{i:06d}.png", rgb)
            writer.append_data(rgb)
    finally:
        writer.close()
    return float(lo), float(hi)


def make_points(pred: dict, rgb_paths: list[Path], threshold: float, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    world = pred["world_points_from_depth"]
    depth = pred["depth"][..., 0]
    conf = pred["depth_conf"]
    points, colors, confidences, frame_ids = [], [], [], []
    for i, path in enumerate(rgb_paths):
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = depth[i].shape
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        sl = (slice(None, None, stride), slice(None, None, stride))
        xyz, rgb, c, dep = world[i][sl], image[sl], conf[i][sl], depth[i][sl]
        mask = np.isfinite(xyz).all(-1) & np.isfinite(dep) & (dep > 0) & np.isfinite(c) & (c >= threshold)
        points.append(xyz[mask].astype(np.float32))
        colors.append(rgb[mask].astype(np.uint8))
        confidences.append(c[mask].astype(np.float32))
        frame_ids.append(np.full(mask.sum(), i, dtype=np.int32))
    return tuple(np.concatenate(x) for x in (points, colors, confidences, frame_ids))


def voxel_filter(points: np.ndarray, colors: np.ndarray, conf: np.ndarray, frame: np.ndarray, size: float):
    if size <= 0 or len(points) == 0:
        return points, colors, conf, frame
    keys = np.floor(points / size).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep.sort()
    return points[keep], colors[keep], conf[keep], frame[keep]


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray, conf: np.ndarray, frame: np.ndarray) -> None:
    with path.open("wb") as f:
        header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(points)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  "property float confidence\nproperty int frame_id\nend_header\n")
        f.write(header.encode("ascii"))
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("confidence", "<f4"), ("frame_id", "<i4")])
        data = np.empty(len(points), dtype=dtype)
        data["x"], data["y"], data["z"] = points.T
        data["red"], data["green"], data["blue"] = colors.T
        data["confidence"], data["frame_id"] = conf, frame
        f.write(data.tobytes())


def trajectory_png(extrinsic: np.ndarray, out: Path) -> None:
    centers = -np.einsum("nij,nj->ni", extrinsic[:, :3, :3].transpose(0, 2, 1), extrinsic[:, :3, 3])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers[:, 0], centers[:, 2], "-o", ms=2, lw=1)
    ax.scatter(centers[0, 0], centers[0, 2], c="green", label="start")
    ax.scatter(centers[-1, 0], centers[-1, 2], c="red", label="end")
    ax.set_xlabel("world X"); ax.set_ylabel("world Z"); ax.set_title("VGGT-Omega camera trajectory")
    ax.axis("equal"); ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def scene_video(pred: dict, rgb_paths: list[Path], points: np.ndarray, out: Path, fps: float, max_points: int) -> None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), min(max_points, len(points)), replace=False)
    p = points[idx]
    extrinsic = pred["extrinsic"]
    centers = -np.einsum("nij,nj->ni", extrinsic[:, :3, :3].transpose(0, 2, 1), extrinsic[:, :3, 3])
    colors = pred["images"]
    if colors.ndim == 4: colors = np.transpose(colors, (0, 2, 3, 1))
    d = pred["depth"][..., 0]; valid = np.isfinite(d) & (d > 0)
    lo, hi = np.percentile(d[valid], (2, 98)); cmap = plt.get_cmap("turbo")
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    try:
        for i, path in enumerate(rgb_paths):
            fig = plt.figure(figsize=(16, 9), dpi=120)
            ax1 = fig.add_subplot(2, 2, 1); ax1.imshow(colors[i].clip(0, 1)); ax1.axis("off"); ax1.set_title("Input RGB")
            ax2 = fig.add_subplot(2, 2, 2); ax2.imshow(cmap(np.clip((d[i]-lo)/(hi-lo), 0, 1))); ax2.axis("off"); ax2.set_title("Predicted depth")
            ax3 = fig.add_subplot(2, 1, 2, projection="3d"); ax3.scatter(p[:,0],p[:,1],p[:,2],c="#4c78a8",s=0.25,alpha=.35)
            ax3.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="#e45756", linewidth=1.5)
            ax3.scatter(centers[i, 0], centers[i, 1], centers[i, 2], color="#54a24b", s=18)
            ax3.set_title(f"Fused world point cloud | frame {i}"); ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
            ax3.view_init(elev=18, azim=35 + i * 0.5)
            fig.tight_layout(); fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[...,:3]; writer.append_data(img); plt.close(fig)
    finally: writer.close()


def run_scene(input_dir: Path, output_dir: Path, model, cfg: argparse.Namespace) -> dict:
    paths = image_files(input_dir, cfg.max_frames, cfg.pattern); output_dir.mkdir(parents=True, exist_ok=True); (output_dir / "depth").mkdir(exist_ok=True)
    # run_model is the official preprocessing, forward, pose decoding and unprojection path.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pred = run_model(str(_materialize_input(paths, output_dir)), model, cfg.image_resolution)
    np.savez_compressed(output_dir / "predictions.npz", **{k:v for k,v in pred.items() if isinstance(v,np.ndarray)})
    d_range = write_depth_outputs(pred["depth"], pred["depth_conf"], output_dir, cfg.fps)
    pts, rgb, conf, frames = make_points(pred, paths, 0.0, cfg.pixel_stride)
    write_ply(output_dir / "scene_full.ply", pts, rgb, conf, frames)
    keep = conf >= cfg.confidence_threshold
    fpts, frgb, fconf, fframes = voxel_filter(pts[keep], rgb[keep], conf[keep], frames[keep], cfg.voxel_size)
    write_ply(output_dir / "scene_filtered.ply", fpts, frgb, fconf, fframes)
    trajectory_png(pred["extrinsic"], output_dir / "camera_trajectory.png")
    scene_video(pred, paths, fpts, output_dir / "scene_demo.mp4", cfg.fps, cfg.max_video_points)
    meta = {"num_frames": len(paths), "image_resolution": list(pred["images"].shape[-2:]), "checkpoint": str(cfg.checkpoint.resolve()), "confidence_threshold": cfg.confidence_threshold, "voxel_size": cfg.voxel_size, "input_camera": "front", "depth_range": d_range, "points_full": len(pts), "points_filtered": len(fpts), "gpu_peak_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None), "protocol": "official VGGT-Omega geometry inference; no novel-view renderer"}
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _materialize_input(paths: list[Path], output_dir: Path) -> Path:
    import shutil
    d = output_dir / "_official_input" / "images"; d.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths): shutil.copy2(p, d / f"{i:06d}{p.suffix.lower()}")
    return d.parent


def main() -> None:
    cfg = args(); cfg.output_dir = cfg.output_dir or (cfg.output_root if cfg.output_root and not cfg.batch_scenes else None)
    if cfg.batch_scenes:
        if not cfg.input_root or not cfg.output_root: raise ValueError("--batch_scenes requires --input_root and --output_root")
        model = load_model(str(cfg.checkpoint)); results = {}
        for scene in sorted(cfg.input_root.iterdir()):
            if scene.is_dir() and (scene / "images").is_dir(): results[scene.name] = run_scene(scene / "images", cfg.output_root / scene.name, model, cfg)
        print(json.dumps(results, indent=2)); return
    if not cfg.output_dir: raise ValueError("single scene requires --output_dir")
    model = load_model(str(cfg.checkpoint)); started = time.perf_counter(); result = run_scene(cfg.input_dir, cfg.output_dir, model, cfg); result["elapsed_seconds"] = time.perf_counter() - started; print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
