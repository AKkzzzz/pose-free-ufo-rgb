#!/usr/bin/env python3
"""Export a VGGT-Omega predictions.npz as colored world-coordinate point clouds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the function actually called by the VGGT-Omega official demo. It
# treats extrinsics as OpenCV camera-from-world matrices and performs the inverse.
from demo_gradio import unproject_depth_map_to_point_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="Path to predictions.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: directory containing the NPZ)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="RGB directory for NPZ files without an images array",
    )
    parser.add_argument(
        "--image-pattern",
        default="*",
        help="Glob in --image-dir; supports {frame_id}, e.g. '{frame_id:03d}_0.jpg'",
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=50.0,
        help="Keep confidence at or above this percentile (official demo default: 50)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="World-coordinate voxel edge length for scene_filtered.ply",
    )
    parser.add_argument(
        "--glb-max-points",
        type=int,
        default=1_000_000,
        help="Maximum filtered points in scene.glb; PLY files are unaffected",
    )
    parser.add_argument("--no-glb", action="store_true", help="Do not export scene.glb")
    return parser.parse_args()


def print_inventory(path: Path, arrays: dict[str, np.ndarray]) -> None:
    print(f"NPZ: {path.resolve()}")
    print(f"keys ({len(arrays)}): {list(arrays)}")
    for key, value in arrays.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")


def array_by_alias(arrays: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in arrays:
            return arrays[name]
    raise KeyError(f"Missing required key; expected one of {names}")


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3:
        depth = depth[..., None]
    if depth.ndim != 4 or depth.shape[-1] != 1:
        raise ValueError(f"depth must have shape (S,H,W) or (S,H,W,1), got {depth.shape}")
    return depth


def rgb_from_npz(images: np.ndarray, expected_shape: tuple[int, int, int]) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"images must be four-dimensional, got {images.shape}")
    if images.shape[1] in (3, 4):
        images = np.transpose(images[:, :3], (0, 2, 3, 1))
    elif images.shape[-1] in (3, 4):
        images = images[..., :3]
    else:
        raise ValueError(f"images must use NCHW or NHWC RGB layout, got {images.shape}")
    if images.shape[:3] != expected_shape:
        raise ValueError(f"images shape {images.shape[:3]} does not match depth {expected_shape}")
    if np.issubdtype(images.dtype, np.floating):
        images = images * 255.0 if np.nanmax(images) <= 1.5 else images
    return np.clip(images, 0, 255).astype(np.uint8)


def image_paths(image_dir: Path, pattern: str, arrays: dict[str, np.ndarray], count: int) -> list[Path]:
    if "{frame_id" in pattern:
        if "frame_ids" not in arrays:
            raise ValueError("--image-pattern uses {frame_id}, but NPZ has no frame_ids key")
        paths = [image_dir / pattern.format(frame_id=int(i)) for i in arrays["frame_ids"]]
    else:
        paths = sorted(path for path in image_dir.glob(pattern) if path.is_file())
    if len(paths) != count:
        raise ValueError(f"Expected {count} RGB images, found {len(paths)} using {image_dir / pattern}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RGB images: {missing[:5]}")
    return paths


def rgb_from_files(paths: list[Path], height: int, width: int) -> np.ndarray:
    images = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read RGB image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        images.append(rgb)
    return np.stack(images)


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, confidence: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if voxel_size <= 0 or len(points) == 0:
        return points, colors, confidence
    voxel = np.floor(points / voxel_size).astype(np.int64)
    order = np.lexsort((-confidence, voxel[:, 2], voxel[:, 1], voxel[:, 0]))
    sorted_voxel = voxel[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(sorted_voxel[1:] != sorted_voxel[:-1], axis=1)
    keep = order[first]
    return points[keep], colors[keep], confidence[keep]


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a standard binary little-endian XYZRGB PLY readable by CloudCompare."""
    vertex = np.empty(
        len(points),
        dtype=np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ]
        ),
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Generated from VGGT-Omega official depth unprojection\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as output:
        output.write(header.encode("ascii"))
        vertex.tofile(output)


def write_glb(path: Path, points: np.ndarray, colors: np.ndarray, max_points: int) -> int:
    import trimesh

    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points, colors = points[indices], colors[indices]
    trimesh.points.PointCloud(points, colors=colors).export(path)
    return len(points)


def main() -> None:
    args = parse_args()
    if not 0 <= args.confidence_percentile <= 100:
        raise ValueError("--confidence-percentile must be in [0, 100]")
    if args.voxel_size < 0:
        raise ValueError("--voxel-size must be non-negative")

    with np.load(args.predictions, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    print_inventory(args.predictions, arrays)

    depth = normalize_depth(array_by_alias(arrays, "depth"))
    confidence = array_by_alias(arrays, "depth_conf")
    extrinsic = array_by_alias(arrays, "extrinsic", "extrinsics")
    intrinsic = array_by_alias(arrays, "intrinsic", "intrinsics")
    frame_shape = depth.shape[:3]
    if confidence.shape == (*frame_shape, 1):
        confidence = confidence[..., 0]
    if confidence.shape != frame_shape:
        raise ValueError(f"depth_conf shape {confidence.shape} does not match depth {frame_shape}")
    if extrinsic.shape != (frame_shape[0], 3, 4):
        raise ValueError(f"extrinsic shape must be {(frame_shape[0], 3, 4)}, got {extrinsic.shape}")
    if intrinsic.shape != (frame_shape[0], 3, 3):
        raise ValueError(f"intrinsic shape must be {(frame_shape[0], 3, 3)}, got {intrinsic.shape}")

    if "images" in arrays:
        colors = rgb_from_npz(arrays["images"], frame_shape)
        color_source = "NPZ key 'images'"
    elif args.image_dir:
        paths = image_paths(args.image_dir, args.image_pattern, arrays, frame_shape[0])
        colors = rgb_from_files(paths, frame_shape[1], frame_shape[2])
        color_source = str(args.image_dir.resolve())
    else:
        raise ValueError("NPZ has no 'images' key; provide --image-dir and optionally --image-pattern")

    # Do not infer or alter the pose convention here. This is the official
    # VGGT-Omega demo geometry function, called with its native argument order.
    world = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    depth_scalar = depth[..., 0]
    valid = (
        np.isfinite(world).all(axis=-1)
        & np.isfinite(depth_scalar)
        & (depth_scalar > 0)
        & np.isfinite(confidence)
    )
    points_full = world[valid].astype(np.float32)
    colors_full = colors[valid]
    confidence_full = confidence[valid].astype(np.float32)
    if not len(points_full):
        raise RuntimeError("No finite positive-depth points found")

    threshold = float(np.percentile(confidence_full, args.confidence_percentile))
    keep = confidence_full >= threshold
    points_filtered, colors_filtered, confidence_filtered = voxel_downsample(
        points_full[keep], colors_full[keep], confidence_full[keep], args.voxel_size
    )

    output_dir = args.output_dir or args.predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "scene_full.ply"
    filtered_path = output_dir / "scene_filtered.ply"
    write_ply(full_path, points_full, colors_full)
    write_ply(filtered_path, points_filtered, colors_filtered)

    glb_points = 0
    if not args.no_glb:
        glb_points = write_glb(output_dir / "scene.glb", points_filtered, colors_filtered, args.glb_max_points)

    metadata = {
        "predictions": str(args.predictions.resolve()),
        "geometry_function": "demo_gradio.unproject_depth_map_to_point_map",
        "color_source": color_source,
        "confidence_percentile": args.confidence_percentile,
        "confidence_threshold": threshold,
        "voxel_size": args.voxel_size,
        "valid_points_full": int(len(points_full)),
        "confidence_points_before_voxel": int(keep.sum()),
        "points_filtered": int(len(points_filtered)),
        "points_glb": int(glb_points),
    }
    (output_dir / "pointcloud_export.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
