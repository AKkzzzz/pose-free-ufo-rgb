#!/usr/bin/env python3
"""Render a complete scene with independent UFO windows and stitch them in order."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import inference as ufo_inference
from ufo.dataset.constants import MEAN, STD
from ufo.dataset.pose_override import PoseOverrideStore
from ufo.utils.config import merge_config_and_args


LOGGER = logging.getLogger("UFO.long")


def build_args():
    parser = ufo_inference.get_args_parser()
    parser.description = "UFO full-scene sliding-window renderer"
    parser.add_argument(
        "--start_indices", type=int, nargs="+",
        default=[0, 20, 40, 60, 80, 100, 120, 140, 160, 178],
        help="Explicit 20-frame window starts in chronological order",
    )
    parser.add_argument(
        "--pose_override_sequence_dir", type=str, default=None,
        help="Root containing start_NNN/<scene_name>/omega_pose_override.npz",
    )
    parser.add_argument(
        "--intrinsics_override_sequence_dir", type=str, default=None,
        help="Root containing per-window Omega intrinsics override NPZ files",
    )
    parser.add_argument("--video_name", default="scene621_long_render_3cam.mp4")
    args = merge_config_and_args(parser, config_path=None)
    args = merge_config_and_args(parser, config_path=args.config)
    args = ufo_inference.add_missing_config_values(args, args.config)
    args.full_window_targets = True
    if args.pose_override_mode != "none":
        if args.pose_override_sequence_dir:
            args.pose_override_dir = str(
                Path(args.pose_override_sequence_dir)
                / f"start_{args.start_indices[0]:03d}"
            )
        elif not args.pose_override_dir:
            parser.error(
                "pose_override_dir or pose_override_sequence_dir is required"
            )

    if args.intrinsics_override_mode != "none":
        if args.intrinsics_override_sequence_dir:
            args.intrinsics_override_dir = str(
                Path(args.intrinsics_override_sequence_dir)
                / f"start_{args.start_indices[0]:03d}"
            )
        elif not args.intrinsics_override_dir:
            parser.error(
                "intrinsics_override_dir or "
                "intrinsics_override_sequence_dir is required"
            )
    return args


def target_frame_ids(target_dict):
    target_count = target_dict["target_image"].shape[1]
    raw = target_dict["target_frame_idx"][0].reshape(target_count, -1)
    if not torch.all(raw == raw[:, :1]):
        raise ValueError("camera views disagree on target frame ids")
    return raw[:, 0].detach().cpu().numpy().astype(int)


def select_timesteps(pred_dict, target_dict, keep):
    total = target_dict["target_image"].shape[1]
    keep_tensor = torch.as_tensor(keep, dtype=torch.long, device=target_dict["target_image"].device)
    selected_pred = dict(pred_dict)
    selected_render = dict(pred_dict["render_results"])
    for key, value in selected_render.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == total:
            selected_render[key] = value.index_select(1, keep_tensor)
    selected_pred["render_results"] = selected_render

    selected_target = {}
    for key, value in target_dict.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == total:
            selected_target[key] = value.index_select(1, keep_tensor)
        else:
            selected_target[key] = value
    return selected_pred, selected_target


def rendered_video_frames(pred_dict, device):
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)
    render = pred_dict["render_results"]
    images = (render[render["rgb_key"]][0] * std + mean).clamp(0.0, 1.0)
    for timestep in images:
        row = torch.cat([view for view in timestep], dim=1)
        yield (row.detach().cpu().float().numpy() * 255).astype(np.uint8)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.video_name
    device = torch.device(args.device)
    model = ufo_inference.build_model(args, device)
    dataset = ufo_inference.build_dataset(args)

    scene_frames = int(dataset.annotations[0]["num_timesteps"])
    fps = float(dataset.annotations[0].get("fps", 10))
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    last_frame_id = -1
    mapping = []
    windows = []
    weighted_metrics = {}
    total_metric_images = 0
    try:
        for start_index in args.start_indices:
            if (
                args.pose_override_mode != "none"
                and args.pose_override_sequence_dir
            ):
                root = (
                    Path(args.pose_override_sequence_dir)
                    / f"start_{start_index:03d}"
                )
                dataset.pose_override_store = PoseOverrideStore(root)

            if (
                args.intrinsics_override_mode != "none"
                and args.intrinsics_override_sequence_dir
            ):
                root = (
                    Path(args.intrinsics_override_sequence_dir)
                    / f"start_{start_index:03d}"
                )
                dataset.intrinsics_override_store = PoseOverrideStore(root)
            args.start_idx = start_index
            LOGGER.info("Rendering window start=%d", start_index)
            pred, target, input_dict, _ = ufo_inference.run_inference(
                model, dataset, args, device
            )
            frame_ids = target_frame_ids(target)
            keep = np.flatnonzero(frame_ids > last_frame_id)
            if not len(keep):
                continue
            selected_pred, selected_target = select_timesteps(pred, target, keep)
            selected_ids = frame_ids[keep]
            for frame_id, frame in zip(selected_ids, rendered_video_frames(selected_pred, device)):
                writer.append_data(frame)
                mapping.append({"video_frame": len(mapping), "source_frame": int(frame_id), "window_start": start_index})

            metrics = ufo_inference.compute_metrics(
                selected_pred, selected_target, input_dict, device
            )
            num_views = int(selected_target["target_image"].shape[2])
            metric_images = len(selected_ids) * num_views
            for key, value in metrics.items():
                if np.isfinite(value):
                    weighted_metrics[key] = weighted_metrics.get(key, 0.0) + value * metric_images
            total_metric_images += metric_images
            windows.append({
                "start_index": start_index,
                "first_frame": int(selected_ids[0]),
                "last_frame": int(selected_ids[-1]),
                "num_frames": len(selected_ids),
                "metrics": metrics,
            })
            last_frame_id = int(selected_ids[-1])
            del pred, target, selected_pred, selected_target, input_dict
            torch.cuda.empty_cache()
    finally:
        writer.close()

    summary = {
        "video": str(output_path.resolve()),
        "pose_override_mode": args.pose_override_mode,
        "pose_override_sequence_dir": args.pose_override_sequence_dir,
        "intrinsics_override_mode": args.intrinsics_override_mode,
        "intrinsics_override_sequence_dir": args.intrinsics_override_sequence_dir,
        "scene_frames": scene_frames,
        "rendered_frames": len(mapping),
        "fps": fps,
        "duration_seconds": len(mapping) / fps,
        "resolution": [args.input_size[1] * args.num_max_cameras, args.input_size[0]],
        "camera_layout": "front_left | front | front_right",
        "render_only": True,
        "window_state_policy": "reset_between_20_frame_windows",
        "metrics": {
            key: value / total_metric_images for key, value in weighted_metrics.items()
        },
        "windows": windows,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "frame_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    if len(mapping) != scene_frames or [item["source_frame"] for item in mapping] != list(range(scene_frames)):
        raise RuntimeError(
            f"expected continuous frames 0..{scene_frames - 1}, got {len(mapping)} frames"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
