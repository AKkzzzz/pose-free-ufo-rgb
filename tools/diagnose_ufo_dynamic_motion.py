#!/usr/bin/env python3
"""Compare predicted and GT-box-oracle UFO dynamic assignment on one window."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from einops import rearrange


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import inference as ufo_inference
from ufo.models.archs.small import transform_gaussian_means_with_instances
from ufo.utils.config import merge_config_and_args
from ufo.utils.diagnostics import assignment_metrics


def save_vertical_comparison(top_path, bottom_path, output_path, fps=10):
    """Stack two render-only videos frame by frame without adding annotations."""
    top_reader = imageio.get_reader(top_path)
    bottom_reader = imageio.get_reader(bottom_path)
    frames = [
        np.concatenate([top, bottom], axis=0)
        for top, bottom in zip(top_reader, bottom_reader)
    ]
    top_reader.close()
    bottom_reader.close()
    imageio.mimsave(output_path, frames, fps=fps)


def build_args():
    parser = ufo_inference.get_args_parser()
    parser.description = "UFO dynamic assignment diagnostic"
    parser.add_argument(
        "--diagnostic_modes", nargs="+", choices=("predicted", "oracle_bbox"),
        default=("predicted", "oracle_bbox"),
    )
    args = merge_config_and_args(parser, config_path=None)
    args = merge_config_and_args(parser, config_path=args.config)
    args = ufo_inference.add_missing_config_values(args, args.config)
    args.full_window_targets = True
    return args


def per_target_diagnostics(input_dict, target_dict):
    gs_params = input_dict["gs_params"]
    means = rearrange(gs_params["means"], "b t v h w c -> b t (v h w) c").float()
    probabilities = rearrange(
        input_dict["bbox_weights"], "b t v h w k -> b t (v h w) k"
    ).float()
    context_pose = input_dict["context_instances_pose"].float()
    target_pose = input_dict["target_instances_pose"].float()
    context_valid = input_dict["context_instances_id"].bool()
    target_valid = input_dict["target_instances_id"].bool()
    batch, context_t, num_boxes = context_valid.shape
    target_t = target_pose.shape[1]

    identity_context = torch.eye(4, device=means.device).reshape(1, 1, 1, 4, 4)
    identity_context = identity_context.expand(batch, context_t, 1, 4, 4)
    identity_target = torch.eye(4, device=means.device).reshape(1, 1, 1, 4, 4)
    identity_target = identity_target.expand(batch, target_t, 1, 4, 4)
    transformed = transform_gaussian_means_with_instances(
        torch.cat([identity_context, context_pose], dim=2),
        torch.cat([identity_target, target_pose], dim=2),
        means,
        probabilities,
        stable_delta=True,
    )
    displacement = (transformed - means[:, None]).norm(dim=-1)
    predicted_foreground = probabilities.argmax(dim=-1) > 0

    assigned_slot = probabilities.argmax(dim=-1) - 1
    safe_slot = assigned_slot.clamp_min(0)
    centers = input_dict["context_instances_corner"].float().mean(dim=-2)
    assigned_centers = torch.gather(
        centers[:, :, None].expand(-1, -1, means.shape[2], -1, -1),
        3,
        safe_slot[..., None, None].expand(-1, -1, -1, 1, 3),
    ).squeeze(3)
    half_diagonal = (
        input_dict["context_instances_corner"].float() - centers[..., None, :]
    ).norm(dim=-1).amax(dim=-1)
    assigned_half_diagonal = torch.gather(
        half_diagonal[:, :, None].expand(-1, -1, means.shape[2], -1),
        3,
        safe_slot[..., None],
    ).squeeze(3)
    assigned_center_distance = (means - assigned_centers).norm(dim=-1)

    pose_rotations = torch.cat([context_pose, target_pose], dim=1)[..., :3, :3]
    eye3 = torch.eye(3, device=means.device)
    pose_orthogonality_error = (
        pose_rotations.transpose(-2, -1) @ pose_rotations - eye3
    ).norm(dim=(-2, -1))

    frame_ids = target_dict["target_frame_idx"][0].reshape(target_t, -1)[:, 0]
    rows = []
    for target_index, frame_id in enumerate(frame_ids.tolist()):
        valid = context_valid[:, :, None] & target_valid[:, None, target_index:target_index + 1]
        valid = valid.expand(-1, -1, 1, -1).squeeze(2)
        translation = (
            target_pose[:, target_index:target_index + 1, :, :3, 3]
            - context_pose[:, :, :, :3, 3]
        ).norm(dim=-1)
        target_displacement = displacement[:, target_index]
        dynamic_displacement = target_displacement[predicted_foreground]
        rows.append({
            "target_index": target_index,
            "frame_id": int(frame_id),
            "bbox_pose_mean_translation": float(translation[valid].mean().item())
                if valid.any() else None,
            "bbox_pose_max_translation": float(translation[valid].max().item())
                if valid.any() else None,
            "object_background_probability": float(probabilities[..., 0].mean().item()),
            "object_predicted_dynamic_ratio": float(predicted_foreground.float().mean().item()),
            "bbox_motion_mean_displacement": float(target_displacement.mean().item()),
            "bbox_motion_dynamic_mean_displacement": float(dynamic_displacement.mean().item())
                if dynamic_displacement.numel() else 0.0,
            "bbox_motion_max_displacement": float(target_displacement.max().item()),
        })

    target_centers = target_pose[..., :3, 3]
    consecutive_valid = target_valid[:, 1:] & target_valid[:, :-1]
    consecutive_motion = (target_centers[:, 1:] - target_centers[:, :-1]).norm(dim=-1)
    max_flat_index = int(displacement.reshape(-1).argmax().item())
    max_location = np.unravel_index(max_flat_index, displacement.shape)
    _, max_target, max_context, max_gaussian = max_location
    return {
        "context_timesteps": context_t,
        "target_timesteps": target_t,
        "gaussians": int(means.shape[1] * means.shape[2]),
        "background_probability": float(probabilities[..., 0].mean().item()),
        "predicted_dynamic_ratio": float(predicted_foreground.float().mean().item()),
        "assigned_point_center_distance_max": float(
            assigned_center_distance[predicted_foreground].max().item()
        ) if predicted_foreground.any() else None,
        "assigned_box_half_diagonal_max": float(
            assigned_half_diagonal[predicted_foreground].max().item()
        ) if predicted_foreground.any() else None,
        "pose_rotation_orthogonality_error_max": float(
            pose_orthogonality_error.max().item()
        ),
        "max_motion_location": {
            "target_index": int(max_target),
            "context_index": int(max_context),
            "gaussian_index": int(max_gaussian),
            "assigned_slot": int(assigned_slot[0, max_context, max_gaussian].item()),
            "point_center_distance": float(
                assigned_center_distance[0, max_context, max_gaussian].item()
            ),
            "box_half_diagonal": float(
                assigned_half_diagonal[0, max_context, max_gaussian].item()
            ),
        },
        "target_pose_consecutive_translation_mean": float(
            consecutive_motion[consecutive_valid].mean().item()
        ) if consecutive_valid.any() else None,
        "target_pose_consecutive_translation_max": float(
            consecutive_motion[consecutive_valid].max().item()
        ) if consecutive_valid.any() else None,
        "per_target": rows,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    args = build_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = ufo_inference.build_model(args, device)
    dataset = ufo_inference.build_dataset(args)
    results = {}
    for mode in args.diagnostic_modes:
        args.inference_assignment_mode = mode
        pred, target, input_dict, data_dict = ufo_inference.run_inference(
            model, dataset, args, device
        )
        video_path = output_dir / f"D_{mode}_start_{args.start_idx:03d}_render_3cam.mp4"
        fps = data_dict[0]["fps"] if isinstance(data_dict, list) else data_dict.get("fps", 10)
        ufo_inference.save_video(pred, input_dict, str(video_path), fps, device)
        result = {
            "mode": mode,
            "video": str(video_path.resolve()),
            "render_metrics": ufo_inference.compute_metrics(pred, target, input_dict, device),
            "aggregate_assignment_metrics": assignment_metrics(input_dict),
            "motion": per_target_diagnostics(input_dict, target),
        }
        results[mode] = result
        (output_dir / f"D_{mode}_start_{args.start_idx:03d}_diagnostics.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        del pred, target, input_dict, data_dict
        torch.cuda.empty_cache()
    summary = {
        "scene_id": args.scene_id,
        "start_idx": args.start_idx,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "results": results,
    }
    if "predicted" in results and "oracle_bbox" in results:
        comparison_path = output_dir / (
            f"D0_predicted_top_D1_oracle_bottom_start_{args.start_idx:03d}_render_3cam.mp4"
        )
        save_vertical_comparison(
            results["predicted"]["video"],
            results["oracle_bbox"]["video"],
            comparison_path,
            fps=fps,
        )
        summary["comparison_video"] = str(comparison_path.resolve())
    (output_dir / "dynamic_assignment_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
