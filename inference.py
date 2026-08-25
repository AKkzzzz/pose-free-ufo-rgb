# Copyright (C) 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
UFO Inference Script
====================
Feed-forward reconstruction of driving scenes from a pre-trained checkpoint.

Usage:
    python inference.py \
        --config config.json \
        --checkpoint path/to/ckpt.pth \
        --scene_id 160 \
        --start_idx 0 \
        --output_dir ./output/inference

Outputs:
    - Rendered video (MP4) of the reconstructed scene
    - Quantitative metrics (PSNR, SSIM, depth RMSE) printed to stdout
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from skimage.metrics import structural_similarity as ssim

from ufo.dataset.constants import DATASET_DICT, MEAN, STD
from ufo.dataset.data_utils import prepare_inputs_and_targets, to_batch_tensor
from ufo.dataset.dataset import UFODataset, UFODatasetEval
from ufo.models import UFO_models
from ufo.utils.config import merge_config_and_args
from ufo.utils.misc import update_scene

logger = logging.getLogger("UFO")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser("UFO Inference", add_help=True)

    # Required
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--scene_id", type=int, required=True,
                        help="Scene index in the dataset")

    # Optional
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting frame index within the scene")
    parser.add_argument("--output_dir", type=str, default="./output/inference",
                        help="Directory to save outputs")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode: save filtering PCD point clouds")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    parser.add_argument("--annotation_file", type=str, default=None,
                        help="Explicit annotation list; defaults to the dataset validation list")
    parser.add_argument("--pose_override_dir", type=str, default=None,
                        help="Root containing <scene_name>/omega_pose_override.npz")
    parser.add_argument("--pose_override_mode", choices=("none", "context", "all"),
                        default="none")
    parser.add_argument("--intrinsics_override_dir", type=str, default=None,
                        help="Root containing Omega intrinsics in the pose override NPZ")
    parser.add_argument("--intrinsics_override_mode", choices=("none", "context", "all"),
                        default="none")
    parser.add_argument(
        "--inference_assignment_mode", choices=("predicted", "oracle_bbox"),
        default="predicted",
        help="Use predicted assignment or a hard GT-box oracle during eval",
    )

    # Model parameters (defaults from config.json, CLI overrides)
    parser.add_argument("--model", default="UFO-B/8", type=str)
    parser.add_argument("--num_context_timesteps", default=4, type=int)
    parser.add_argument("--num_target_timesteps", default=4, type=int)
    parser.add_argument("--gs_dim", default=3, type=int)
    parser.add_argument("--use_sky_token", action="store_true")
    parser.add_argument("--use_affine_token", action="store_true")
    parser.add_argument("--use_latest_gsplat", action="store_true")
    parser.add_argument("--decoder_type", type=str, default="dummy",
                        choices=["dummy", "conv"])
    parser.add_argument("--num_motion_tokens", default=16, type=int)
    parser.add_argument("--disable_grad_checkpointing", action="store_true")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--num_mem_tokens", default=0, type=int)
    parser.add_argument("--filter_num", default=3600, type=int)

    # Dataset parameters
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--input_size", default=(160, 240), type=int, nargs=2)
    parser.add_argument("--num_max_cameras", type=int, default=3)
    parser.add_argument("--timespan", type=float, default=2.0)
    parser.add_argument("--dataset", default="waymo", type=str,
                        choices=list(DATASET_DICT.keys()))
    parser.add_argument("--load_depth", action="store_true")
    parser.add_argument("--load_flow", action="store_true")
    parser.add_argument("--load_ground", action="store_true")
    parser.add_argument("--skip_sky_mask", action="store_true")
    parser.add_argument("--num_target_chunks", default=1, type=int)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--num_bbox", default=32, type=int)
    parser.add_argument("--num_window_chunks", default=3, type=int)
    parser.add_argument("--recurrent", action="store_true")
    parser.add_argument("--ar", action="store_true")

    return parser


def add_missing_config_values(args, config_path):
    """Retain config-only model and dataset fields absent from this CLI parser."""
    config = json.loads(Path(config_path).read_text())
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def add_missing_checkpoint_values(args, checkpoint):
    """Fill parser gaps from the exact Namespace persisted during training."""
    checkpoint_args = checkpoint.get("args")
    if checkpoint_args is None:
        return args
    values = vars(checkpoint_args) if hasattr(checkpoint_args, "__dict__") else checkpoint_args
    for key, value in values.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(args, device):
    """Create and load a UFO model from checkpoint."""
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    add_missing_checkpoint_values(args, checkpoint)
    model = UFO_models[args.model](
        img_size=args.input_size,
        gs_dim=args.gs_dim,
        decoder_type=args.decoder_type,
        grad_checkpointing=not args.disable_grad_checkpointing,
        use_sky_token=args.use_sky_token,
        use_affine_token=args.use_affine_token,
        num_motion_tokens=args.num_motion_tokens,
        use_latest_gsplat=args.use_latest_gsplat,
        static=args.static,
        num_mem_tokens=args.num_mem_tokens,
        args=args,
    )

    state_dict = checkpoint["model"]
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        logger.warning(f"Missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        logger.warning(f"Unexpected keys: {msg.unexpected_keys}")

    model = model.to(device).eval()
    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(args):
    """Create evaluation dataset."""
    dataset_meta = DATASET_DICT[args.dataset]
    val_annotation = args.annotation_file or dataset_meta["annotation_txt_file_val"]
    if val_annotation is not None and not args.annotation_file:
        val_annotation = f"{args.data_root}/{val_annotation}"
        if not os.path.exists(val_annotation):
            raise FileNotFoundError(f"Annotation file not found: {val_annotation}")

    dataset = UFODataset(
        data_root=args.data_root,
        annotation_txt_file_list=val_annotation,
        subset_indices=[args.scene_id] if args.annotation_file else None,
        target_size=args.input_size,
        equispaced=True,
        num_context_timesteps=args.num_context_timesteps,
        num_target_timesteps=args.num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=args.load_depth,
        load_flow=args.load_flow,
        load_dynamic_mask=getattr(args, "load_dynamic_mask", False),
        load_ground_label=getattr(args, "load_ground", False),
        skip_sky_mask=args.skip_sky_mask,
        num_target_chunks=args.num_target_chunks,
        static=args.static,
        reverse=args.reverse,
        args=args,
    )
    logger.info(f"Dataset: {args.dataset}, {len(dataset.annotations)} scenes")
    return dataset


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def concatenate_chunk_targets(inout_dicts):
    """Combine every chunk's render inputs and GT along the target-time axis."""
    if not inout_dicts:
        raise ValueError("cannot concatenate targets from an empty chunk list")
    render_input = inout_dicts[-1][0].copy()
    target_keys = [
        key for key in render_input
        if key.startswith("target_") and isinstance(render_input[key], torch.Tensor)
    ]
    for key in target_keys:
        values = [input_dict[key] for input_dict, _ in inout_dicts]
        render_input[key] = torch.cat(values, dim=1)

    target_dict = {}
    for key in inout_dicts[0][1]:
        values = [chunk_target[key] for _, chunk_target in inout_dicts]
        target_dict[key] = torch.cat(values, dim=1)
    return render_input, target_dict

@torch.no_grad()
def run_inference(model, dataset, args, device):
    """Run autoregressive inference on a single scene.

    Returns:
        pred_dict: model predictions (rendered images, depth, etc.)
        target_dict: ground-truth data for the last chunk
        input_dict: input data for the last chunk (with accumulated scene)
        data_dict: raw data dict (contains fps, scene_name, etc.)
    """
    dataset_index = 0 if args.annotation_file else args.scene_id
    data_dict = dataset.__getitem__(dataset_index, args.start_idx, return_all=True)
    data_dict = to_batch_tensor(data_dict)

    inout_dicts = prepare_inputs_and_targets(
        data_dict, device, timespan=args.timespan, from_list=True, args=args
    )
    if getattr(args, "inference_assignment_mode", "predicted") == "oracle_bbox":
        full_target_valid = torch.cat(
            [input_dict["target_instances_id"] for input_dict, _ in inout_dicts],
            dim=1,
        ).bool().all(dim=1)
        for input_dict, _ in inout_dicts:
            input_dict["oracle_target_valid_throughout"] = full_target_valid
    num_chunks = len(inout_dicts)
    logger.info(f"Scene {args.scene_id} (start_idx={args.start_idx}): {num_chunks} chunks")

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        scene = {}
        for i in range(num_chunks):
            input_dict, target_dict = inout_dicts[i]

            t0 = time.perf_counter()
            log_dir = ""
            if args.debug:
                log_dir = os.path.join(args.output_dir, f"scene_{args.scene_id:05d}")
                os.makedirs(os.path.dirname(log_dir) or ".", exist_ok=True)
            _, scene = update_scene(
                input_dict, model, scene=scene,
                render=False, filter_num=args.filter_num,
                log_dir=log_dir,
            )
            elapsed = time.perf_counter() - t0
            n_tokens = scene["gs_state"].shape[1] if "gs_state" in scene else 0
            logger.info(f"  Chunk {i + 1}/{num_chunks}: {n_tokens} tokens ({elapsed:.3f}s)")

        # Render all target frames against the final accumulated scene.
        input_dict, target_dict = concatenate_chunk_targets(inout_dicts)
        logger.info(
            "Rendering final scene at %d target timesteps...",
            target_dict["target_image"].shape[1],
        )
        input_dict.update(scene)
        input_dict = model(input_dict, stage=2, motion=False)
        pred_dict = model(input_dict, stage=3)

    return pred_dict, target_dict, input_dict, data_dict


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_dict, target_dict, input_dict, device):
    """Compute evaluation metrics (PSNR, SSIM, depth RMSE)."""
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)

    def denormalize(x, channel_last=False):
        if not channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        x = (x * std + mean).clamp(0.0, 1.0)
        return rearrange(x, "t v h w c -> t v c h w")

    # Extract predictions and ground truth
    render_results = pred_dict["render_results"]
    target_images = target_dict["target_image"][0]  # [T, V, C, H, W]
    pred_images = render_results[render_results["rgb_key"]][0]  # [T, V, H, W, C]

    gt_rgb = denormalize(target_images)                    # [T, V, C, H, W]
    pr_rgb = denormalize(pred_images, channel_last=True)  # [T, V, C, H, W]
    # Both are now [T, V, C, H, W] after denormalize

    _, target_t, target_v, _, H, W = target_dict["target_image"].shape
    logger.info(f"[metrics debug] gt_rgb: {gt_rgb.shape}, pr_rgb: {pr_rgb.shape}, target_t={target_t}, target_v={target_v}")

    # Depth
    pred_depth = None
    if render_results.get("decoder_depth_key") is not None:
        pred_depth = render_results[render_results["decoder_depth_key"]][0]
    elif render_results.get("depth_key") is not None:
        pred_depth = render_results[render_results["depth_key"]][0]

    gt_depth = target_dict.get("target_depth", None)
    if gt_depth is not None:
        gt_depth = gt_depth[0]

    # Masks
    gt_sky = target_dict.get("target_sky_masks", None)
    if gt_sky is not None:
        occupied = (gt_sky[0] == 0)
    else:
        occupied = torch.ones((target_t, target_v, H, W), dtype=torch.bool, device=device)

    gt_dyn = target_dict.get("target_dynamic_masks", None)
    dynamic = gt_dyn[0].bool() if gt_dyn is not None else torch.zeros_like(occupied)

    valid_depth = gt_depth > 0.0 if gt_depth is not None else None

    # Flatten to per-frame, cast to float32 for metric precision
    gt_flat = rearrange(gt_rgb, "t v c h w -> (t v) h w c").float()
    pr_flat = rearrange(pr_rgb, "t v c h w -> (t v) h w c").float()
    occ_flat = rearrange(occupied, "t v h w -> (t v) h w")
    dyn_flat = rearrange(dynamic, "t v h w -> (t v) h w")

    metrics = {
        "psnr": [], "ssim": [],
        "occupied_psnr": [], "occupied_ssim": [],
        "dynamic_psnr": [], "dynamic_ssim": [],
        "depth_rmse": [], "dynamic_depth_rmse": [],
    }

    def _to_np(x):
        return x.detach().cpu().float().numpy()

    for idx in range(gt_flat.shape[0]):
        gt_np = _to_np(gt_flat[idx])
        pr_np = _to_np(pr_flat[idx])

        # PSNR
        mse = F.mse_loss(
            rearrange(pr_flat[idx], "h w c -> c h w"),
            rearrange(gt_flat[idx], "h w c -> c h w"),
        ).item()
        frame_psnr = -10.0 * np.log10(max(mse, 1e-12))
        if idx < 6:
            logger.info(f"[metrics debug] frame {idx}: mse={mse:.6f}, psnr={frame_psnr:.2f}")
        metrics["psnr"].append(frame_psnr)

        # SSIM
        ssim_val = ssim(pr_np, gt_np, data_range=1.0, channel_axis=-1)
        metrics["ssim"].append(float(ssim_val))

        ssim_map = None

        # Occupied PSNR/SSIM
        occ = occ_flat[idx]
        if occ.any():
            pr_occ = rearrange(pr_flat[idx], "h w c -> c h w")[:, occ]
            gt_occ = rearrange(gt_flat[idx], "h w c -> c h w")[:, occ]
            mse_occ = F.mse_loss(pr_occ, gt_occ).item()
            metrics["occupied_psnr"].append(-10.0 * np.log10(max(mse_occ, 1e-12)))
            ssim_map = ssim(
                pr_np, gt_np, data_range=1.0, channel_axis=-1, full=True
            )[1]
            occ_np = occ.cpu().numpy()
            metrics["occupied_ssim"].append(float(ssim_map[occ_np].mean()))

        # Dynamic PSNR/SSIM
        dm = dyn_flat[idx]
        if dm.any():
            pr_dyn = rearrange(pr_flat[idx], "h w c -> c h w")[:, dm]
            gt_dyn = rearrange(gt_flat[idx], "h w c -> c h w")[:, dm]
            mse_dyn = F.mse_loss(pr_dyn, gt_dyn).item()
            metrics["dynamic_psnr"].append(-10.0 * np.log10(max(mse_dyn, 1e-12)))
            if ssim_map is None:
                ssim_map = ssim(pr_np, gt_np, data_range=1.0, channel_axis=-1, full=True)[1]
            dyn_np = dm.cpu().numpy()
            metrics["dynamic_ssim"].append(float(ssim_map[dyn_np].mean()))

        # Depth RMSE
        if pred_depth is not None and gt_depth is not None:
            pr_d = rearrange(pred_depth, "t v h w -> (t v) h w")[idx]
            gt_d = rearrange(gt_depth, "t v h w -> (t v) h w")[idx]
            vm = rearrange(valid_depth, "t v h w -> (t v) h w")[idx]
            if vm.any():
                rmse = torch.sqrt(F.mse_loss(pr_d[vm], gt_d[vm])).item()
                metrics["depth_rmse"].append(rmse)
                vmd = vm & dm
                if vmd.any():
                    metrics["dynamic_depth_rmse"].append(
                        torch.sqrt(F.mse_loss(pr_d[vmd], gt_d[vmd])).item()
                    )

    # Average
    results = {}
    for key, vals in metrics.items():
        results[key] = float(np.nanmean(vals)) if vals else float("nan")
    return results


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

def save_video(pred_dict, input_dict, output_path, fps, device):
    """Save rendered predictions as an MP4 video."""
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)

    render_results = pred_dict["render_results"]
    pred_images = render_results[render_results["rgb_key"]][0]  # [T, V, H, W, C]
    pred_images = (pred_images * std + mean).clamp(0.0, 1.0)

    # Arrange as rows of views per timestep
    T, V, H, W, C = pred_images.shape
    frames = []
    for t in range(T):
        row = torch.cat([pred_images[t, v] for v in range(V)], dim=1)  # [H, V*W, C]
        frame = (row.cpu().float().numpy() * 255).astype(np.uint8)
        frames.append(frame)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    logger.info(f"Video saved to {output_path}")


def save_images(pred_dict, output_dir, device):
    """Save rendered predictions as individual PNG images (one file per frame per view)."""
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)

    render_results = pred_dict["render_results"]
    pred_images = render_results[render_results["rgb_key"]][0]  # [T, V, H, W, C]
    pred_images = (pred_images * std + mean).clamp(0.0, 1.0)

    T, V = pred_images.shape[:2]
    os.makedirs(output_dir, exist_ok=True)
    for t in range(T):
        for v in range(V):
            img = (pred_images[t, v].cpu().float().numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(output_dir, f"frame_{t:04d}_view_{v:02d}.png"), img)
    logger.info(f"Saved {T * V} predicted images to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = get_args_parser()
    args = merge_config_and_args(parser, config_path=None)

    # Re-parse with the config file specified
    config_path = args.config
    args = merge_config_and_args(parser, config_path=config_path)
    args = add_missing_config_values(args, config_path)

    device = torch.device(args.device)
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")

    # Build model and dataset
    model = build_model(args, device)
    dataset = build_dataset(args)

    # Run inference
    pred_dict, target_dict, input_dict, data_dict = run_inference(
        model, dataset, args, device
    )

    # Save video
    fps = data_dict[0]["fps"] if isinstance(data_dict, list) else data_dict.get("fps", 10)
    output_path = os.path.join(
        args.output_dir, f"scene_{args.scene_id:05d}_start_{args.start_idx:03d}.mp4"
    )
    save_video(pred_dict, input_dict, output_path, fps, device)

    # Save individual predicted images
    images_dir = os.path.join(
        args.output_dir, f"scene_{args.scene_id:05d}_start_{args.start_idx:03d}_pred"
    )
    save_images(pred_dict, images_dir, device)

    # Compute and print metrics
    metrics = compute_metrics(pred_dict, target_dict, input_dict, device)

    logger.info("=" * 50)
    logger.info("Metrics:")
    for key, val in metrics.items():
        if "psnr" in key:
            logger.info(f"  {key:25s}: {val:.2f} dB")
        elif "ssim" in key:
            logger.info(f"  {key:25s}: {val:.4f}")
        elif "rmse" in key:
            logger.info(f"  {key:25s}: {val:.3f} m")
        else:
            logger.info(f"  {key:25s}: {val:.4f}")
    logger.info("=" * 50)

    # Save metrics to JSON
    metrics_path = output_path.replace(".mp4", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
