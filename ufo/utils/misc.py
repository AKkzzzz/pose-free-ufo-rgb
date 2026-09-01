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

import collections.abc
import datetime
import logging
import math
import os
import random
from collections import OrderedDict
from glob import glob
from itertools import repeat

import numpy as np
import torch
from torch import inf

logger = logging.getLogger("UFO")


from typing import List, Dict, Any, Union
import torch
import torch.nn.functional as F
import torch
import open3d as o3d
import numpy as np
import time
from einops import rearrange
from ufo.paper_contract import relative_se3, transform_directions, transform_points

def save_point_cloud(points: torch.Tensor, save_path: str) -> None:
    """
    Save a PyTorch tensor as a point cloud file using Open3D.
    
    Args:
        points: PyTorch tensor of shape [..., 3] representing 3D points.
                Can be on GPU and/or in the computation graph.
        save_path: Path where the point cloud will be saved (e.g., 'output.ply', 'output.pcd')
    
    Example:
        >>> points = torch.randn(1000, 3, device='cuda', requires_grad=True)
        >>> save_point_cloud(points, 'my_pointcloud.ply')
    """
    import open3d as o3d
    # Detach from computation graph and move to CPU
    points_np = points.detach().cpu().numpy()
    
    # Reshape to (N, 3) if needed
    points_np = points_np.reshape(-1, 3)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    
    # Save the point cloud
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Point cloud saved to {save_path} with {len(pcd.points)} points")

def combine_dict_entries(dict_list: List[Dict[str, Any]], 
                         keys_to_combine: List[str]) -> Dict[str, Any]:
    """
    Combine entries from multiple dictionaries with support for various types:
    - Tensors: concatenate along dimension 1
    - Dictionaries: recursively combine
    - Strings: keep any single one (the first encountered)
    
    Args:
        dict_list: List of dictionaries containing various types
        keys_to_combine: List of keys specifying which entries to combine
    
    Returns:
        Dictionary with combined entries for specified keys
    
    Example:
        >>> dict1 = {
        ...     'tensor': torch.randn(2, 3),
        ...     'nested': {'a': torch.randn(2, 2), 'b': torch.randn(2, 1)},
        ...     'label': 'category_1'
        ... }
        >>> dict2 = {
        ...     'tensor': torch.randn(2, 5),
        ...     'nested': {'a': torch.randn(2, 3), 'b': torch.randn(2, 2)},
        ...     'label': 'category_1'
        ... }
        >>> result = combine_dict_entries([dict1, dict2], ['tensor', 'nested', 'label'])
        >>> result['tensor'].shape  # torch.Size([2, 8])
        >>> result['nested']['a'].shape  # torch.Size([2, 5])
        >>> result['label']  # 'category_1'
    """
    if not dict_list:
        raise ValueError("dict_list cannot be empty")
    
    if not keys_to_combine:
        raise ValueError("keys_to_combine cannot be empty")

    # Remove empty dicts
    dict_list = [d for d in dict_list if d]
    
    # Check if we have any non-empty dicts left
    if not dict_list:
        raise ValueError("All dictionaries in dict_list are empty")
    
    # Verify all specified keys exist in all dictionaries
    for key in keys_to_combine:
        for i, d in enumerate(dict_list):
            if key not in d:
                raise KeyError(f"Key '{key}' not found in dictionary at index {i}")
    
    # Create the result dictionary
    result = {}
    
    for key in keys_to_combine:
        # Get the first entry to determine the type
        first_entry = dict_list[0][key]
        
        if isinstance(first_entry, torch.Tensor):
            # Collect all tensors for this key
            tensors_to_concat = [d[key] for d in dict_list]
            
            # Verify all entries are tensors
            for i, tensor in enumerate(tensors_to_concat):
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Expected tensor for key '{key}' in dictionary at index {i}, "
                                  f"got {type(tensor).__name__}")
            
            # Concatenate along dimension 1
            result[key] = torch.cat(tensors_to_concat, dim=1)
            
        elif isinstance(first_entry, dict):
            # Recursively combine dictionaries
            # First, get all keys that appear in any of the nested dictionaries
            all_nested_keys = set()
            for d in dict_list:
                if not isinstance(d[key], dict):
                    raise TypeError(f"Expected dict for key '{key}', got {type(d[key]).__name__}")
                all_nested_keys.update(d[key].keys())
            
            # Recursively combine for each nested key
            nested_dicts = [d[key] for d in dict_list]
            result[key] = combine_dict_entries(nested_dicts, list(all_nested_keys))
            
        elif isinstance(first_entry, str):
            # For strings, keep the first non-empty one, or just the first if all are valid
            for d in dict_list:
                if not isinstance(d[key], str):
                    raise TypeError(f"Expected string for key '{key}', got {type(d[key]).__name__}")
            
            # Keep the first string (you can modify this logic if needed)
            result[key] = first_entry
            
        else:
            # For other types, you might want to handle them differently
            # For now, just keep the first one
            result[key] = first_entry
            print(f"Warning: Unhandled type {type(first_entry).__name__} for key '{key}'. "
                  f"Keeping first value.")
    
    return result

def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _ntuple(n):
    """
    Creates a parser that converts an input to a tuple of length n.

    Args:
        n (int): Length of the tuple.

    Returns:
        Callable: A function that parses the input into a tuple of length n.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def cleanup_checkpoints(ckpt_dir, keep_num=1):
    """
    Clean up old checkpoints, keeping only the latest 'keep_num' checkpoints.

    Args:
        ckpt_dir (str): Directory containing the checkpoints.
        keep_num (int): Number of recent checkpoints to keep.
    """
    ckpts = glob(f"{ckpt_dir}/*.pth")
    ckpts = [ckpt for ckpt in ckpts if "latest" not in ckpt and "best" not in ckpt]
    ckpts = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Remove older checkpoints
    for ckpt in ckpts[:-keep_num]:
        os.remove(ckpt)
        logger.info(f"Removed checkpoint: {ckpt}")

    # Create or update latest symlink
    if ckpts:
        latest_symlink = f"{ckpt_dir}/latest.pth"
        try:
            os.remove(latest_symlink)
        except FileNotFoundError:
            pass
        os.symlink(os.path.abspath(ckpts[-1]), latest_symlink)
        logger.info(f"Created symlink: {latest_symlink} -> {ckpts[-1]}")


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_model(args, model_without_ddp, optimizer=None, loss_scaler=None):
    """
    Load model, optimizer, and loss scaler states from a checkpoint.

    Args:
        args: Arguments containing checkpoint paths and loading configurations.
        model_without_ddp (torch.nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer for loading states.
        loss_scaler (torch.cuda.amp.GradScaler, optional): Loss scaler for AMP.

    Returns:
        int: Visualization slice ID if available.
    """
    vis_slice_id, checkpoint_loaded = 0, False
    if args.resume_from or args.auto_resume:
        if not args.resume_from:
            # Checkpoint not provided, auto-resume from the latest checkpoint
            checkpoints = [ckpt for ckpt in glob(f"{args.ckpt_dir}/*.pth") if "latest" not in ckpt]
            checkpoints = sorted(checkpoints, key=os.path.getmtime)
            if len(checkpoints) > 0:
                # Resume from the latest checkpoint
                args.resume_from = checkpoints[-1]

        if args.resume_from and os.path.exists(args.resume_from):
            logger.info(f"[Model-resume] Resuming from: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
            msg = model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
            logger.info(f"[Model-resume] Loaded model: {msg}")
            checkpoint_loaded = True
            if "optimizer" in checkpoint and "latest_step" in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info("[Model-resume] Loaded optimizer state")
                args.start_iteration = checkpoint["latest_step"] + 1
                if "loss_scaler" in checkpoint and loss_scaler is not None:
                    loss_scaler.load_state_dict(checkpoint["loss_scaler"])
                    logger.info("[Model-resume] Loaded loss scaler state")
                if "vis_slice_id" in checkpoint:
                    vis_slice_id = checkpoint["vis_slice_id"] + 1
            if "latest_step" in checkpoint:
                args.prev_num_iterations = checkpoint["latest_step"]
                args.start_iteration = checkpoint["latest_step"] + 1

            if "total_elapsed_time" in checkpoint:
                args.total_elapsed_time = float(checkpoint["total_elapsed_time"])
                elapsed_time_str = str(datetime.timedelta(seconds=int(args.total_elapsed_time)))
                logger.info(f"Loaded elapsed_time: {elapsed_time_str}")
            if "best_validation_psnr" in checkpoint:
                args.best_validation_psnr = float(checkpoint["best_validation_psnr"])
                logger.info("Restored best validation PSNR: %.4f", args.best_validation_psnr)
            rng_states = checkpoint.get("rng_states")
            if rng_states:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                restore_rng_state(rng_states[min(rank, len(rng_states) - 1)])
                logger.info("Restored RNG state for rank %d", rank)
            del checkpoint

    if not checkpoint_loaded and args.load_from and os.path.exists(args.load_from):
        # args.resume_from has the highest priority. If it's not found, try args.load_from
        # this is useful for loading a model without optimizer and scheduler states
        # or for loading a pre-trained model for initialization, fine-tuning, or evaluation.
        logger.info(f"Loading checkpoint from: {args.load_from}")
        checkpoint = torch.load(
            args.load_from,
            map_location="cpu",
            weights_only=False,
        )
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        try:
            msg = model_without_ddp.load_state_dict(checkpoint, strict=False)
            checkpoint_loaded = True
            logger.info(f"[Model-init] Loaded model: {msg}")
        except Exception as e:
            logger.error(e)
            logger.info(f"[Model-init] Loading model from {args.load_from} failed. Error: {e}")
            model_state_dict = model_without_ddp.state_dict()
            # Create a new OrderedDict that will only contain matching parameter shapes
            filtered_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_dict[k] = v
                    else:
                        logger.info(
                            f"Skipping parameter due to shape mismatch: {k} "
                            f"({v.shape} vs {model_state_dict[k].shape})"
                        )
                else:
                    logger.info(f"Skipping unexpected key: {k}")

            # Load the filtered state dict into the model (strict=False to allow missing keys)
            msg = model_without_ddp.load_state_dict(filtered_dict, strict=False)
            logger.info(f"Load status: {msg}")
        del checkpoint

    if not checkpoint_loaded:
        logger.info(f"Training from scratch. No checkpoint found.")
    return vis_slice_id


def adjust_learning_rate(optimizer, iteration, args):
    """
    Adjust the learning rate using a cosine decay schedule with warmup.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to update learning rate.
        iteration (int): Current training iteration.
        args: Arguments defining the learning rate schedule.

    Returns:
        float: Updated learning rate.
    """
    if iteration < args.warmup_iters:
        lr = args.lr * iteration / args.warmup_iters
    else:
        if args.lr_sched == "constant":
            lr = args.lr
        elif args.lr_sched == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (iteration - args.warmup_iters)
                    / (args.num_iterations - args.warmup_iters)
                )
            )
        else:
            raise ValueError(f"Unknown lr_sched: {args.lr_sched}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)

    return lr


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_grad_norm_(parameters, norm_type=2.0):
    """
    Compute gradient norm for a set of parameters.

    Args:
        parameters (Iterable): Parameters to compute gradients for.
        norm_type (float): Norm type for gradient computation.

    Returns:
        torch.Tensor: Gradient norm.
    """
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    return total_norm


class NativeScalerWithGradNormCount:
    """
    A wrapper for torch.cuda.amp.GradScaler with gradient norm tracking.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
    """

    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def __call__(
        self,
        loss,
        optimizer,
        parameters,
        clip_grad=None,
        create_graph=False,
        update_grad=True,
    ):
        self.backward(loss, create_graph=create_graph)
        norm = None
        if update_grad:
            norm = self.step(optimizer, parameters=parameters, clip_grad=clip_grad)
        return norm

    def backward(self, loss, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)

    def step(self, optimizer, parameters, clip_grad=None):
        self._scaler.unscale_(optimizer)
        if clip_grad is not None and clip_grad > 0.0:
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            norm = get_grad_norm_(parameters)
        self._scaler.step(optimizer)
        self._scaler.update()
        return norm

    def state_dict(self):
        """Save state dictionary for the scaler."""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dictionary for the scaler."""
        self._scaler.load_state_dict(state_dict)


def detach_tensors(value):
    """Detach tensor leaves in a recurrent scene container."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_tensors(item) for item in value)
    return value


def forward_ar(args, inout_dicts):
    if args.reverse:
        range_inout = range(len(inout_dicts) - 1, -1, -1)
    else:
        range_inout = range(len(inout_dicts))
    



import open3d as o3d
import numpy as np
from pathlib import Path


def create_camera_frustum(extrinsic, intrinsic, H, W, scale=0.5):
    """
    Create a camera frustum line set for visualization.
    
    Args:
        extrinsic: Camera-to-world transformation [4, 4]
        intrinsic: Camera intrinsic matrix [3, 3]
        H: Image height
        W: Image width
        scale: Scale factor for frustum size
    
    Returns:
        Open3D LineSet representing the camera frustum
    """
    # Define image corners in pixel coordinates
    corners_2d = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ], dtype=np.float32)
    
    # Convert to normalized image coordinates
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    corners_3d_cam = []
    for x, y in corners_2d:
        # Unproject to camera space at depth = scale
        X = (x - cx) / fx * scale
        Y = (y - cy) / fy * scale
        Z = scale
        corners_3d_cam.append([X, Y, Z])
    
    corners_3d_cam = np.array(corners_3d_cam)
    
    # Add camera center
    points = np.vstack([
        np.array([[0, 0, 0]]),  # Camera center
        corners_3d_cam           # Frustum corners
    ])
    
    # Transform to world coordinates
    points_homo = np.hstack([points, np.ones((5, 1))])
    points_world = (extrinsic @ points_homo.T).T[:, :3]
    
    # Define lines for frustum
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # Corners to corners
    ]
    
    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    return line_set

def save_debug_visualization(points, extrinsics, intrinsics, H, W, visibility, save_path,
                             frustum_scale=1.0, points_per_edge=50):
    """
    Save point cloud and camera frustums for debugging.
    Cameras are saved as colored point clouds for CloudCompare compatibility.

    Args:
        points: 3D points [N, 3] (torch tensor)
        extrinsics: Camera extrinsics [M, 4, 4] (torch tensor, world-to-camera)
        intrinsics: Camera intrinsics [M, 3, 3] (torch tensor)
        H: Image height
        W: Image width
        visibility: Visibility mask [M, N] (torch tensor)
        save_path: Path to save (e.g. 'debug/scene.ply')
        frustum_scale: Depth of frustum in world units
        points_per_edge: Number of points sampled per frustum edge
    """
    points_np = points.detach().cpu().numpy().reshape(-1, 3)
    extrinsics_np = extrinsics.detach().cpu().numpy()
    intrinsics_np = intrinsics.detach().cpu().numpy()
    visibility_np = visibility.detach().cpu().numpy()

    M = extrinsics_np.shape[0]

    # --- Scene points ---
    colors = np.zeros((points_np.shape[0], 3))
    visible_any = visibility_np.any(axis=0)
    colors[visible_any] = [0, 1, 0]
    colors[~visible_any] = [1, 0, 0]

    # --- Camera frustums as point clouds ---
    colors_palette = np.array([
        [1, 0, 0], [0, 0, 1], [1, 1, 0],
        [1, 0, 1], [0, 1, 1], [1, 0.5, 0],
    ])

    cam_points_list = []
    cam_colors_list = []

    for i in range(M):
        K = intrinsics_np[i]          # [3, 3]
        E = extrinsics_np[i]          # [4, 4]
        R, t = E[:3, :3], E[:3, 3]
        # camera center in world frame
        cam_center = -R.T @ t         # [3]

        # four image corners -> rays in camera frame -> world frame
        corners_px = np.array([
            [0, 0],
            [W, 0],
            [W, H],
            [0, H],
        ], dtype=np.float64)

        # unproject to camera coords at z=1
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        corners_cam = np.stack([
            (corners_px[:, 0] - cx) / fx,
            (corners_px[:, 1] - cy) / fy,
            np.ones(4),
        ], axis=-1)  # [4, 3]

        # to world frame at depth = frustum_scale
        corners_world = cam_center[None] + (R.T @ (corners_cam * frustum_scale).T).T  # [4, 3]

        # edges to sample: 4 edges from center to corners + 4 edges around the far plane
        edges = []
        for c in corners_world:
            edges.append((cam_center, c))
        for j in range(4):
            edges.append((corners_world[j], corners_world[(j + 1) % 4]))

        # sample points along each edge
        frustum_pts = [cam_center[None]]  # include the apex
        for start, end in edges:
            ts = np.linspace(0, 1, points_per_edge)[:, None]
            frustum_pts.append(start * (1 - ts) + end * ts)

        frustum_pts = np.concatenate(frustum_pts, axis=0)
        color = colors_palette[i % len(colors_palette)]
        frustum_colors = np.broadcast_to(color, frustum_pts.shape).copy()

        cam_points_list.append(frustum_pts)
        cam_colors_list.append(frustum_colors)

    # --- Save ---
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    base = save_path.rsplit('.', 1)[0]
    ext = '.' + save_path.rsplit('.', 1)[1]

    # scene points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    scene_path = f"{base}_points{ext}"
    o3d.io.write_point_cloud(scene_path, pcd)

    # camera frustums
    if cam_points_list:
        cam_all = np.concatenate(cam_points_list, axis=0)
        col_all = np.concatenate(cam_colors_list, axis=0)
        cam_pcd = o3d.geometry.PointCloud()
        cam_pcd.points = o3d.utility.Vector3dVector(cam_all.astype(np.float64))
        cam_pcd.colors = o3d.utility.Vector3dVector(col_all)
        cam_path = f"{base}_cameras{ext}"
        o3d.io.write_point_cloud(cam_path, cam_pcd)

    # print(f"Saved: {points_np.shape[0]} points ({visible_any.sum()} visible), {M} cameras")
    # print(f"  Points:  {scene_path}")
    # if cam_points_list:
    #     print(f"  Cameras: {cam_path}")


def compute_point_visibility(
    points_3d: torch.Tensor,      # [B, N, 3]
    cam_extrinsics: torch.Tensor, # [B, M, 4, 4] camera to world
    cam_intrinsics: torch.Tensor, # [B, M, 3, 3]
    H: int,
    W: int,
    debug: bool = False,
    debug_path: str = "debug_visibility.ply"
) -> torch.Tensor:                # [B, M, N] uint8
    """
    Compute visibility of 3D points in multiple camera views.
    
    Args:
        points_3d: 3D points in world coordinates [B, N, 3]
        cam_extrinsics: Camera-to-world transformation matrices [B, M, 4, 4]
        cam_intrinsics: Camera intrinsic matrices [B, M, 3, 3]
        H: Image height
        W: Image width
        debug: If True, save visualization to debug_path
        debug_path: Path to save debug visualization (default: "debug_visibility.ply")
    
    Returns:
        Visibility map [B, M, N] with 1 if point is visible, 0 otherwise
    """
    B, N, _ = points_3d.shape
    M = cam_extrinsics.shape[1]
    device = points_3d.device
    
    # Convert points to homogeneous coordinates [B, N, 4]
    points_homo = torch.cat([
        points_3d, 
        torch.ones(B, N, 1, device=device)
    ], dim=-1)
    
    # Invert extrinsics to get world-to-camera transformation [B, M, 4, 4]
    world_to_cam = torch.inverse(cam_extrinsics)
    
    # Transform points to camera coordinates [B, M, N, 4]
    # Using einsum for efficient batched matrix multiplication
    points_cam = torch.einsum('bmij,bnj->bmni', world_to_cam, points_homo)
    
    # Extract 3D camera coordinates [B, M, N, 3]
    points_cam_3d = points_cam[..., :3]
    
    # Project points using intrinsics [B, M, N, 3]
    points_proj = torch.einsum('bmij,bmnj->bmni', cam_intrinsics, points_cam_3d)
    
    # Normalize by depth to get pixel coordinates
    # points_proj: [x*z, y*z, z]
    depth = points_proj[..., 2:3]  # [B, M, N, 1]
    pixel_coords = points_proj[..., :2] / (depth + 1e-8)  # [B, M, N, 2]
    
    # Check visibility constraints
    # 1. Point is in front of camera (positive depth)
    in_front = depth[..., 0] > 0  # [B, M, N]
    
    # 2. Point is within image bounds
    x_coords = pixel_coords[..., 0]  # [B, M, N]
    y_coords = pixel_coords[..., 1]  # [B, M, N]
    
    in_x_bounds = (x_coords >= 0) & (x_coords < W)
    in_y_bounds = (y_coords >= 0) & (y_coords < H)
    
    # Combine all constraints
    visible = in_front & in_x_bounds & in_y_bounds
    
    # Debug visualization
    if debug:
        save_debug_visualization(
            points_3d[0],           # First batch only
            cam_extrinsics[0],      # First batch only
            cam_intrinsics[0],      # First batch only
            H, W,
            visible[0],             # First batch visibility
            debug_path
        )
    
    return visible.to(torch.uint8)


    import torch

@torch.no_grad()
def compute_visible_topk_indices_any_view(
    points_3d: torch.Tensor,      # [B, N, 3] world coords
    cam_extrinsics: torch.Tensor, # [B, M, 4, 4] camera-to-world
    cam_intrinsics: torch.Tensor, # [B, M, 3, 3]
    H: int,
    W: int,
    filter_num: int,              # output tensor size (padded with -1)
    debug_pcd_path: str = None,
    cell_size: int = 8,           # grid cell size in pixels (None = global top-k fallback)
    max_per_cell: int = 8,        # max tokens kept per grid cell
) -> torch.Tensor:                # [B, k] long, -1 where not enough visible
    """
    For each batch, among points visible in *any* camera, return indices of the top-k
    closest points by Euclidean distance to a camera center, as specified by
    UFO Eq. (2). Frustum visibility is still evaluated independently per view.

    Returns:
        Long tensor [B, k]; entries are point indices in [0, N-1], or -1 when
        there are fewer than k visible points in the batch.
    """
    B, N, _ = points_3d.shape
    # assert 0 < filter_num < N, "filter_num must satisfy 0 < filter_num < N"
    device = points_3d.device
    M = cam_extrinsics.shape[1]

    # Homogeneous points: [B, N, 4]
    points_homo = torch.cat([points_3d, torch.ones(B, N, 1, device=device)], dim=-1)

    # World->Camera: [B, M, 4, 4]
    world_to_cam = torch.inverse(cam_extrinsics)

    # Camera-space points: [B, M, N, 4] -> xyz & depth
    points_cam = torch.einsum('bmij,bnj->bmni', world_to_cam, points_homo)  # [B, M, N, 4]
    pts_cam_3d = points_cam[..., :3]                                        # [B, M, N, 3]
    zc = pts_cam_3d[..., 2]                                                 # [B, M, N]

    # Project with intrinsics to get pixel coords
    proj = torch.einsum('bmij,bmnj->bmni', cam_intrinsics, pts_cam_3d)      # [B, M, N, 3]
    denom = proj[..., 2].unsqueeze(-1)                                      # [B, M, N, 1]
    uv = proj[..., :2] / (denom + 1e-8)                                     # [B, M, N, 2]
    u, v = uv[..., 0], uv[..., 1]                                           # [B, M, N], [B, M, N]

    # Frustum visibility per camera
    in_front = zc > 0
    in_x = (u >= 0) & (u < W)
    in_y = (v >= 0) & (v < H)
    visible = in_front & in_x & in_y                                       # [B, M, N]

    camera_centers = cam_extrinsics[..., :3, 3]
    distances = torch.linalg.vector_norm(
        points_3d[:, None] - camera_centers[:, :, None], dim=-1
    )
    distance_masked = torch.where(
        visible, distances, torch.full_like(distances, float('inf'))
    )
    min_depth, best_cam = distance_masked.min(dim=1)

    # ---- Token selection strategy ----
    # Two modes:
    #   cell_size=None  -> Global top-k: keep the filter_num tokens with smallest depth.
    #                      Simple but clusters selections in nearby regions.
    #   cell_size=int   -> Grid-based diversity: divide the image plane into
    #                      (H//cell_size) x (W//cell_size) cells, keep at most
    #                      max_per_cell tokens per cell (ranked by depth).
    #                      Guarantees spatial coverage across the field of view.
    #
    # Both modes return [B, filter_num] with -1 padding for unfilled slots.

    if cell_size is None:
        # --- Global top-k by depth (original behavior) ---
        k = min(filter_num, N)
        top_vals, top_idx = torch.topk(min_depth, k=k, dim=-1, largest=False, sorted=True)
        invalid = ~torch.isfinite(top_vals)
        top_idx = top_idx.masked_fill(invalid, -1)
        # Pad to filter_num if scene has fewer tokens than requested
        if k < filter_num:
            padding = torch.full((B, filter_num - k), -1, device=device, dtype=torch.long)
            top_idx = torch.cat([top_idx, padding], dim=1)
    else:
        # --- Grid-based spatial diversity filtering ---
        #
        # Overview:
        #   1. For each token, find the camera where it is closest (best_cam).
        #   2. Project the token into that camera's image plane to get pixel (u, v).
        #   3. Quantize (u, v) into a grid cell.
        #   4. Within each cell, rank tokens by depth and keep the top max_per_cell.
        #
        # The sorting uses two stable sorts (depth then cell_id) so that after
        # grouping by cell, tokens within each cell are already depth-ordered.
        # Per-cell rank is computed via segment boundaries -- fully vectorized, no loops.

        # Step 1: Get pixel coords from the best (closest) camera for each token
        best_cam_exp = best_cam.unsqueeze(1)                                # [B, 1, N]
        u_best = torch.gather(u, 1, best_cam_exp).squeeze(1)               # [B, N]
        v_best = torch.gather(v, 1, best_cam_exp).squeeze(1)               # [B, N]

        # Step 2: Quantize pixel coords into grid cells
        # e.g., 160x240 image with cell_size=8 -> 20x30 = 600 cells
        grid_h = H // cell_size
        grid_w = W // cell_size
        cell_y = (v_best / cell_size).long().clamp(0, grid_h - 1)          # [B, N]
        cell_x = (u_best / cell_size).long().clamp(0, grid_w - 1)          # [B, N]
        cell_id = cell_y * grid_w + cell_x                                  # [B, N]

        # Step 3: Per-cell top-k selection (vectorized, assumes B=1)
        # Filter to visible tokens only (invisible tokens have depth = inf)
        is_visible = torch.isfinite(min_depth[0])                           # [N]
        vis_idx = is_visible.nonzero(as_tuple=True)[0]                      # [V]
        V = vis_idx.shape[0]

        if V == 0:
            top_idx = torch.full((1, filter_num), -1, device=device, dtype=torch.long)
        else:
            vis_depths = min_depth[0, vis_idx]                              # [V]
            vis_cells = cell_id[0, vis_idx]                                 # [V]

            # Sort by (cell_id, depth) using two stable sorts:
            #   1st sort by depth  -> tokens globally ordered by depth
            #   2nd sort by cell   -> groups by cell, depth order preserved within each cell
            _, depth_order = vis_depths.sort(stable=True)
            _, cell_order = vis_cells[depth_order].sort(stable=True)
            sort_order = depth_order[cell_order]

            sorted_orig_idx = vis_idx[sort_order]                           # original token indices, sorted
            sorted_cells = vis_cells[sort_order]                            # cell ids, sorted

            # Compute each token's rank within its cell (0-indexed):
            #   group_starts[i] = True where cell_id changes -> marks cell boundaries
            #   segment_id      = which cell group each token belongs to (0, 0, 0, 1, 1, ...)
            #   start_positions = index of the first token in each cell group
            #   rank_in_cell    = position_in_sorted - start_of_my_group
            group_starts = torch.ones(V, dtype=torch.bool, device=device)
            if V > 1:
                group_starts[1:] = sorted_cells[1:] != sorted_cells[:-1]
            start_positions = group_starts.nonzero(as_tuple=True)[0]        # [num_segments]
            segment_id = group_starts.long().cumsum(0) - 1                  # [V]
            rank_in_cell = torch.arange(V, device=device) - start_positions[segment_id]

            # Keep only the closest max_per_cell tokens per cell
            selected = sorted_orig_idx[rank_in_cell < max_per_cell]

            # Pad with -1 to filter_num, or truncate if exceeding budget
            num_selected = selected.shape[0]
            if num_selected < filter_num:
                padding = torch.full((filter_num - num_selected,), -1, device=device, dtype=torch.long)
                selected = torch.cat([selected, padding])
            elif num_selected > filter_num:
                selected = selected[:filter_num]

            top_idx = selected.unsqueeze(0)                                 # [1, filter_num]

    if debug_pcd_path is not None:
        # Build a [1, N] mask: True for tokens that were selected by filtering
        selected_mask = torch.zeros(1, N, dtype=torch.bool, device=device)  # [1, N]
        valid_idx = top_idx[0][top_idx[0] >= 0]
        selected_mask[0, valid_idx] = True
        save_debug_visualization(
            points_3d[0],
            world_to_cam[0],
            cam_intrinsics[0],
            H, W,
            selected_mask,          # green = selected, red = not selected
            debug_pcd_path
        )

    return top_idx.long()



def project_boxes_to_image(boxes_3d, camera_to_world, intrinsic, image, ids):
    """
    Project 3D bounding boxes onto an image and draw them.
    
    Args:
        boxes_3d: [N, 8, 3] tensor of 3D box corners in world space
        camera_to_world: [4, 4] camera-to-world transformation matrix
        intrinsic: [3, 3] camera intrinsic matrix
        image: [3, H, W] input image tensor
    
    Returns:
        image_with_boxes: [3, H, W] image with projected boxes drawn
    """
    device = image.device
    N = boxes_3d.shape[0]
    H, W = image.shape[1], image.shape[2]
    
    # Clone image to avoid modifying the original
    image_out = image.clone()
    
    # Convert to world-to-camera transformation
    world_to_camera = torch.inverse(camera_to_world)
    
    # Transform boxes from world space to camera space
    # Convert to homogeneous coordinates
    boxes_3d_homo = torch.cat([boxes_3d, torch.ones(N, 8, 1, device=device)], dim=-1)  # [N, 8, 4]
    
    # Transform all points: [N, 8, 4] @ [4, 4].T -> [N, 8, 4]
    boxes_camera = torch.matmul(boxes_3d_homo, world_to_camera.T)[:, :, :3]  # [N, 8, 3]
    
    # Project to image space using intrinsic matrix
    # Points in front of camera have positive Z
    boxes_2d_homo = torch.matmul(boxes_camera, intrinsic.T)  # [N, 8, 3]
    
    # Perspective division (x/z, y/z)
    boxes_2d = boxes_2d_homo[:, :, :2] / (boxes_2d_homo[:, :, 2:3] + 1e-8)  # [N, 8, 2]
    
    # Get depth (z values in camera space)
    depths = boxes_camera[:, :, 2]  # [N, 8]
    
    # Define the 12 edges of a box (connecting corner indices)
    edges = torch.tensor([
        [0, 1], [1, 3], [3, 2], [2, 0],  # bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ], device=device)
    
    # Draw each box
    for i in range(N):
        # invalid box
        if ids[i] == 0:
            continue
        box_2d = boxes_2d[i]  # [8, 2]
        box_depths = depths[i]  # [8]
        
        # Draw each edge
        for edge in edges:
            p1_idx, p2_idx = edge[0], edge[1]
            
            # Skip if either point is behind the camera
            if box_depths[p1_idx] <= 0 or box_depths[p2_idx] <= 0:
                continue
            
            p1 = box_2d[p1_idx]  # [2]
            p2 = box_2d[p2_idx]  # [2]
            
            # Draw line between p1 and p2
            image_out = draw_line(image_out, p1, p2, color=[1.0, 0.0, 0.0], thickness=2)
    
    return image_out

def draw_line(image, p1, p2, color=[1.0, 0.0, 0.0], thickness=2):
    """
    Draw a line on an image using GPU-accelerated linear interpolation.
    
    Args:
        image: [3, H, W] image tensor
        p1: [2] start point (x, y) - can be tensor
        p2: [2] end point (x, y) - can be tensor
        color: [3] RGB color (values in [0, 1])
        thickness: line thickness in pixels
    
    Returns:
        image: [3, H, W] image with line drawn
    """
    H, W = image.shape[1], image.shape[2]
    device = image.device
    
    # Keep everything on GPU - no .item() calls!
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    
    # Compute line length (number of pixels to sample)
    dx = (x2 - x1).abs()
    dy = (y2 - y1).abs()
    num_points = torch.maximum(dx, dy).ceil().long() + 1
    
    # Generate interpolation parameters [num_points]
    t = torch.linspace(0, 1, num_points.item(), device=device)
    
    # Interpolate along the line
    x = x1 + t * (x2 - x1)  # [num_points]
    y = y1 + t * (y2 - y1)  # [num_points]
    
    # Round to integer coordinates
    x = x.round().long()
    y = y.round().long()
    
    # Apply thickness using offsets
    if thickness > 1:
        offsets = torch.arange(-thickness//2, thickness//2 + 1, device=device)
        # Broadcast: [num_points, 1] + [thickness] = [num_points, thickness]
        x = x.unsqueeze(1) + offsets  # [num_points, thickness]
        y = y.unsqueeze(1) + offsets  # [num_points, thickness]
        # Flatten
        x = x.flatten()
        y = y.flatten()
    
    # Clamp to image boundaries
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x = x[valid]
    y = y[valid]
    
    # Convert color to tensor
    if not isinstance(color, torch.Tensor):
        color_tensor = torch.tensor(color, device=device, dtype=image.dtype)
    else:
        color_tensor = color
    
    # Draw all pixels at once (single GPU operation)
    image[:, y, x] = color_tensor.view(3, 1)
    
    return image


def convert_to_chunks(images, chunk_size=8):
    """
    Convert image tensor to chunks.
    
    Args:
        images: Tensor of shape [batch, dim1, dim2, height, width, channels]
        chunk_size: Size of each square chunk (default: 8)
    
    Returns:
        Tensor of shape [batch, dim1, dim2, n_chunks_h, n_chunks_w, chunk_size, chunk_size, channels]
    """
    batch, dim1, dim2, height, width, channels = images.shape
    
    # Calculate number of chunks
    n_chunks_h = height // chunk_size  # 160 // 8 = 20
    n_chunks_w = width // chunk_size   # 240 // 8 = 30
    
    # Reshape: split height and width into chunks
    # [1, 4, 3, 160, 240, 3] → [1, 4, 3, 20, 8, 30, 8, 3]
    chunks = images.reshape(batch, dim1, dim2, n_chunks_h, chunk_size, n_chunks_w, chunk_size, channels)
    
    # Permute to get desired order: [1, 4, 3, 20, 30, 8, 8, 3]
    chunks = chunks.permute(0, 1, 2, 3, 5, 4, 6, 7)
    
    return chunks


def batched_index_gather(values, indices, fill_value=0):
    B, N = values.shape[:2]
    rest = values.shape[2:]
    device = values.device

    if indices.dtype != torch.long: indices = indices.long()
    indices = indices.to(device)

    valid = (indices >= 0) & (indices < N)           # [B, k]
    idx_safe = indices.clamp(0, max(N-1, 0))         # replace invalid with 0

    idx_exp = idx_safe.view(B, -1, *([1]*len(rest))).expand(B, -1, *rest)
    out = torch.gather(values, dim=1, index=idx_exp) # [B, k, *rest]

    if valid.ndim == 2:
        valid_exp = valid.view(B, -1, *([1]*len(rest))).expand_as(out)
    else:
        valid_exp = valid
    # Zero (or fill_value) the positions that came from invalid indices
    if fill_value != 0:
        out = torch.where(valid_exp, out, torch.full_like(out, fill_value))
    else:
        out = torch.where(valid_exp, out, torch.zeros_like(out))
    return out


def batched_index_update(
    values: torch.Tensor,        # [B, N, ...]
    indices: torch.Tensor,       # [B, k] (long), may contain -1 to mean "no-op"
    update_values: torch.Tensor, # [B, k, ...]
    reduction: str = "replace",  # "replace" or "add"
) -> torch.Tensor:               # [B, N, ...]
    """
    Update `values` at per-batch row indices with `update_values`.

    - reduction="replace": out[b, idx[b,i], ...] = update_values[b,i,...]
      (duplicates: last-write wins, not deterministic across threads)
    - reduction="add":     out[b, idx[b,i], ...] += update_values[b,i,...]
      (duplicates: summed)

    Indices <0 or >=N are ignored.
    """
    assert values.dim() >= 2 and update_values.dim() >= 2
    B, N = values.shape[:2]
    Bi, k = indices.shape
    assert B == Bi, "Batch size mismatch"
    assert update_values.shape[:2] == (B, k), "update_values must be [B, k, ...]"
    assert indices.dtype == torch.long, "indices must be torch.long"
    assert values.device == indices.device == update_values.device

    rest = values.shape[2:]

    # mask invalid indices (e.g., -1 padding)
    valid = (indices >= 0) & (indices < N)
    if reduction == "replace":
        out = values.clone()
        if valid.any():
            # Flatten valid picks into a single advanced indexing op
            b_arange = torch.arange(B, device=values.device).unsqueeze(1).expand(B, k)
            b_idx = b_arange[valid]         # [Nv]
            n_idx = indices[valid]          # [Nv]
            upd   = update_values[valid]    # [Nv, *rest]
            out[b_idx, n_idx] = upd
        return out

    elif reduction in ("add", "sum"):
        out = values.clone()
        if valid.any():
            # scatter_add_ along dim=1 with broadcasted indices
            # invalid entries contribute zero
            idx_exp = indices.view(B, k, *([1]*len(rest))).expand(B, k, *rest)
            upd = update_values
            if valid.ndim == 2:
                valid_exp = valid.view(B, k, *([1]*len(rest))).expand_as(upd)
            else:
                valid_exp = valid
            upd = torch.where(valid_exp, upd, torch.zeros_like(upd))
            out.scatter_add_(dim=1, index=idx_exp, src=upd)
        return out

    else:
        raise ValueError(f"Unsupported reduction: {reduction}")



def update_scene(
    input_dict, model, scene=None, export_ply=False, profile=False, render=True,
    filter_num=3600, log_dir='', detach_old_scene=True, collect_diagnostics=True,
):
    """ Update the scene representation with new input frames

    Args:
        filter_num: Number of visible tokens to keep when filtering scene (k for top-k filtering)
        log_dir: Directory where PLY files will be exported when export_ply=True
    """

    scene = {} if scene is None else scene
    profile=False
    export_ply=False
    
    # Mapping between the current camera-centric frame and persistent scene world.
    # The relation is camera-independent; use the first synchronized camera.
    with torch.autocast(device_type=input_dict['context_image'].device.type, enabled=False):
        scene_from_local = relative_se3(
            input_dict['context_camtoworlds_global'][:, 0, 0].float(),
            input_dict['context_camtoworlds'][:, 0, 0].float(),
        )
        local_from_scene = torch.linalg.inv(scene_from_local)

    # metadata
    current_chunk_id = scene["_chunk_id"] + 1 if scene and "_chunk_id" in scene else 0
    previous_token_count = int(scene["gs_state"].shape[1]) if scene and "gs_state" in scene else 0
    visible_token_count = 0
    old_update_l2_mean = 0.0
    old_update_l2_max = 0.0



    # Temp workaround
    input_dict['prev_gs_time'] = None
    input_dict['prev_gs_origins'] = None
    input_dict['prev_gs_dirs'] = None
    input_dict['prev_gs_c2w_global'] = None
    input_dict['gs_state'] = None
    input_dict['mis_state'] = None



    model_args = getattr(model.module if hasattr(model, "module") else model, "args")
    if not model_args.disable_legacy_time_offset:
        # Official v1 forces every chunk's first context time to -1. This is
        # retained only for baseline compatibility; it destroys global dt.
        offset = abs(int(input_dict['context_time'].flatten()[0])) - 1
        input_dict['context_time'] += offset

    # 1. Filter scene scene tokens if scene is not empty

    if profile:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
    if scene and len(scene['gs_state']) > 0 and filter_num > 0:
        
        
        # Export Gaussians before updating
        if export_ply:
            temp_dict = input_dict.copy()
            temp_dict.update(scene)
            temp_dict = model(temp_dict, stage=2, motion=False)
            export_path = log_dir + "_chunk_" + str(current_chunk_id) + "_1_before.ply"
            model.forward_3_export(temp_dict, export_path)

        filtering = True
        b_context, t_context, v_context, _, H_context, W_context = input_dict['context_image'].shape
        if filtering:
            # find gs tokens that fall within each context frame
            b_context, t_context, v_context, _, H_context, W_context = input_dict['context_image'].shape
            context_extrinsics_flat = rearrange(input_dict['context_camtoworlds_global'], "b t v f1 f2 -> b (t v) f1 f2")
            context_intrinsics_flat = rearrange(input_dict['context_intrinsics'], "b t v f1 f2 -> b (t v) f1 f2")
            context_visibility_map = compute_visible_topk_indices_any_view(
                                                scene['gs_token_means'], # - toglobal_translation.reshape(1, 1, 3),
                                                context_extrinsics_flat,
                                                context_intrinsics_flat,
                                                H_context,
                                                W_context,
                                                filter_num=filter_num,
                                                debug_pcd_path=log_dir + "_chunk_" + str(current_chunk_id) + "_visibility.pcd" if log_dir else None,
                                                cell_size=None
                                            )
            
            # gather visible scene tokens
            old_state = scene['gs_state'].detach() if detach_old_scene else scene['gs_state']
            posterior_gs = batched_index_gather(old_state, context_visibility_map)

            posterior_gs_xyz = transform_points(
                local_from_scene[:, None], scene['gs_token_means']
            )
            posterior_gs_xyz = batched_index_gather(
                posterior_gs_xyz, context_visibility_map
            )

            posterior_gs_origins = transform_points(
                local_from_scene[:, None, None, None, None], scene['gs_origins']
            )
            posterior_gs_origins = convert_to_chunks(posterior_gs_origins).reshape(b_context, -1, 8, 8, 3)
            posterior_gs_origins = batched_index_gather(posterior_gs_origins, context_visibility_map)

            posterior_gs_dirs = transform_directions(
                local_from_scene[:, None, None, None, None], scene['gs_dirs']
            )
            posterior_gs_dirs = convert_to_chunks(posterior_gs_dirs).reshape(b_context, -1, 8, 8, 3)
            posterior_gs_dirs = batched_index_gather(posterior_gs_dirs, context_visibility_map)

            posterior_image = scene['image']
            posterior_image = convert_to_chunks(scene['image'].permute(0, 1, 2, 4, 5, 3)).reshape(b_context, -1, 8, 8, 3)
            posterior_image = batched_index_gather(posterior_image, context_visibility_map)


            posterior_gs_time = scene['gs_time']
            posterior_gs_time = posterior_gs_time.unsqueeze(-1).repeat(1, 1, 1, 600).reshape(b_context, -1)
            posterior_gs_time = batched_index_gather(posterior_gs_time, context_visibility_map)

            # Strip invalid tokens (where visibility map is -1) to prevent
            # zeroed inputs from producing non-zero features via bias/time_embed
            per_batch_valid = context_visibility_map != -1
            if not torch.equal(per_batch_valid, per_batch_valid[:1].expand_as(per_batch_valid)):
                raise ValueError("batched scenes must expose the same visible-token count")
            valid_mask = per_batch_valid[0]
            if not valid_mask.all():
                posterior_gs = posterior_gs[:, valid_mask]
                posterior_gs_origins = posterior_gs_origins[:, valid_mask]
                posterior_gs_dirs = posterior_gs_dirs[:, valid_mask]
                posterior_image = posterior_image[:, valid_mask]
                posterior_gs_time = posterior_gs_time[:, valid_mask]
                posterior_gs_xyz = posterior_gs_xyz[:, valid_mask]
                # Update visibility map for write-back in step 4
                context_visibility_map = context_visibility_map[:, valid_mask]

            visible_token_count = int(context_visibility_map.shape[1])

            input_dict['posterior_gs_dirs'] = posterior_gs_dirs
            input_dict['posterior_gs_origins'] = posterior_gs_origins
            input_dict['posterior_gs_time'] = posterior_gs_time
            input_dict['posterior_gs'] = posterior_gs
            input_dict['posterior_gs_xyz'] = posterior_gs_xyz
            input_dict['posterior_image'] = posterior_image
        else:

            local_origins = transform_points(
                local_from_scene[:, None, None, None, None], scene['gs_origins']
            )
            local_dirs = transform_directions(
                local_from_scene[:, None, None, None, None], scene['gs_dirs']
            )
            input_dict['posterior_gs_origins'] = convert_to_chunks(local_origins).reshape(b_context, -1, 8, 8, 3)
            input_dict['posterior_gs_dirs'] = convert_to_chunks(local_dirs).reshape(b_context, -1, 8, 8, 3)
            input_dict['posterior_gs_time'] = scene['gs_time'].unsqueeze(-1).repeat(1, 1, 1, 600).reshape(b_context, -1)
            input_dict['posterior_gs'] = (
                scene['gs_state'].detach() if detach_old_scene else scene['gs_state']
            )
            input_dict['posterior_gs_xyz'] = transform_points(
                local_from_scene[:, None], scene['gs_token_means']
            )

            input_dict['posterior_image'] = convert_to_chunks(scene['image'].permute(0, 1, 2, 4, 5, 3)).reshape(b_context, -1, 8, 8, 3)
            input_dict['posterior_c2w'] = scene['c2w'].clone()
            input_dict['posterior_time'] = scene['time']
            input_dict['posterior_c2w'] = local_from_scene[:, None, None] @ input_dict['posterior_c2w']
            input_dict['posterior_intr'] = scene['intr']

        if model_args.recurrent_aux_tokens:
            input_dict['posterior_aux_state'] = scene.get('aux_state')
    
    if profile:
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        logger.info(f"1. Scene filtering time: {end_time - start_time:.4f} seconds")

    # 2. model stage 1: global scene token update
    if profile:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
    input_dict = model(input_dict, stage=1)
    if profile:
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        logger.info(f"2. Model stage 1 (global scene token update) time: {end_time - start_time:.4f} seconds")

    # 3. model stage 2: token to gaussian
    if profile:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
    input_dict = model(input_dict, stage=2, motion=True)
    if profile:
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        logger.info(f"3. Model stage 2 (token to gaussian) time: {end_time - start_time:.4f} seconds")

    # 4. update existing tokens in scene
    if profile:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
    if scene and len(scene['gs_state']) > 0 and filter_num > 0:
        if collect_diagnostics:
            old_update_delta = (
                input_dict['updated_posterior'].detach().float()
                - input_dict['posterior_gs'].detach().float()
            ).norm(dim=-1)
            old_update_l2_mean = old_update_delta.mean().item()
            old_update_l2_max = old_update_delta.max().item()
        if filtering:
            scene['gs_state'] = batched_index_update(
                scene['gs_state'],
                context_visibility_map,
                input_dict['updated_posterior']
                )
        else:
            scene['gs_state'] = input_dict['updated_posterior']
        
    if profile:
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        logger.info(f"4. Update existing tokens time: {end_time - start_time:.4f} seconds")
        
        
    if export_ply:
        temp_dict = input_dict.copy()
        temp_dict.update(scene)
        temp_dict = model(temp_dict, stage=2, motion=False)
        export_path = log_dir + "_chunk_" + str(current_chunk_id) + "_2_after.ply"
        model.forward_3_export(temp_dict, export_path)

    # 5. Adding new scene tokens to scene
    if profile:
        torch.cuda.synchronize()
        start_time = time.perf_counter()

    # get location of each scene token TODO: current implementation is expensive
    with torch.no_grad():
        gs_means = rearrange(input_dict['gs_params']['means'], 'b t v h w c -> (b t v) c h w')

        # token means is the average of all gs means
        token_means = rearrange(F.avg_pool2d(gs_means, kernel_size=8, stride=8), '(b t v) c h w -> b (t v h w) c', b=input_dict['gs_params']['means'].shape[0], t=input_dict['gs_params']['means'].shape[1])
        token_means = transform_points(scene_from_local[:, None], token_means)

        assignment_anchor_means = input_dict.get('assignment_anchor_means')
        if assignment_anchor_means is not None:
            assignment_anchor_means = transform_points(
                scene_from_local[:, None], assignment_anchor_means
            )

        global_origins = transform_points(
            scene_from_local[:, None, None, None, None], input_dict['gs_origins']
        )
        global_dirs = transform_directions(
            scene_from_local[:, None, None, None, None], input_dict['gs_dirs']
        )

    # create new scene dict
    new_scene = {
        
        # scene token metadata (constant)
        "gs_origins": global_origins,
        "gs_dirs": global_dirs,
        "gs_time": input_dict['gs_time'],
        "sam_track_ids": input_dict.get("sam_track_ids"),
        
        # variable scene token state and position
        "gs_state": input_dict['gs_state'],
        "gs_token_means": token_means,  # location of each scene token in global frame
        "assignment_anchor_means": assignment_anchor_means,
        "assignment_anchor_valid": input_dict.get('assignment_anchor_valid'),
        "assignment_coverage_targets": input_dict.get('assignment_coverage_targets'),
        "assignment_coverage_valid": input_dict.get('assignment_coverage_valid'),
        
        
        
        # keep record of bbox metadata (constant)
        "bbox_weights": input_dict['bbox_weights'],
        "bbox_token_weights": input_dict['bbox_token_weights'],
        "bbox_token_logits": input_dict.get('bbox_token_logits'),
        "context_instances_corner": input_dict['context_instances_corner'],
        "context_instances_id": input_dict['context_instances_id'],
        "context_instances_pose": input_dict['context_instances_pose'],
        "context_instances_track_id": input_dict.get('context_instances_track_id'),


        # 0317 debug
        "image": input_dict['context_image'],
        "c2w": input_dict['context_camtoworlds_global'],
        "intr": input_dict['context_intrinsics'],
        "time": input_dict['context_time']
        
    }
    
    if export_ply:
        temp_dict = input_dict.copy()
        temp_dict.update(new_scene)
        temp_dict = model(temp_dict, stage=2, motion=False)
        model.forward_3_export(temp_dict, f"{log_dir}_chunk_" + str(current_chunk_id) + "_3_new.ply")
    new_scene = {key: value for key, value in new_scene.items() if value is not None}
    scene = combine_dict_entries([scene, new_scene], [k for k in new_scene.keys() if not k.startswith('_')])
    if model_args.recurrent_aux_tokens:
        scene['aux_state'] = input_dict['mis_state']
    scene["_chunk_id"] = current_chunk_id
    scene_diagnostics = {}
    if collect_diagnostics:
        xyz = scene["gs_token_means"].detach().float()
        scene_diagnostics = {
            "chunk": current_chunk_id,
            "scene_token_count": int(scene["gs_state"].shape[1]),
            "visible_token_count": visible_token_count,
            "visible_ratio": visible_token_count / max(previous_token_count, 1),
            "new_token_count": int(new_scene["gs_state"].shape[1]),
            "token_xyz_min_xyz": xyz.amin(dim=(0, 1)).cpu().tolist(),
            "token_xyz_max_xyz": xyz.amax(dim=(0, 1)).cpu().tolist(),
            "token_xyz_mean_norm": xyz.norm(dim=-1).mean().item(),
            "old_token_update_l2_mean": old_update_l2_mean,
            "old_token_update_l2_max": old_update_l2_max,
        }
    if profile:
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        logger.info(f"5. Adding new scene tokens time: {end_time - start_time:.4f} seconds")

    # 6. model stage 3: render all scene toke
    # ns
    if render:
        if profile:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        input_dict.update(scene)
        input_dict = model(input_dict, stage=2, motion=False)
        pred_dict = model(input_dict, stage=3)
        pred_dict["scene_diagnostics"] = scene_diagnostics
        if profile:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            logger.info(f"6. Model stage 3 (render) time: {end_time - start_time:.4f} seconds")

        return pred_dict, scene
    else:
        return input_dict, scene
