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

import logging
import time
from ipdb import iex
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from ufo.utils.misc import compute_point_visibility, compute_visible_topk_indices_any_view, batched_index_gather
from ufo.utils.misc import convert_to_chunks, batched_index_update
from ufo.dataset.constants import MEAN, STD
from ufo.dataset.data_utils import (
    prepare_inputs_and_targets,
    prepare_inputs_and_targets_novel_view,
    to_batch_tensor,
)
from ufo.utils.misc import update_scene
from .annotation import add_label
from .layout import add_border, hcat, prep_image, vcat
from .visualization_tools import depth_visualizer, scene_flow_to_rgb
from ufo.utils.misc import combine_dict_entries, project_boxes_to_image
import os
import datetime
from skimage.metrics import structural_similarity as ssim
from ufo.utils.losses import compute_scene_flow_metrics

# Import depth evaluation functions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from reference_depth_eval import depth_evaluation

logger = logging.getLogger("UFO")


# ============================================================================
# SEGMENTATION VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_segmentation(segmentation_map, num_classes, colormap_path='segmentation_colormap.png'):
    """
    Visualize segmentation maps by mapping class indices to RGB colors.
    
    Args:
        segmentation_map: PyTorch tensor of shape [..., H, W] containing class indices
        num_classes: Total number of classes (int)
        colormap_path: Path to save the color-to-index correspondence image (str)
    
    Returns:
        RGB tensor of shape [..., 3, H, W] with values in range [0, 1]
    
    Example:
        >>> seg_map = torch.randint(0, 5, (2, 256, 256))  # Batch of 2 images
        >>> rgb_output = visualize_segmentation(seg_map, num_classes=5)
        >>> print(rgb_output.shape)  # torch.Size([2, 3, 256, 256])
    """
    device = segmentation_map.device
    dtype = torch.float32
    
    # Generate color palette
    colors = generate_color_palette(num_classes)  # Shape: (num_classes, 3)
    colors_tensor = torch.from_numpy(colors).to(device).to(dtype)
    
    # Save colormap visualization if it doesn't exist
    if not Path(colormap_path).exists():
        save_colormap_legend(colors, num_classes, colormap_path)
    
    # Get shape information
    *prefix, H, W = segmentation_map.shape
    
    # Flatten all prefix dimensions
    seg_flat = segmentation_map.reshape(-1, H, W).long()
    
    # Clamp indices to valid range
    seg_flat = torch.clamp(seg_flat, 0, num_classes - 1)
    
    # Use advanced indexing to map class indices to colors
    # colors_tensor[seg_flat] gives shape (batch, H, W, 3)
    rgb_flat = colors_tensor[seg_flat]  # Shape: (batch, H, W, 3)
    
    # Permute to get channels first: (batch, 3, H, W)
    rgb_flat = rgb_flat.permute(0, 3, 1, 2)
    
    # Reshape back to original prefix shape
    if prefix:
        rgb_output = rgb_flat.reshape(*prefix, 3, H, W)
    else:
        rgb_output = rgb_flat.squeeze(0)
    
    return rgb_output


def generate_color_palette(num_classes):
    """
    Generate distinct colors for each class using HSV color space.
    
    Args:
        num_classes: Number of classes
    
    Returns:
        numpy array of shape (num_classes, 3) with RGB values in [0, 1]
    """
    if num_classes == 1:
        return np.array([[1.0, 0.0, 0.0]])  # Red for single class
    
    # Generate evenly spaced hues
    hues = np.linspace(0, 1, num_classes, endpoint=False)
    
    # Create HSV colors with full saturation and value
    hsv_colors = np.stack([
        hues,
        np.ones(num_classes) * 0.8,  # Saturation
        np.ones(num_classes) * 0.95   # Value (brightness)
    ], axis=1)
    
    # Convert to RGB
    rgb_colors = hsv_to_rgb(hsv_colors)
    
    # Optional: Set class 0 (often background) to black
    # rgb_colors[0] = [0, 0, 0]
    
    return rgb_colors


def save_colormap_legend(colors, num_classes, save_path):
    """
    Save a legend showing the correspondence between colors and class indices.
    
    Args:
        colors: numpy array of shape (num_classes, 3) with RGB values
        num_classes: Number of classes
        save_path: Path to save the legend image
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, max(4, num_classes * 0.3)))
    
    # Create color patches
    for idx in range(num_classes):
        # Draw colored rectangle
        rect = plt.Rectangle((0, idx), 1, 0.8, facecolor=colors[idx])
        ax.add_patch(rect)
        
        # Add text label
        ax.text(1.1, idx + 0.4, f'Class {idx}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Set axis properties
    ax.set_xlim(-0.1, 3)
    ax.set_ylim(-0.5, num_classes)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.invert_yaxis()
    
    plt.title(f'Segmentation Color Map ({num_classes} classes)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Colormap legend saved to: {save_path}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _to_numpy_im(img_t):
    return img_t.detach().cpu().numpy()


def get_pca_map(x):
    x_shape = x.shape
    x = x.view(-1, x.shape[-1])
    x = x @ torch.pca_lowrank(x, q=3, niter=20)[2]
    x = (x - x.min(dim=0)[0]) / (x.max(dim=0)[0] - x.min(dim=0)[0])
    return x.view(*x_shape[:-1], 3)


# ============================================================================
# GAUSSIAN TRANSFORMATION FUNCTIONS
# ============================================================================

def transform_gaussians(transform_matrix, quaternions):
    """
    Apply a 4x4 transformation matrix (rotation + translation only) to 3D Gaussians.
    
    Args:
        means: (N, 3) - Gaussian centers
        quaternions: (N, 4) - Rotation as quaternions (w, x, y, z)
        transform_matrix: (4, 4) or (N, 4, 4) - Transformation matrix (rotation + translation)
        
    Returns:
        means_new: (N, 3) - Transformed centers
        quaternions_new: (N, 4) - Transformed rotations
    """
    N = quaternions.shape[0]
    
    # Extract rotation and translation from transformation matrix
    if transform_matrix.dim() == 2:
        R = transform_matrix[:3, :3]  # (3, 3)
        t = transform_matrix[:3, 3]    # (3,)
        R = R.unsqueeze(0).expand(N, -1, -1)  # (N, 3, 3)
        t = t.unsqueeze(0).expand(N, -1)      # (N, 3)
    else:
        R = transform_matrix[:, :3, :3]  # (N, 3, 3)
        t = transform_matrix[:, :3, 3]   # (N, 3)
    
    # 2. Transform rotations: compose the rotations
    # Convert transformation rotation to quaternion
    quat_transform = rotation_matrix_to_quaternion(R)  # (N, 4)
    
    # Compose quaternions: q_new = q_transform * q_original
    quaternions_new = quaternion_multiply(quat_transform, quaternions)
    
    
    return quaternions_new


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q1 * q2
    
    Args:
        q1: (N, 4) - (w, x, y, z)
        q2: (N, 4) - (w, x, y, z)
    Returns:
        q: (N, 4) - (w, x, y, z)
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return F.normalize(torch.stack([w, x, y, z], dim=-1), p=2, dim=-1)


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrices to quaternions (w, x, y, z).
    
    Args:
        R: (N, 3, 3)
    Returns:
        quaternions: (N, 4) - (w, x, y, z)
    """
    batch_size = R.shape[0]
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s = torch.sqrt(trace[mask1] + 1.0) * 2
    q[mask1, 0] = 0.25 * s
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s
    
    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
    q[mask2, 1] = 0.25 * s
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s
    
    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
    q[mask3, 2] = 0.25 * s
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
    
    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
    q[mask4, 3] = 0.25 * s
    
    return F.normalize(q, p=2, dim=-1)


# ============================================================================
# VIDEO GENERATION AND EVALUATION
# ============================================================================

@torch.no_grad()
@iex
def make_video(
    args,
    dataset,
    model,
    device,
    output_filename,
    scene_id=None,
    skip_plot_gt_depth_and_flow: bool = False,
    data_dict=None,
    input_dict=None,
    target_dict=None,
    pred_dict=None,
    eval_metrics=True,
    log_writer=None,
    start_idx=None
):
    
    
    output_prefix = output_filename.split('.mp4')[0]
    if data_dict is None:
        if scene_id is None:
            scene_id = np.random.randint(0, len(dataset))
        if start_idx is None:
            start_idx = np.random.randint(0, 70)
        data_dict = dataset.__getitem__(scene_id, start_idx, return_all=True)
        data_dict = to_batch_tensor(data_dict)


    inout_dicts = prepare_inputs_and_targets(data_dict, device, timespan=args.timespan, from_list=True, args=args)

    
    model = model.eval()
    render_all = False

    # ========================================================================
    # Model inference
    # ========================================================================

    if pred_dict is None:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            if True:
                # Initialize state variables for autoregressive processing
                pred_dict_list = []
                input_dict_list = []
                target_dict_list = []
                all_gs_features = {}

                # Autoregressive processing loop
                if args.reverse:
                    range_inout = range(len(inout_dicts) - 1, -1, -1)
                else:
                    range_inout = range(len(inout_dicts))

                for i in range_inout:
                    input_dict, target_dict = inout_dicts[i]

                    input_dict_list.append(input_dict)
                    target_dict_list.append(target_dict)

                    # scene update
                    pred_dict, all_gs_features = update_scene(input_dict, model, scene=all_gs_features, export_ply=True, profile=True, render=False, filter_num=args.filter_num, log_dir=output_prefix)
                    
                    if render_all:
                        pred_dict_list.append(pred_dict)

                # Final rendering of all Gaussian features
                input_dict.update(all_gs_features)
                input_dict = model(input_dict, stage=2, motion=False)

                pred_dict = model(input_dict, stage=3)

                # render full mode

                if render_all:
                    target_dict = combine_dict_entries(target_dict_list, target_dict_list[0].keys())
                    pred_dict = combine_dict_entries(pred_dict_list, [l for l in list(pred_dict_list[0].keys()) if 'loss' not in l])

            else:
                raise Exception("Not implemented")

    # ========================================================================
    # Prepare images and visualization
    # ========================================================================

    B, context_t, context_v, _, H, W = input_dict["context_image"].shape
    _, target_t, target_v, _, H_tgt, W_tgt = target_dict["target_image"].shape

    device = input_dict["context_image"].device
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)

    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        x = (x * std + mean).clamp(0.0, 1.0)
        return rearrange(x, "t v h w c -> t v c h w")

    # t, v, c, h, w
    context_images = input_dict["context_image"][0]
    context_images = denormalize(context_images)

    if context_v <= 3:
        n_ctx_per_row = 2
    else:
        n_ctx_per_row = 1
        # concate context images horizontally
    context_frames = []
    for t in range(context_t):
        current_frame_idx = int(input_dict["context_frame_idx"][0][t].item())
        row = add_label(
            hcat(*[context_images[t][v_id] for v_id in range(context_v)]),
            f"Context RGB (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        context_frames.append(row)
    num_rows = max(1, len(context_frames) // n_ctx_per_row)
    context_frames = vcat(
        *[
            hcat(
                *context_frames[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                gap=24,
            )
            for row in range(num_rows)
        ]
    )

    target_images = target_dict["target_image"][0]
    target_images = denormalize(target_images)
    render_results = pred_dict["render_results"]

    pred_images = render_results[render_results["rgb_key"]][0]
    pred_images = denormalize(pred_images, already_channel_last=True)
    if "rendered_motion_seg" in render_results:
        # Get the max index (clusters) from the rendered results
        max_idx = render_results["rendered_motion_seg"][0]
        visualization_result = visualize_segmentation(render_results["rendered_motion_seg"][0], 33, colormap_path='segmentation_colormap.png')
        cluster_image = visualization_result
    else:
        cluster_image = None

    # Add bounding boxes to target images
    for t in range(pred_dict['target_camtoworlds'].shape[1]):
        for v in range(pred_dict['target_camtoworlds'].shape[2]):
            target_images[t, v] = project_boxes_to_image(pred_dict['target_instances_corner_local'][0, t], pred_dict['target_camtoworlds'][0, t, v], pred_dict['target_intrinsics'][0, t, v], target_images[t, v], pred_dict['target_instances_id'][0, t])

    # ========================================================================
    # Generate video frames
    # ========================================================================

    video_frames = []
    for t in range(target_t):
        frame_list = []
        current_frame_idx = int(target_dict["target_frame_idx"][0][t].item())
        pred_rgb = add_label(
            hcat(*[pred_images[t][v_id] for v_id in range(target_v)]),
            f"Predicted RGB (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        frame_list.append(pred_rgb)

        gt_rgb = add_label(
            hcat(*[target_images[t][v_id] for v_id in range(target_v)]),
            f"Target GT RGB (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        frame_list.append(gt_rgb)

        render_full = True
        if render_full:
            if render_results["decoder_depth_key"] is not None:
                # this is a decoder depth map
                depth_image = render_results[render_results["decoder_depth_key"]][0][t]
                alpha_image = None
                depth_image = depth_image.detach().cpu().numpy()

                depth_image = depth_visualizer(depth_image, alpha_image)
                depth_image = torch.from_numpy(depth_image)
                depth_image = rearrange(depth_image, "v h w c -> v c h w")
                pred_depth = add_label(
                    hcat(*depth_image),
                    f"Predicted Decoder Depth (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(pred_depth)

            if render_results["depth_key"] is not None:
                # actual gs depth map
                depth_image = render_results[render_results["depth_key"]][0][t]
                alpha_image = render_results[render_results["alpha_key"]][0][t]
                if depth_image.shape[-2] != H_tgt or depth_image.shape[-1] != W_tgt:
                    depth_image = F.interpolate(
                        depth_image.unsqueeze(-3),
                        size=(H_tgt, W_tgt),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(-3)
                    alpha_image = F.interpolate(
                        alpha_image.unsqueeze(-3),
                        size=(H_tgt, W_tgt),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(-3)
                depth_image = depth_image.detach().cpu().numpy()
                alpha_image = alpha_image.detach().cpu().numpy()
                depth_image = depth_visualizer(depth_image, alpha_image)
                depth_image = torch.from_numpy(depth_image)
                depth_image = rearrange(depth_image, "v h w c -> v c h w")
                pred_depth = add_label(
                    hcat(*depth_image),
                    f"Predicted Depth (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(pred_depth)

            if "target_depth" in target_dict.keys():
                gt_depth = target_dict["target_depth"][0][t]
                gt_depth = gt_depth.detach().cpu().numpy()
                gt_depth = depth_visualizer(gt_depth, gt_depth > 0)
                gt_depth = torch.from_numpy(gt_depth)
                gt_depth = rearrange(gt_depth, "v h w c -> v c h w")
                gt_depth = add_label(
                    hcat(*gt_depth),
                    f"Target GT Depth (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(gt_depth)
            else:
                if not skip_plot_gt_depth_and_flow:
                    gt_depth = torch.full((target_v, 3, H_tgt, W_tgt), 0.5)
                    gt_depth = add_label(
                        hcat(*gt_depth),
                        f"Target GT Depth (t={current_frame_idx})",
                        font_size=24,
                        align="center",
                    )
                    frame_list.append(gt_depth)

            # Lifespan visualization
            if "rendered_lifespan" in render_results:
                lifespan_image = render_results["rendered_lifespan"][0][t]
                if lifespan_image.shape[-2] != H_tgt or lifespan_image.shape[-1] != W_tgt:
                    lifespan_image = F.interpolate(
                        lifespan_image.unsqueeze(-3),
                        size=(H_tgt, W_tgt),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(-3)
                lifespan_image = lifespan_image.detach().cpu().numpy()
                # Visualize lifespan using depth_visualizer with no alpha masking
                lifespan_image = depth_visualizer(lifespan_image, None)
                lifespan_image = torch.from_numpy(lifespan_image)
                lifespan_image = rearrange(lifespan_image, "v h w c -> v c h w")
                pred_lifespan = add_label(
                    hcat(*lifespan_image),
                    f"Predicted Lifespan (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(pred_lifespan)

            if render_results["depth_key"] is not None:
                alpha_image = torch.from_numpy(alpha_image).unsqueeze(1)
                alpha_image = alpha_image.repeat(1, 3, 1, 1)
                alpha_image = add_label(
                    hcat(*alpha_image),
                    f"Predicted Opacity (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(alpha_image)
            if "target_sky_masks" in target_dict.keys():
                sky_mask = target_dict["target_sky_masks"][0][t].unsqueeze(1)
                sky_mask = sky_mask.repeat(1, 3, 1, 1)
                sky_mask = add_label(
                    hcat(*sky_mask),
                    f"GT Sky Mask (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(sky_mask)
            if cluster_image is not None:
                cluster_image_t = cluster_image[t]
                # cluster_image_t = rearrange(cluster_image_t, "v h w c -> v c h w")
                cluster_image_t = add_label(
                    hcat(*cluster_image_t),
                    f"Motion Segmentation (t={current_frame_idx})",
                    font_size=24,
                    align="center",
                )
                frame_list.append(cluster_image_t)


        num_rows = len(frame_list) // n_ctx_per_row
        if render_full:
            frame = vcat(
                *[
                    hcat(
                        *frame_list[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                        gap=24,
                    )
                    for row in range(num_rows)
                ]
            )
        else:
            frame = vcat(
                    *[
                        hcat(
                            *frame_list[row * n_ctx_per_row : (row + 1) * n_ctx_per_row],
                            gap=24,
                        )
                        for row in range(num_rows)
                    ]
                )
        # if there's a residual, we add it to the end
        if len(frame_list) % n_ctx_per_row != 0:
            frame = vcat(
                frame,
                hcat(
                    *frame_list[num_rows * n_ctx_per_row :],
                    gap=24,
                ),
            )

        frame = add_border(
            add_label(
                frame,
                f"Scene{input_dict['scene_id']:03d}-{input_dict['scene_name'][:15]}",
                font_size=24,
                align="center",
            )
        )
        video_frames.append(prep_image(frame))

    imageio.mimsave(output_filename, video_frames, fps=data_dict[0]["fps"])

    # ========================================================================
    # Compute evaluation metrics
    # ========================================================================

    # Collect GT / Pred tensors with correct shapes
    gt_rgb_tvchw = target_images
    pred_rgb_tvchw = pred_images

    # Resize pred to target H/W if needed
    if (pred_rgb_tvchw.shape[-2] != H_tgt) or (pred_rgb_tvchw.shape[-1] != W_tgt):
        raise Exception("wrong size")

    # Depth: prefer decoder depth if present, else GS depth
    pred_depth_tvhw = None
    if render_results["decoder_depth_key"] is not None:
        pred_depth_tvhw = render_results[render_results["decoder_depth_key"]][0]
    elif render_results["depth_key"] is not None:
        pred_depth_tvhw = render_results[render_results["depth_key"]][0]
        if (pred_depth_tvhw.shape[-2] != H_tgt) or (pred_depth_tvhw.shape[-1] != W_tgt):
            raise Exception("wrong size")

    gt_depth_tvhw = target_dict.get("target_depth", None)
    if gt_depth_tvhw is not None:
        gt_depth_tvhw = gt_depth_tvhw[0]

    # Masks: occupied (non-sky), dynamic, valid depth
    gt_sky = target_dict.get("target_sky_masks", None)
    occupied_tvhw = None
    if gt_sky is not None:
        occupied_tvhw = (gt_sky[0] == 0)
    else:
        occupied_tvhw = torch.ones((target_t, target_v, H_tgt, W_tgt), dtype=torch.bool, device=device)

    gt_dyn = target_dict.get("target_dynamic_masks", None)
    if gt_dyn is not None:
        dynamic_tvhw = gt_dyn[0].bool()
    else:
        dynamic_tvhw = torch.ones_like(occupied_tvhw, dtype=torch.bool)

    valid_depth_tvhw = None
    if gt_depth_tvhw is not None:
        valid_depth_tvhw = gt_depth_tvhw > 0.0

    # Prepare RGB for SSIM/PSNR in HWC numpy
    gt_rgb_hwctv = rearrange(gt_rgb_tvchw, "t v c h w -> (t v) h w c")
    pr_rgb_hwctv = rearrange(pred_rgb_tvchw, "t v c h w -> (t v) h w c")

    # Prepare depth/masks if available
    if pred_depth_tvhw is not None:
        pr_d_hw = rearrange(pred_depth_tvhw, "t v h w -> (t v) h w")
    if gt_depth_tvhw is not None:
        gt_d_hw = rearrange(gt_depth_tvhw, "t v h w -> (t v) h w")
    occ_hw = rearrange(occupied_tvhw, "t v h w -> (t v) h w")
    dyn_hw = rearrange(dynamic_tvhw, "t v h w -> (t v) h w")
    if valid_depth_tvhw is not None:
        vld_hw = rearrange(valid_depth_tvhw, "t v h w -> (t v) h w")

    # Accumulate metrics
    tot_samples = 0
    tot_dyn_samples = 0
    tot_dyn_depth_samples = 0

    psnr_sum = 0.0
    ssim_sum = 0.0
    depth_rmse_sum = 0.0

    occ_psnr_sum = 0.0
    occ_ssim_sum = 0.0

    dyn_psnr_sum = 0.0
    dyn_ssim_sum = 0.0
    dyn_depth_rmse_sum = 0.0

    # Depth evaluation metrics from reference_depth_eval.py
    depth_abs_rel_sum = 0.0
    depth_sq_rel_sum = 0.0
    depth_log_rmse_sum = 0.0
    depth_delta1_sum = 0.0
    depth_delta2_sum = 0.0
    depth_delta3_sum = 0.0
    tot_depth_samples = 0

    for idx in range(pr_rgb_hwctv.shape[0]):
        gt_rgb_np = _to_numpy_im(gt_rgb_hwctv[idx])
        pr_rgb_np = _to_numpy_im(pr_rgb_hwctv[idx])

        # SSIM
        ssim_score = ssim(pr_rgb_np, gt_rgb_np, data_range=1.0, channel_axis=-1)
        ssim_sum += float(ssim_score)

        # Occupied SSIM using SSIM map
        ssim_map = ssim(pr_rgb_np, gt_rgb_np, data_range=1.0, channel_axis=-1, full=True)[1]
        occ_mask_np = occ_hw[idx].detach().cpu().numpy()
        if occ_mask_np.any():
            occ_ssim_sum += float(ssim_map[occ_mask_np].mean())
        else:
            occ_ssim_sum += float("nan")

        # PSNR (global)
        mse = F.mse_loss(
            rearrange(pr_rgb_hwctv[idx], "h w c -> c h w"),
            rearrange(gt_rgb_hwctv[idx], "h w c -> c h w"),
        ).item()
        psnr_sum += -10.0 * np.log10(max(mse, 1e-12))

        # Occupied PSNR
        occ_mask = occ_hw[idx]
        if occ_mask.any():
            pr_occ = rearrange(pr_rgb_hwctv[idx], "h w c -> c h w")[:, occ_mask]
            gt_occ = rearrange(gt_rgb_hwctv[idx], "h w c -> c h w")[:, occ_mask]
            mse_occ = F.mse_loss(pr_occ, gt_occ).item()
            occ_psnr_sum += -10.0 * np.log10(max(mse_occ, 1e-12))
        else:
            occ_psnr_sum += float("nan")

        # Depth RMSE (all valid)
        if (pred_depth_tvhw is not None) and (gt_depth_tvhw is not None):
            vm = vld_hw[idx]
            if vm.any():
                rmse = torch.sqrt(F.mse_loss(pr_d_hw[idx][vm], gt_d_hw[idx][vm])).item()
                depth_rmse_sum += rmse

                # Comprehensive depth metrics using depth_evaluation
                try:
                    depth_metrics, _, _, _ = depth_evaluation(
                        pr_d_hw[idx].unsqueeze(0),
                        gt_d_hw[idx].unsqueeze(0),
                        max_depth=80,
                        custom_mask=None,
                        align_with_lstsq=False,
                        use_gpu=torch.cuda.is_available()
                    )
                    depth_abs_rel_sum += depth_metrics["Abs Rel"]
                    depth_sq_rel_sum += depth_metrics["Sq Rel"]
                    depth_log_rmse_sum += depth_metrics["Log RMSE"]
                    depth_delta1_sum += depth_metrics["δ < 1.25"]
                    depth_delta2_sum += depth_metrics["δ < 1.25^2"]
                    depth_delta3_sum += depth_metrics["δ < 1.25^3"]
                    tot_depth_samples += 1
                except Exception as e:
                    logger.warning(f"Depth evaluation failed for sample {idx}: {e}")

        # Dynamic splits (if any dynamic pixel exists)
        dyn_mask_np = dyn_hw[idx].detach().cpu().numpy()
        if dyn_mask_np.any():
            # Dynamic SSIM
            dyn_ssim = float(ssim_map[dyn_mask_np].mean())
            dyn_ssim_sum += dyn_ssim

            # Dynamic PSNR
            dm = dyn_hw[idx]
            pr_dyn = rearrange(pr_rgb_hwctv[idx], "h w c -> c h w")[:, dm]
            gt_dyn_ = rearrange(gt_rgb_hwctv[idx], "h w c -> c h w")[:, dm]
            mse_dyn = F.mse_loss(pr_dyn, gt_dyn_).item()
            dyn_psnr_sum += -10.0 * np.log10(max(mse_dyn, 1e-12))
            tot_dyn_samples += 1

            # Dynamic Depth RMSE (also requires valid)
            if (pred_depth_tvhw is not None) and (gt_depth_tvhw is not None) and (valid_depth_tvhw is not None):
                vmd = (dyn_hw[idx] & vld_hw[idx])
                if vmd.any():
                    rmse_dyn = torch.sqrt(F.mse_loss(pr_d_hw[idx][vmd], gt_d_hw[idx][vmd])).item()
                    dyn_depth_rmse_sum += rmse_dyn
                    tot_dyn_depth_samples += 1

        tot_samples += 1

    def _safe_mean(x_sum, n):
        return (x_sum / max(n, 1)) if n > 0 else float("nan")

    val_metrics = {
        # RGB metrics
        "psnr": _safe_mean(psnr_sum, tot_samples),
        "ssim": _safe_mean(ssim_sum, tot_samples),
        "occupied_psnr": _safe_mean(occ_psnr_sum, tot_samples),
        "occupied_ssim": _safe_mean(occ_ssim_sum, tot_samples),
        "dynamic_psnr": _safe_mean(dyn_psnr_sum, tot_dyn_samples),
        "dynamic_ssim": _safe_mean(dyn_ssim_sum, tot_dyn_samples),

        # Basic depth metrics
        "depth_rmse": _safe_mean(depth_rmse_sum, tot_samples),
        "dynamic_depth_rmse": _safe_mean(dyn_depth_rmse_sum, tot_dyn_depth_samples),

        # Comprehensive depth metrics from reference_depth_eval.py
        "depth_abs_rel": _safe_mean(depth_abs_rel_sum, tot_depth_samples),
        "depth_sq_rel": _safe_mean(depth_sq_rel_sum, tot_depth_samples),
        "depth_log_rmse": _safe_mean(depth_log_rmse_sum, tot_depth_samples),
        "depth_delta_1.25": _safe_mean(depth_delta1_sum, tot_depth_samples),
        "depth_delta_1.25^2": _safe_mean(depth_delta2_sum, tot_depth_samples),
        "depth_delta_1.25^3": _safe_mean(depth_delta3_sum, tot_depth_samples),
    }

    return val_metrics
