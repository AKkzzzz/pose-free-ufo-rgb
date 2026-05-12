# Copyright (c) Xiaomi Corporation.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from einops import rearrange, repeat
from gsplat.rendering import rasterization
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from ufo.dataset.constants import MEAN, STD
import time
from ..modules import (
    DummyDecoder,
    LayerNorm2d,
    ModulatedLinearLayer,
    PluckerEmbedder,
    TimestepEmbedder,
)
from ..vit import Mlp, VisionTransformer as ViT
import numpy as np
import torch
from plyfile import PlyData, PlyElement


from ufo.utils.misc import save_point_cloud
import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F



def ray_distance_loss(points, origins, directions):
    """
    Compute loss that regularizes points to lie on rays.
    
    The loss measures the squared perpendicular distance from each point 
    to its corresponding ray defined by origin + t * direction.
    
    Args:
        points: [B, 3] - 3D points to regularize
        origins: [B, 3] - ray origins
        directions: [B, 3] - ray directions (will be normalized internally)
    
    Returns:
        loss: scalar - mean squared distance from points to rays
    """
    # Vector from origin to point
    v = points - origins  # [B, 3]
    
    # Normalize directions to unit vectors
    directions_norm = F.normalize(directions, dim=-1, eps=1e-6)  # [B, 3]
    
    # Project v onto the direction vector
    # proj_length = v · d (dot product)
    proj_length = (v * directions_norm).sum(dim=-1, keepdim=True)  # [B, 1]
    
    # Projection vector onto the ray
    proj = proj_length * directions_norm  # [B, 3]
    
    # Perpendicular component (rejection) - this is the distance vector
    perp = v - proj  # [B, 3]
    
    # Squared perpendicular distance
    squared_dist = (perp ** 2).sum(dim=-1)  # [B]
    
    # Return mean loss
    loss = squared_dist.mean()
    
    return loss
def points_in_boxes_probability(points, boxes, valid_mask, temperature=0.1, background_bias=0.0):
    """
    Compute probability of points being in boxes, with background class.
    
    Args:
        points: [B, N, 3] - 3D points in world coordinates
        boxes: [B, M, 8, 3] - 3D bounding boxes as 8 corners
        temperature: float - softmax temperature (lower = sharper distribution)
        background_bias: float - bias for background class (higher = more likely to assign to background)
    
    Returns:
        probs: [B, N, M+1] - probability distribution over M boxes + 1 background
                            probs[:, :, 0] = background probability
                            probs[:, :, 1:] = probability for each box
    """
    B, N, _ = points.shape
    _, M, _, _ = boxes.shape
    device = points.device
    
    # Extract box properties from corners
    # Center: mean of all 8 corners [B, M, 3]
    centers = boxes.mean(dim=2)
    
    # Compute box axes from corners (average of parallel edges)
    # X-axis: direction from corners 0->1, 2->3, 4->5, 6->7
    x_axis = (boxes[:, :, 1] - boxes[:, :, 0] + 
              boxes[:, :, 3] - boxes[:, :, 2] +
              boxes[:, :, 5] - boxes[:, :, 4] +
              boxes[:, :, 7] - boxes[:, :, 6]) / 4.0  # [B, M, 3]
    
    # Y-axis: direction from corners 0->2, 1->3, 4->6, 5->7
    y_axis = (boxes[:, :, 2] - boxes[:, :, 0] +
              boxes[:, :, 3] - boxes[:, :, 1] +
              boxes[:, :, 6] - boxes[:, :, 4] +
              boxes[:, :, 7] - boxes[:, :, 5]) / 4.0  # [B, M, 3]
    
    # Z-axis: direction from corners 0->4, 1->5, 2->6, 3->7
    z_axis = (boxes[:, :, 4] - boxes[:, :, 0] +
              boxes[:, :, 5] - boxes[:, :, 1] +
              boxes[:, :, 6] - boxes[:, :, 2] +
              boxes[:, :, 7] - boxes[:, :, 3]) / 4.0  # [B, M, 3]
    
    # Compute extents (half-lengths along each axis)
    x_extent = torch.norm(x_axis, dim=2, keepdim=True) / 2.0  # [B, M, 1]
    y_extent = torch.norm(y_axis, dim=2, keepdim=True) / 2.0  # [B, M, 1]
    z_extent = torch.norm(z_axis, dim=2, keepdim=True) / 2.0  # [B, M, 1]
    
    # Normalize axes to unit vectors
    x_axis = x_axis / (2.0 * x_extent + 1e-8)  # [B, M, 3]
    y_axis = y_axis / (2.0 * y_extent + 1e-8)  # [B, M, 3]
    z_axis = z_axis / (2.0 * z_extent + 1e-8)  # [B, M, 3]
    
    # Compute relative positions: [B, N, M, 3]
    points_expanded = points.unsqueeze(2)      # [B, N, 1, 3]
    centers_expanded = centers.unsqueeze(1)     # [B, 1, M, 3]
    relative_pos = points_expanded - centers_expanded  # [B, N, M, 3]
    
    # Project points onto box local axes
    x_axis_expanded = x_axis.unsqueeze(1)  # [B, 1, M, 3]
    y_axis_expanded = y_axis.unsqueeze(1)  # [B, 1, M, 3]
    z_axis_expanded = z_axis.unsqueeze(1)  # [B, 1, M, 3]
    
    local_x = (relative_pos * x_axis_expanded).sum(dim=3)  # [B, N, M]
    local_y = (relative_pos * y_axis_expanded).sum(dim=3)  # [B, N, M]
    local_z = (relative_pos * z_axis_expanded).sum(dim=3)  # [B, N, M]
    
    # Normalize by extents to get coordinates in [-1, 1] range
    x_extent_expanded = x_extent.squeeze(2).unsqueeze(1)  # [B, 1, M]
    y_extent_expanded = y_extent.squeeze(2).unsqueeze(1)  # [B, 1, M]
    z_extent_expanded = z_extent.squeeze(2).unsqueeze(1)  # [B, 1, M]
    
    norm_x = local_x / (x_extent_expanded + 1e-8)  # [B, N, M]
    norm_y = local_y / (y_extent_expanded + 1e-8)  # [B, N, M]
    norm_z = local_z / (z_extent_expanded + 1e-8)  # [B, N, M]
    
    # Compute signed distance to box boundary
    # For a point inside: all |norm_coord| <= 1, distance is negative
    # For a point outside: distance is positive
    dist_x = torch.abs(norm_x) - 1.0
    dist_y = torch.abs(norm_y) - 1.0
    dist_z = torch.abs(norm_z) - 1.0
    
    # Overall distance is the maximum violation across dimensions
    distance = torch.maximum(torch.maximum(dist_x, dist_y), dist_z)  # [B, N, M]
  
    # Convert distance to scores (negative distance = higher probability)
    # Points inside boxes (negative distance) get higher scores
    box_scores = -distance / temperature  # [B, N, M]

    invalid_mask_expanded = (1 - valid_mask.unsqueeze(1).expand(-1, N, -1)).type(torch.bool)
    box_scores = torch.where(invalid_mask_expanded, 
                                 torch.tensor(float('-inf'), device=device, dtype=box_scores.dtype),
                                 box_scores)
    
    # Create background scores [B, N, 1]
    # Background score acts as a baseline - can be adjusted with background_bias
    background_scores = torch.full((B, N, 1), background_bias, 
                                   device=device, dtype=points.dtype)
    
    # Concatenate background and box scores: [B, N, M+1]
    all_scores = torch.cat([background_scores, box_scores], dim=2)  # [B, N, M+1]
    
    # Apply softmax over all M+1 classes to get probabilities
    probs = torch.softmax(all_scores, dim=2)  # [B, N, M+1]
    
    return probs
def construct_list_of_attributes():
    """Construct the list of attributes for the PLY file header."""
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # Add spherical harmonics coefficients
    for i in range(3):
        l.append(f'f_dc_{i}')
    for i in range(45):  # 45 additional SH coefficients for degree > 0
        l.append(f'f_rest_{i}')
    l.append('opacity')
    for i in range(3):
        l.append(f'scale_{i}')
    for i in range(4):
        l.append(f'rot_{i}')
    return l


def transform_gaussian_means_with_instances(
    context_instances_pose,  # [B, T_context, N_box, 4, 4]
    target_instances_pose,   # [B, T_target, N_box, 4, 4]
    means,                   # [B, T_context, N_gs, 3]
    probabilities           # [B, T_context, N_gs, N_box]
):
    """
    Transform 3D Gaussian means according to dynamic instance movements.
    
    Args:
        context_instances_pose: [B, T_context, N_box, 4, 4] object-to-world matrices at context times
        target_instances_pose: [B, T_target, N_box, 4, 4] object-to-world matrices at target times
        means: [B, T_context, N_gs, 3] 3D Gaussian centers at context times
        probabilities: [B, T_context, N_gs, N_box] probability of each Gaussian belonging to each instance
    
    Returns:
        target_means: [B, T_target, T_context, N_gs, 3] transformed Gaussian centers
    """

    B, T_context, N_gs, _ = means.shape
    T_target = target_instances_pose.shape[1]
    N_box = context_instances_pose.shape[2]
    
    # Extract rotations and translations
    ctx_R = context_instances_pose[..., :3, :3]  # [B, T_context, N_box, 3, 3]
    ctx_t = context_instances_pose[..., :3, 3]   # [B, T_context, N_box, 3]
    tgt_R = target_instances_pose[..., :3, :3]   # [B, T_target, N_box, 3, 3]
    tgt_t = target_instances_pose[..., :3, 3]    # [B, T_target, N_box, 3]
    
    # Step 1: Transform means from world to local frame at context time
    # local_pos = R_ctx^T @ (world_pos - t_ctx)
    # 
    # Expand dimensions:
    # means: [B, T_context, N_gs, 3] -> [B, T_context, 1, N_gs, 1, 3]
    # ctx_t: [B, T_context, N_box, 3] -> [B, T_context, 1, 1, N_box, 3]
    # ctx_R: [B, T_context, N_box, 3, 3] -> [B, T_context, 1, 1, N_box, 3, 3]
    
    means_exp = means.unsqueeze(2).unsqueeze(4)  # [B, T_context, 1, N_gs, 1, 3]
    ctx_t_exp = ctx_t.unsqueeze(2).unsqueeze(3)  # [B, T_context, 1, 1, N_box, 3]
    ctx_R_exp = ctx_R.unsqueeze(2).unsqueeze(3)  # [B, T_context, 1, 1, N_box, 3, 3]
    
    # Transform to local coordinates for each box
    # [B, T_context, 1, N_gs, N_box, 3]
    local_pos = torch.einsum('...ij,...j->...i', 
                             ctx_R_exp.transpose(-2, -1), 
                             means_exp - ctx_t_exp)
    
    # Step 2: Transform from local frame to world at target time
    # world_pos = R_tgt @ local_pos + t_tgt
    #
    # Expand target dimensions:
    # local_pos: [B, T_context, 1, N_gs, N_box, 3] -> [B, T_context, T_target, N_gs, N_box, 3]
    # tgt_R: [B, T_target, N_box, 3, 3] -> [B, 1, T_target, 1, N_box, 3, 3]
    # tgt_t: [B, T_target, N_box, 3] -> [B, 1, T_target, 1, N_box, 3]

    # local_pos_exp = local_pos.unsqueeze(2)  # [B, T_context, 1, N_gs, N_box, 3]
    tgt_R_exp = tgt_R.unsqueeze(1).unsqueeze(3)  # [B, 1, T_target, 1, N_box, 3, 3]
    tgt_t_exp = tgt_t.unsqueeze(1).unsqueeze(3)  # [B, 1, T_target, 1, N_box, 3]
    
    # Transform to world coordinates at target time for each box
    # [B, T_context, T_target, N_gs, N_box, 3]
    transformed_means = torch.einsum('...ij,...j->...i', 
                                     tgt_R_exp, 
                                     local_pos) + tgt_t_exp
    
    # Step 3: Weight by probabilities and sum over boxes
    # probabilities: [B, T_context, N_gs, N_box] -> [B, T_context, 1, N_gs, N_box, 1]
    probs_exp = probabilities.unsqueeze(2).unsqueeze(-1)
    
    # Weighted sum over boxes: [B, T_context, T_target, N_gs, 3]
    weighted_means = (transformed_means * probs_exp).sum(dim=-2)
    
    # Reorder to [B, T_target, T_context, N_gs, 3]
    target_means = weighted_means.transpose(1, 2)
    
    return target_means
def corners_to_params(corners):
    """
    Convert 3D bounding box corners to parametric representation.
    
    This follows the convention from get_box_corners where:
    - X-axis (width): Left (−) to Right (+)
    - Y-axis (height): Bottom (−) to Top (+)  
    - Z-axis (depth): Back (−) to Front (+)
    
    Corner ordering (binary encoding):
      0: [-X, -Y, -Z]  back-bottom-left
      1: [+X, -Y, -Z]  back-bottom-right
      2: [-X, +Y, -Z]  back-top-left
      3: [+X, +Y, -Z]  back-top-right
      4: [-X, -Y, +Z]  front-bottom-left
      5: [+X, -Y, +Z]  front-bottom-right
      6: [-X, +Y, +Z]  front-top-left
      7: [+X, +Y, +Z]  front-top-right
    
    Args:
        corners: torch.Tensor of shape [..., 8, 3]
                 8 corners in [x, y, z] format
    
    Returns:
        params: torch.Tensor of shape [..., 7]
                [center_x, center_y, center_z, width, height, depth, yaw]
                where yaw is rotation around Y-axis (vertical)
    """
    # 1. Center: mean of all 8 corners
    center = corners.mean(dim=-2)  # [..., 3]
    
    # 2. Compute edge vectors from corner 0
    # edge_x: 0 -> 1 (width direction, X-axis)
    # edge_y: 0 -> 2 (height direction, Y-axis)
    # edge_z: 0 -> 4 (depth direction, Z-axis)
    edge_x = corners[..., 1, :] - corners[..., 0, :]  # [..., 3]
    edge_y = corners[..., 2, :] - corners[..., 0, :]  # [..., 3]
    edge_z = corners[..., 4, :] - corners[..., 0, :]  # [..., 3]
    
    # 3. Dimensions (Euclidean distances)
    width = torch.norm(edge_x, dim=-1, keepdim=True)    # [..., 1]
    height = torch.norm(edge_y, dim=-1, keepdim=True)   # [..., 1]
    depth = torch.norm(edge_z, dim=-1, keepdim=True)    # [..., 1]
    
    # 4. Yaw angle: rotation around Y-axis (vertical)
    # Forward direction is +Z axis (depth direction)
    # Yaw = angle from +Z axis in the XZ plane
    # yaw = atan2(x_component, z_component)
    yaw = torch.atan2(edge_z[..., 0:1], edge_z[..., 2:3])  # [..., 1]
    
    # 5. Concatenate all parameters
    params = torch.cat([center, width, height, depth, yaw], dim=-1)  # [..., 7]
    
    return params

def construct_list_of_attributes():
    """Construct the list of attributes for the PLY file header."""
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # Add spherical harmonics coefficients
    for i in range(3):
        l.append(f'f_dc_{i}')
    for i in range(45):  # 45 additional SH coefficients for degree > 0
        l.append(f'f_rest_{i}')
    l.append('opacity')
    for i in range(3):
        l.append(f'scale_{i}')
    for i in range(4):
        l.append(f'rot_{i}')
    return l

def save_gaussians_to_ply(filepath, means, quats, scales, opacities, colors, opacity_threshold=0.005):
    """
    Save 3D Gaussians to a PLY file compatible with Gaussian Splatting viewers.
    Filters out low opacity Gaussians to reduce file size.
    
    Args:
        filepath: Path to save the PLY file
        means: Gaussian centers [N, 3]
        quats: Quaternions for rotation [N, 4] 
        scales: Actual scales [N, 3] (not log scales)
        opacities: Actual opacities [N, 1] or [N] in range [0, 1]
        colors: Colors [N, C] in range [0, 1] (RGB or SH coefficients)
        opacity_threshold: Minimum opacity to keep a Gaussian (default 0.005)
    """
    # Ensure everything is on CPU and convert to numpy
    means = means.detach().cpu().numpy() if isinstance(means, torch.Tensor) else means
    quats = quats.detach().cpu().numpy() if isinstance(quats, torch.Tensor) else quats
    scales = scales.detach().cpu().numpy() if isinstance(scales, torch.Tensor) else scales
    opacities = opacities.detach().cpu().numpy() if isinstance(opacities, torch.Tensor) else opacities
    colors = colors.detach().cpu().numpy() if isinstance(colors, torch.Tensor) else colors
    
    # Ensure opacities has the right shape
    if opacities.ndim == 1:
        opacities = opacities[:, None]
    
    # Filter by opacity threshold (opacities are already in [0, 1])
    mask = opacities[:, 0] > opacity_threshold
    
    # Apply mask to all arrays
    means = means[mask]
    quats = quats[mask]
    scales = scales[mask]
    opacities = opacities[mask]
    colors = colors[mask]
    
    # Number of Gaussians after filtering
    N = means.shape[0]
    print(f"Keeping {N} Gaussians out of {len(mask)} (removed {len(mask) - N} low-opacity Gaussians)")
    
    # Normalize quaternions
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    
    # Convert actual scales to log scales (INRIA format stores log scales)
    log_scales = np.log(scales + 1e-8)  # Add small epsilon to avoid log(0)
    
    # Convert colors from [0, 1] to SH coefficients
    # SH coefficients are typically stored as (color - 0.5) / 0.28209479177387814
    C0 = 0.28209479177387814
    if colors.shape[-1] >= 3:
        # Convert RGB to SH DC coefficients
        sh_dc = (colors[..., :3] - 0.5) / C0
    else:
        # Pad with zeros if less than 3 channels
        padded_colors = np.pad(colors, ((0, 0), (0, 3 - colors.shape[-1])), mode='constant', constant_values=0.5)
        sh_dc = (padded_colors - 0.5) / C0
    
    # Higher order SH coefficients (if available)
    if colors.shape[-1] > 3:
        # Assume these are already SH coefficients or additional color channels
        sh_rest = (colors[..., 3:48] - 0.5) / C0 if colors.shape[-1] > 3 else np.zeros((N, 45))
        # Pad if necessary
        if sh_rest.shape[-1] < 45:
            sh_rest = np.pad(sh_rest, ((0, 0), (0, 45 - sh_rest.shape[-1])), mode='constant')
    else:
        sh_rest = np.zeros((N, 45))
    
    # Construct the dtype for the structured array
    dtype_list = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    
    # Create structured array
    elements = np.zeros(N, dtype=dtype_list)
    
    # Fill in the data
    elements['x'] = means[:, 0]
    elements['y'] = means[:, 1] 
    elements['z'] = means[:, 2]
    # Normals (not used, set to 0)
    elements['nx'] = 0
    elements['ny'] = 0
    elements['nz'] = 0
    # DC spherical harmonics coefficients
    elements['f_dc_0'] = sh_dc[:, 0]
    elements['f_dc_1'] = sh_dc[:, 1]
    elements['f_dc_2'] = sh_dc[:, 2]
    # Higher order SH coefficients
    for i in range(45):
        elements[f'f_rest_{i}'] = sh_rest[:, i]
    # Opacity (already in [0, 1])
    elements['opacity'] = opacities[:, 0]
    # Scale (in log space for INRIA format)
    elements['scale_0'] = log_scales[:, 0]
    elements['scale_1'] = log_scales[:, 1]
    elements['scale_2'] = log_scales[:, 2]
    # Rotation quaternion
    elements['rot_0'] = quats[:, 0]
    elements['rot_1'] = quats[:, 1]
    elements['rot_2'] = quats[:, 2]
    elements['rot_3'] = quats[:, 3]
    
    # Create PLY element and save
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el], text=False).write(filepath)
    
    print(f"Saved {N} Gaussians to {filepath}")


def init_to_zero_output(module):
    """Initialize the final linear layer to zero for near-zero initial output."""
    # Find the last Linear layer
    last_linear = None
    for m in module:
        if isinstance(m, nn.Linear):
            last_linear = m
    
    # Zero-initialize the final layer
    if last_linear is not None:
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)


class UFO(ViT):
    def __init__(
        self,
        img_size=224,
        in_chans=9,
        gs_dim=3,
        decoder_type="dummy",
        near=0.2,
        far=400,
        scale_offset=-2.3,
        opacity_offset=-2.0,
        num_cams=3,  # to ablate
        max_scale=0.5,
        disable_pos_embed=False,
        use_sky_token=True,
        use_affine_token=True,
        num_motion_tokens=32,
        tau=0.5,
        projected_motion_dim=32,
        # ViT parameters
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        grad_checkpointing=True,
        use_latest_gsplat=False,
        sigmoid_rgb=False, # a legacy oversight: the sigmoid was accidentally omitted in the earlier implementation
        static=False,
        num_mem_tokens=0,
        args=None,
        **kwargs,
    ):
        super(UFO, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            grad_checkpointing=grad_checkpointing,
        )

        self.num_memory_tokens = num_mem_tokens
        self.enable_mem_gs = False
        self.args = args

        self.num_heads = num_heads

        self.static = static
        # basic attributes
        self.disable_pos_embed = disable_pos_embed
        self.gs_dim = gs_dim
        self.out_channels = gs_dim + 9
        self.num_cams = num_cams
        self.grad_checkpointing = grad_checkpointing
        self.use_latest_gsplat = use_latest_gsplat

        # ------- UFO v.s. Latent-UFO -------
        self.decoder_type = decoder_type
        self.decoder_upsample_ratio = decoder_upsample_ratio = self.patch_size

        # ------- motion predictor -------
        self.num_motion_tokens = num_motion_tokens
        self.tau = tau
        num_velocity_channels = 3


        ###============== bounding box for dynamic objects=====================
        # bbox motion predictor
        self.num_bbox = args.num_bbox
        ## bbox embedder

        # self.bbox_embed = Mlp(31, 32, embed_dim)
        self.bbox_time_embed = Mlp(embed_dim, embed_dim, embed_dim)
        self.bbox_embed = nn.Linear(31, embed_dim)
        self.background_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # used for upscaling the low-resolution image features to the pixel-resolution
        # very handcrafted and never tuned
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
            LayerNorm2d(512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            LayerNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            nn.GELU(),
        )
        self.bbox_query_head = Mlp(128, 256, projected_motion_dim)
        self.bbox_key_head = Mlp(self.embed_dim, self.embed_dim, projected_motion_dim)

        ###==========================================================================



        # ------- embedders -------
        self.plucker_embedder = PluckerEmbedder(img_size=img_size)
        self.time_embedder = TimestepEmbedder(embed_dim)

        # Ray encoder for posterior GS features
        # Processes 8x8 grid of Plucker ray coordinates into feature space
        self.ray_encoder_posterior = nn.Conv2d(
            in_channels=6,  # Plucker coordinates (6D)
            out_channels=embed_dim,
            kernel_size=8,
            stride=8
        )

        # ------- auxiliary tokens -------
        self.use_sky_token = use_sky_token
        self.use_affine_token = use_affine_token

        posterior = args.num_target_chunks > 1
        if posterior:
            self.adaptor_posterior_output = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )


            # self.adaptor_posterior_input = nn.Sequential(
            #     nn.Linear(embed_dim, embed_dim),
            #     nn.GELU(),
            #     nn.Linear(embed_dim, embed_dim)
            # )

        if self.use_sky_token:
            self.sky_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.sky_head = ModulatedLinearLayer(
                3,
                hidden_channels=512,
                condition_channels=embed_dim,
                out_channels=self.gs_dim,
            )

        if self.use_affine_token:
            self.affine_token = nn.Parameter(torch.randn(1, self.num_cams, embed_dim) * 0.02)
            self.affine_linear = nn.Linear(embed_dim, self.gs_dim * (self.gs_dim + 1))

        # ------- gs predictor and mask decoder -------
        if decoder_type == "dummy":
            self.gs_pred = nn.Linear(embed_dim, decoder_upsample_ratio**2 * self.out_channels)
            self.gs_life_pred = nn.Linear(embed_dim, decoder_upsample_ratio ** 2)
            if self.num_memory_tokens > 0 and self.enable_mem_gs:
                self.mem_gs_pred = nn.Linear(embed_dim, decoder_upsample_ratio ** 2 * 14)
            else:
                self.mem_gs_pred = None
            self.decoder = DummyDecoder()
            self.unpatch_size = decoder_upsample_ratio

            if self.num_motion_tokens > 0:
                if self.decoder_upsample_ratio == 8:
                    # used for upscaling the low-resolution image features to the pixel-resolution
                    # very handcrafted and never tuned
                    self.output_upscaling = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
                        LayerNorm2d(512),
                        nn.GELU(),
                        nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                        LayerNorm2d(256),
                        nn.GELU(),
                        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                        LayerNorm2d(128),
                        nn.GELU(),
                    )
                elif self.decoder_upsample_ratio == 16:
                    # used for upscaling the low-resolution image features to the pixel-resolution
                    # very handcrafted and never tuned
                    self.output_upscaling = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
                        LayerNorm2d(512),
                        nn.GELU(),
                        nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                        LayerNorm2d(256),
                        nn.GELU(),
                        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                        LayerNorm2d(128),
                        nn.GELU(),
                        nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                        LayerNorm2d(128),
                        nn.GELU(),
                    )

        else:
            raise ValueError(
                f"Invalid decoder_type={decoder_type!r}; only 'dummy' is supported. "
                "The 'conv' branch was removed in cleanup; restore ConvDecoder + its layer "
                "dependencies (GroupNorm/Swish/ResidualBlock/UpSampleBlock/NonLocalBlock) to use it."
            )
        # ------- activation functions for gs parameters -------
        self.max_scale = nn.Parameter(torch.tensor([float(max_scale)]), requires_grad=False)
        self.scale_act_fn = lambda x: torch.minimum(torch.exp(x + scale_offset), self.max_scale)
        self.opacity_act_fn = lambda x: torch.sigmoid(x + opacity_offset)
        self.depth_act_fn = lambda x: near + torch.sigmoid(x) * (far - near)

        # Softplus activation for lifespan: log(1 + exp(x))
        # More stable than exp, smoother gradients, range: (0, ∞)
        self.life_act_fn = lambda x: F.softplus(x)
        def inverse_log_transform(y):
            """
            Apply inverse log transform: sign(y) * (exp(|y|) - 1)

            Args:
                y: Input tensor

            Returns:
                Transformed tensor
            """
            return torch.sign(y) * (torch.expm1(torch.abs(y / 50)))
        self.xyz_act_fn = inverse_log_transform

        # self.xyz_act_fn = lambda x: far * torch.tanh(x / 50)
        # self.xyz_act_fn = lambda x: x
        self.rgb_act_fn = lambda x: torch.sigmoid(x) * 2 - 1 if sigmoid_rgb else x
        self.near, self.far = near, far

        # ------- motion predictor -------
        if self.num_motion_tokens > 0:
            self.motion_key_head = Mlp(128, 256, projected_motion_dim)
            self.motion_tokens = nn.Parameter(torch.randn(1, num_motion_tokens, embed_dim) * 0.02)
            self.motion_query_heads = nn.ModuleList(
                [
                    Mlp(embed_dim, embed_dim, projected_motion_dim)
                    for _ in range(self.num_motion_tokens)
                ]
            )
            self.motion_basis_decoder = Mlp(embed_dim, 256, num_velocity_channels)
        else:
            self.motion_tokens = None
            self.motion_key_head = None
            self.motion_basis_decoder = None

        if self.num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(1, self.num_memory_tokens, embed_dim) * 0.02)
        else:
            self.memory_tokens = None

        self.init_weights()
        if posterior:
            init_to_zero_output(self.adaptor_posterior_output)
            # init_to_zero_output(self.adaptor_posterior_input)

        if disable_pos_embed:  # remove the default pos_embed in vit
            del self.pos_embed
            self.pos_embed = None



        if self.static:
            assert self.num_motion_tokens == 0

    def _pos_embed(self, x: Tensor) -> Tensor:
        if not self.disable_pos_embed:
            return super()._pos_embed(x)
        return rearrange(x, "b h w c -> b (h w) c")

    def _time_embed(self, x: Tensor, time: Tensor, num_views=1) -> Tensor:
        if time.ndim == 3:
            b, t, v = time.shape
            time_embedding = (
                self.time_embedder(time.flatten())  # (bt, c)
                .view(b, t, v, -1)  # (b, t, v, c)
                .view(-1, 1, self.embed_dim)  # (btv, 1, c)
                .repeat(1, x.shape[1], 1)  # (btv, n, c)
            )
        else:
            time_embedding = (
                self.time_embedder(time.flatten())  # (bt, c)
                .view(time.shape[0], time.shape[1], 1, -1)  # (b, t, 1, c)
                .repeat(1, 1, num_views, 1)  # (b, t, v, c)
                .view(-1, 1, self.embed_dim)  # (btv, 1, c)
                .repeat(1, x.shape[1], 1)  # (btv, n, c)
            )
        return x + time_embedding

    def forward_decoder(self, render_results):
        render_results["rgb_key"] = "rendered_image"
        render_results["depth_key"] = "rendered_depth"
        render_results["alpha_key"] = "rendered_alpha"
        render_results["flow_key"] = "rendered_flow"
        render_results["decoder_depth_key"] = None
        render_results["decoder_alpha_key"] = None
        render_results["decoder_flow_key"] = None
        render_results = self.decoder(render_results)
        decoded_depth_key = render_results["decoder_depth_key"]
        if decoded_depth_key is not None:
            decoded_depth = self.depth_act_fn(render_results[decoded_depth_key])
            render_results[decoded_depth_key] = decoded_depth
        return render_results


    def init_feature_from_patch(self, image_patches, ray_origins, ray_directions, patch_times, setnum=True):
        """
        image_patches: [B, L, 8, 8, 3]
        ray_origins:   [B, L, 8, 8, 3]
        ray_directions:[B, L, 8, 8, 3]
        patch_times:   [B, L]
        """
        B, L, H, W, C = image_patches.shape  # H=W=8, C=3

        # 1. Compute Plucker coordinates from ray origins/directions
        dirs_flat    = rearrange(ray_directions, 'b l h w c -> (b l) h w c')  # [B*L, 8, 8, 3]
        origins_flat = rearrange(ray_origins,    'b l h w c -> (b l) h w c')  # [B*L, 8, 8, 3]
        ray_dict = self.plucker_embedder.forward_from_rays(
            rays_origins=origins_flat,
            rays_directions=dirs_flat,
            image_size=(H, W),
        )
        plucker = ray_dict['plucker']  # [B*L, 8, 8, 6]

        # 2. Concatenate image + Plucker → 9 channels (mirrors init_feature_image)
        image_flat  = rearrange(image_patches, 'b l h w c -> (b l) c h w')  # [B*L, 3, 8, 8]
        plucker_chw = rearrange(plucker,       'bl h w c -> bl c h w')       # [B*L, 6, 8, 8]
        x = torch.cat([image_flat, plucker_chw], dim=1)                       # [B*L, 9, 8, 8]

        # 3. Apply patch_embed per patch independently.
        #    patch_embed is Conv2d(9, 768, kernel=8, stride=8). Applied to an 8×8 input this is
        #    equivalent to a linear projection — no cross-patch information leaks.
        x = self.patch_embed(x)  # [B*L, 1, 1, 768] (NHWC output format)

        # 4. Skip _pos_embed — patches come from arbitrary spatial locations, so
        #    a spatial positional embedding would be meaningless.
        x = rearrange(x, 'bl 1 1 d -> bl d')  # [B*L, 768]

        # 5. Add per-patch time embedding (patch_times is [B, L], flatten to [B*L])
        time_embedding = self.time_embedder(patch_times.flatten()).view(B * L, -1)
        x = x + time_embedding

        # 6. Reshape to [B, L, D]
        x = rearrange(x, '(b l) d -> b l d', b=B, l=L)

        if setnum:
            self.num_patch_feature = x.shape[1]
        return x




    def init_posterior_gs_from_rays(self, posterior_gs, posterior_gs_dirs, posterior_gs_origins, posterior_gs_time):
        """
        posterior_gs: [B, L, D=768] (D: same dimension as other features)
        posterior_gs_dirs: [B, L, 8, 8, 3]
        posterior_gs_origins: [B, L, 8, 8, 3]
        posterior_gs_time: [B, L]

        return GS features: [B, L, D]
        """
        B, L, D = posterior_gs.shape

        # 1. Convert rays to Plucker coordinates
        # Reshape to process all tokens at once: [B, L, 8, 8, 3] -> [B*L, 8, 8, 3]
        dirs_flat = rearrange(posterior_gs_dirs, 'b l h w c -> (b l) h w c')
        origins_flat = rearrange(posterior_gs_origins, 'b l h w c -> (b l) h w c')

        # Get Plucker coordinates using forward_from_rays
        # PluckerEmbedder has default patch_size=1, so grid_size = (8//1, 8//1) = (8, 8)
        # This correctly outputs [B*L, 8, 8, 6] plucker embeddings
        ray_dict = self.plucker_embedder.forward_from_rays(
            rays_origins=origins_flat,        # [B*L, 8, 8, 3]
            rays_directions=dirs_flat,        # [B*L, 8, 8, 3]
            image_size=(8, 8)                 # Size of the ray grid
        )
        plucker = ray_dict['plucker']  # [B*L, 8, 8, 6]

        # 2. Process through ray encoder
        # Rearrange for Conv2d: [B*L, 8, 8, 6] -> [B*L, 6, 8, 8]
        plucker = rearrange(plucker, 'bl h w c -> bl c h w')

        # Apply Conv2d: [B*L, 6, 8, 8] -> [B*L, D, 1, 1]
        ray_features = self.ray_encoder_posterior(plucker)  # [B*L, D, 1, 1]

        # Reshape to [B, L, D]
        ray_features = rearrange(ray_features, '(b l) d 1 1 -> b l d', b=B, l=L)

        # 3. Combine with existing posterior_gs features (residual connection)
        combined_features = posterior_gs + ray_features

        # 4. Add time embedding
        # TimestepEmbedder expects 1D tensor of time values [N] -> outputs [N, embed_dim]
        # Flatten [B, L] -> [B*L], get embeddings, then reshape back to [B, L, embed_dim]
        time_embedding = self.time_embedder(posterior_gs_time.flatten()).view(B, L, -1)
        output = combined_features + time_embedding

        return output
    def init_posterior_gs(self, x, plucker_embeds, time):
        ### TODO
        b, t, v, h, w, c = x.size()
        # x = rearrange(x, "b t v c h w -> (b t v) c h w")
        x = rearrange(x, "b t v h w c -> (b t v) c h w")
        plucker_embeds = rearrange(plucker_embeds, "b t v h w c-> (b t v) c h w")
        x = torch.cat([x, plucker_embeds], dim=1) # on top of rgb dimensions, add 6 additional dimensions

        # patch_embed is simply a conv layer, Conv2d(9, 768, kernel_size=(8, 8), stride=(8, 8))
        # patches are non-overlapping
        x = self.patch_embed_gs(x)  # (b t v) c1 h w -> (b t v) h w c2

        # denotes image position, has learnable embedding 1, h'xw', c2 and added to x
        # x = self._pos_embed(x)  # (b t v) (h w) c2

        x = rearrange(x, "btv h w c -> btv (h w) c")

        # time embedding is sinusoidal, after that a simple MLP is used to upsample feature to 768
        # and repeated across spatial dimensions
        # and added to x
        x = self._time_embed(x, time, num_views=v)

        # B L D
        x = rearrange(x, "(b t v) hw c -> b (t v hw) c", t=t, v=v)


        # core computation
        # simple transformer block repeated 12 times
        # self.num_image_feature = x.shape[1]
        return x

        pass    
    def init_posterior_gs_from_ray(self, gs, gs_plucker, gs_time):
        ### TODO
        import ipdb ; ipdb.set_trace()
    def init_feature_gs_legacy(self, gs_plucker, gs_time):
        b = gs_plucker.shape[0]
        tt = gs_plucker.shape[1]
        v = gs_plucker.shape[2]
        gs_plucker = rearrange(gs_plucker, "b tt v h  w c-> (b tt v) c h w")
        gs_plucker = self.target_patch_embed(gs_plucker)

        gs_plucker = rearrange(gs_plucker, "(b tt v) c h w -> b tt v c h w", b=b, tt=tt)

        ### pose embedding needed?
        x = rearrange(gs_plucker, "b tt v c h w -> (b tt v) c h w")
        x = self._pos_embed(x)  # (b t v) (h w) c2

        ### time embedding
        x = self._time_embed(x, gs_time, num_views=v)

        # B L D
        x = rearrange(x, "(b tt v) hw c -> b (tt v hw) c", tt=tt, v=v)

        self.num_gs_feature = x.shape[1]

        return x



    def init_feature_all(self, x, plucker_embeds, time, gs_plucker, gs_time):
        b, t, v, c, h, w = x.size()
        tt = gs_plucker.shape[1]
        x = rearrange(x, "b t v c h w -> (b t v) c h w")
        plucker_embeds = rearrange(plucker_embeds, "b t v h w c-> (b t v) c h w")

        gs_plucker = rearrange(gs_plucker, "b tt v h w c-> (b tt v) c h w")
        x = torch.cat([x, plucker_embeds], dim=1) # on top of rgb dimensions, add 6 additional dimensions

        # patch_embed is simply a conv layer, Conv2d(9, 768, kernel_size=(8, 8), stride=(8, 8))
        # patches are non-overlapping
        x = self.patch_embed(x)  # (b t v) c1 h w -> (b t v) h w c2

        gs_plucker = self.target_patch_embed(gs_plucker)

        x = torch.cat([rearrange(x, "(b t v) c h w -> b t v c h w", b=b, t=t), rearrange(gs_plucker, "(b tt v) c h w -> b tt v c h w", b=b, tt=tt)], dim=1)
        x = rearrange(x, "b t v c h w -> (b t v) c h w")

        # denotes image position, has learnable embedding 1, h'xw', c2 and added to x
        x = self._pos_embed(x)  # (b t v) (h w) c2

        # time embedding is sinusoidal, after that a simple MLP is used to upsample feature to 768
        # and repeated across spatial dimensions
        # and added to x
        time = torch.cat([time, gs_time], dim=1)
        x = self._time_embed(x, time, num_views=v)

        # B L D
        x = rearrange(x, "(b t v) hw c -> b (t v hw) c", t=t+tt, v=v)


        # these additional tokens are directly learnable, initialized by randn * 0.02
        if self.num_motion_tokens > 0:
            motion_tokens = repeat(self.motion_tokens, "1 k d -> b k d", b=x.shape[0]) # N_motion(16) positions
            x = torch.cat([motion_tokens, x], dim=-2)
        if self.use_affine_token:
            affine_token = repeat(self.affine_token, "1 k d -> b k d", b=b) # N_cam positions
            x = torch.cat([affine_token, x], dim=-2)
        if self.use_sky_token:
            sky_token = repeat(self.sky_token, "1 1 d -> b 1 d", b=x.shape[0]) # 1 position
            x = torch.cat([sky_token, x], dim=-2)

        # core computation
        # simple transformer block repeated 12 times
        return x

    def init_feature_mis(self, batch_size):
        # these additional tokens are directly learnable, initialized by randn * 0.02
        features = []
        if self.use_sky_token:
            sky_token = repeat(self.sky_token, "1 1 d -> b 1 d", b=batch_size) # 1 position
            features.append(sky_token)
        if self.use_affine_token:
            affine_token = repeat(self.affine_token, "1 k d -> b k d", b=batch_size) # N_cam positions
            features.append(affine_token)
        if self.num_motion_tokens > 0:
            motion_tokens = repeat(self.motion_tokens, "1 k d -> b k d", b=batch_size) # N_motion(16) positions
            features.append(motion_tokens)
        if self.num_memory_tokens > 0:
            memory_tokens = repeat(self.memory_tokens, "1 k d -> b k d", b=batch_size)
            features.append(memory_tokens)
        features = torch.cat(features, dim=-2) if len(features) > 0 else torch.empty((batch_size, 0, self.embed_dim)).cuda()
        self.num_mis_feature = features.shape[1]
        return features

    def init_feature_image(self, x, plucker_embeds, time, setnum=True):
        b, t, v, c, h, w = x.size()
        x = rearrange(x, "b t v c h w -> (b t v) c h w")
        plucker_embeds = rearrange(plucker_embeds, "b t v h w c-> (b t v) c h w")
        x = torch.cat([x, plucker_embeds], dim=1) # on top of rgb dimensions, add 6 additional dimensions

        # patch_embed is simply a conv layer, Conv2d(9, 768, kernel_size=(8, 8), stride=(8, 8))
        # patches are non-overlapping
        x = self.patch_embed(x)  # (b t v) c1 h w -> (b t v) h w c2

        # denotes image position, has learnable embedding 1, h'xw', c2 and added to x
        x = self._pos_embed(x)  # (b t v) (h w) c2

        # time embedding is sinusoidal, after that a simple MLP is used to upsample feature to 768
        # and repeated across spatial dimensions
        # and added to x
        x = self._time_embed(x, time, num_views=v)

        # B L D
        x = rearrange(x, "(b t v) hw c -> b (t v hw) c", t=t, v=v)


        # core computation
        # simple transformer block repeated 12 times
        if setnum:
            self.num_image_feature = x.shape[1]
        return x

    def forward_features(self, x):
        # main computation
        x = self.transformer(x) # go to ufo/models/layers/Transformer

        # layer norm (token-wise, keep feature scale in control, training stability)
        x = self.norm(x)
        return x

    def forward_motion_predictor(self, x, motion_tokens=None, gs_params=None):
        b, t, v, h, w, _ = gs_params["means"].shape

        # B L D -> (B T N_cam) D H' W'
        img_embeds = self.unpatchify(
            rearrange(x, "b (t v hw) c -> (b t v) hw c", t=t, v=v),
            hw=(h // self.unpatch_size, w // self.unpatch_size),
            patch_size=1,
        )

        # output_upscaling is a seriece of upconvnet with net effect: channel 768 -> 128, H' W' -> H W (restores original size)
        if self.grad_checkpointing:
            img_embeds = checkpoint(self.output_upscaling, img_embeds, use_reentrant=False)
        else:
            img_embeds = self.output_upscaling(img_embeds)
        img_embeds = rearrange(img_embeds, "(b t v) c h w -> b t v h w c", t=t, v=v) # B, T, N_cam, H, W, 128



        

        if self.num_motion_tokens > 0:
            # motion_key_head is lineary layer with net effect: channel 128 -> 32
            img_keys = self.motion_key_head(img_embeds) # B, T, N_cam, H, W, 32
            hyper_in_list = []
            for i in range(self.num_motion_tokens):

                # motion_query_heads are different for different motion tokens
                # motoin_query_heads are self.num_motion_tokens x Linear layer
                hyper_in = self.motion_query_heads[i](motion_tokens[:, i])
                hyper_in_list.append(hyper_in)
            motion_token_queries = torch.stack(hyper_in_list, dim=1) # B, num_motion_tokens, 32

            # linear layer with net effect: 768 -> 3
            # motion_bases serve as value is cross attention
            motion_bases = self.motion_basis_decoder(motion_tokens)
            dot_product_similarity = torch.einsum(
                "b k c, b t v h w c -> b t v h w k",
                motion_token_queries,
                img_keys,
            )
            motion_weights = torch.softmax(dot_product_similarity / self.tau, dim=-1)

            # forward scene flow B, T, N_cam, H, W, 3
            forward_flow = torch.einsum(
                "b t v h w k, b k c -> b t v h w c", motion_weights, motion_bases
            )
            gs_params["motion_weights"] = motion_weights
            gs_params["motion_bases"] = motion_bases
        else:
            # if there's no motion token, directly predict the velocity from the upsampled image features
            # forward_flow = self.motion_basis_decoder(img_keys)
            forward_flow = torch.zeros_like(gs_params['means'])

        gs_params["forward_flow"] = forward_flow
        return {k: v for k, v in gs_params.items() if v is not None}
    

    def forward_motion_predictor_bbox(self, data_dict, gs_params):

        x = data_dict['gs_state']
        b, t, v, h, w, _ = gs_params["means"].shape


        # if False:
        #     x = self.relative_gs_decoder(x[..., 3:])

        # todo: add gs means information


        # B L D -> (B T N_cam) D H' W'
        img_embeds = self.unpatchify(
            rearrange(x, "b (t v hw) c -> (b t v) hw c", t=t, v=v),
            hw=(h // self.unpatch_size, w // self.unpatch_size),
            patch_size=1,
        )

        # output_upscaling is a seriece of upconvnet with net effect: channel 768 -> 128, H' W' -> H W (restores original size)
        if False:
            img_embeds = checkpoint(self.output_upscaling, img_embeds, use_reentrant=False)
        else:
            img_embeds = self.output_upscaling(img_embeds)
        img_embeds = rearrange(img_embeds, "(b t v) c h w -> b t v h w c", t=t, v=v) # B, T, N_cam, H, W, 128



        if True:
            img_queries = self.bbox_query_head(img_embeds)
            bbox_keys = self.bbox_key_head(data_dict['bbox_feature'])
            bbox_weights = torch.zeros(b, t, v, h, w, 1 + self.num_bbox).to(img_queries)
            for _t in range(t):

                _bbox_keys = torch.cat([bbox_keys[:, :1], bbox_keys[:, 1 + _t * self.num_bbox : 1 + (_t + 1) * self.num_bbox]], dim=1)
                _img_queries = img_queries[:, _t]
                _dot_product_similarity = torch.einsum(
                    "b k c, b v h w c -> b v h w k",
                    _bbox_keys,
                    _img_queries
                )

                # TODO mask out invalid keys
                # data_dict['context_instances_id']

                _bbox_weights = torch.softmax(_dot_product_similarity / self.tau, dim=-1)
                bbox_weights[:, _t] = _bbox_weights


        data_dict['bbox_weights'] = bbox_weights

        return data_dict


    def forward_memory_gs(self, x):

        # embed_dim: 768 -> [x, y, z, q1, q2, q3, q4, r, g, b, s1, s2, s3, o]
        b, ntk, d = x.shape

        gs_params = self.mem_gs_pred(x)

        gs_params = gs_params.reshape(b, ntk * self.patch_size ** 2, -1)

        means, scales, quats, opacitys, colors = gs_params.split([3, 3, 4, 1, self.gs_dim], dim=-1)

        scales = self.scale_act_fn(scales)
        opacitys = self.opacity_act_fn(opacitys)
        means = self.depth_act_fn(means)
        colors = self.rgb_act_fn(colors)


        return {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacitys.squeeze(-1),
            "colors": colors,
            "depth": torch.ones_like(means[..., 0])
        }


    def forward_gs_predictor(self, x, origins, directions, return_whole=False):
        b, t, v, h, w, _ = origins.shape
        x = rearrange(x, "b (t v hw) c -> (b t v) hw c", t=t, v=v)

        # gs_pred is a linear layer with input dim 768 output dim 768 
        # this is a coincidence, the out dimension is 768 because 768 = 8 ^ 2 x 12
        gs_params = self.gs_pred(x)
        gs_lifespan = self.gs_life_pred(x)
        # note here each token(patch) predicts patch_size ^ 2 gaussians
        gs_params = self.unpatchify(gs_params, hw=(h, w), patch_size=self.unpatch_size)
        gs_params = rearrange(gs_params, "(b t v) c h w -> b t v h w c", t=t, v=v)
        gs_lifespan = self.unpatchify(gs_lifespan, hw=(h, w), patch_size=self.unpatch_size)
        gs_lifespan = rearrange(gs_lifespan, "(b t v) c h w -> b t v h w c", t=t, v=v)
        gs_params = torch.cat([gs_params, gs_lifespan], dim=-1)
        if return_whole:
            return gs_params
        depth, scales, quats, opacitys, colors, lifespan = gs_params.split([1, 3, 4, 1, self.gs_dim, 1], dim=-1)
        scales = self.scale_act_fn(scales)
        opacitys = self.opacity_act_fn(opacitys)
        depths = self.depth_act_fn(depth)
        colors = self.rgb_act_fn(colors)
        lifespan = self.life_act_fn(lifespan)


        # Note: the definition of depths here is the distance to the origin
        means = origins + directions * depths

        return {
            "means": means,
            "scales": scales,# * depths.detach(), # correlate scale and depth
            "quats": quats,
            "opacities": opacitys.squeeze(-1),
            "colors": colors,
            "lifespan": lifespan
        }

    def forward_renderer(self, gs_params, data_dict, render_motion_seg=True, radius_clip=0.0, visualize_only=False, save_ply_name="gaussians.ply", static_only=False):
        b, t, v, h, w, _ = gs_params["means"].shape
        tgt_h, tgt_w = data_dict["height"], data_dict["width"]
        tgt_t, tgt_v = data_dict["target_camtoworlds_global"].shape[1:3]
        means = rearrange(gs_params["means"], "b t v h w c -> b (t v h w) c")
        scales = rearrange(gs_params["scales"], "b t v h w c -> b (t v h w) c")
        quats = rearrange(gs_params["quats"], "b t v h w c -> b (t v h w) c")
        opacities = rearrange(gs_params["opacities"], "b t v h w -> b (t v h w)")
        colors = rearrange(gs_params["colors"], "b t v h w c -> b (t v h w) c")
        forward_v = rearrange(gs_params["forward_flow"], "b t v h w c -> b (t v h w) c") if 'forward_flow' in gs_params.keys() else torch.zeros_like(means)

        lifespan = rearrange(gs_params['lifespan'], "b t v h w c -> b (t v h w) c")
        ### transform gaussian means

        probabilities = rearrange(data_dict['bbox_weights'], "b t v h w k -> b t (v h w) k")
        t_means = rearrange(gs_params["means"], "b t v h w c -> b t (v h w) c")


        with torch.no_grad():
            gt_prob = torch.zeros_like(probabilities)
            ### TODO: should be able to avoid loop
            for _t in range(t):
                gt_prob[:, _t] = points_in_boxes_probability(t_means[:, _t].detach(), data_dict['context_instances_corner'][:, _t], data_dict['context_instances_id'][:, _t], temperature=0.01)

        gt_prob = gt_prob.argmax(dim=-1)
        loss_mask = gt_prob.float() > -1
        loss_gt_prob = gt_prob[loss_mask]
        loss_pred_porb = probabilities[loss_mask]
        log_probs = torch.log(loss_pred_porb + 1e-8)  # Add small epsilon to avoid log(0)

        class_weights = torch.ones(self.num_bbox + 1)
        class_weights[0] = 1e-1
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(log_probs.device)  # Move to same device

        loss = F.nll_loss(log_probs, loss_gt_prob, weight=class_weights)
        data_dict['class_loss'] = loss * 1e-2


        dummy_context_instance_pose = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(b, t, 1, 1, 1).to(probabilities)
        dummy_target_instance_pose = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(b, tgt_t, 1, 1, 1).to(probabilities)

        ### bbox not related to time
        transformed_means = transform_gaussian_means_with_instances(
            torch.cat([dummy_context_instance_pose, data_dict['context_instances_pose']], dim=2),  # [B, T_context, N_box, 4, 4]
            torch.cat([dummy_target_instance_pose, data_dict['target_instances_pose']], dim=2),   # [B, T_target, N_box, 4, 4]
            t_means,                   # [B, T_context, N_gs, 3]
            probabilities           # [B, T_context, N_gs, N_box]
        )

        transformed_means = rearrange(transformed_means, "b t v n c -> (b t) (v n) c")


        ntk_dynamic = forward_v.shape[1]

        if static_only:
            ntk_dynamic = 0
            means = means[:, :ntk_dynamic]
            scales = scales[:, :ntk_dynamic]
            quats = quats[:, :ntk_dynamic]
            opacities = opacities[:, :ntk_dynamic]
            colors = colors[:, :ntk_dynamic]
            forward_v = forward_v[:, :ntk_dynamic]


        if 'mem_gs_params' in gs_params:
            means = torch.cat([means, gs_params['mem_gs_params']['means']], dim=1)
            scales = torch.cat([scales, gs_params['mem_gs_params']['scales']], dim=1)
            quats = torch.cat([quats, gs_params['mem_gs_params']['quats']], dim=1)
            opacities = torch.cat([opacities, gs_params['mem_gs_params']['opacities']], dim=1)
            colors = torch.cat([colors, gs_params['mem_gs_params']['colors']], dim=1)
            forward_v = torch.cat([forward_v, torch.ones_like(gs_params['mem_gs_params']['means'])], dim=1)
            ntk_static = gs_params['mem_gs_params']['means'].shape[1]



        means_batched = means.repeat_interleave(tgt_t, dim=0)
        scales_batched = scales.repeat_interleave(tgt_t, dim=0)
        quats_batched = quats.repeat_interleave(tgt_t, dim=0)
        opacities_batched = opacities.repeat_interleave(tgt_t, dim=0)
        color_batched = colors.repeat_interleave(tgt_t, dim=0)
        forward_v_batched = forward_v.repeat_interleave(tgt_t, dim=0)

        lifespan_batched = lifespan.repeat_interleave(tgt_t, dim=0)

        ctx_time = data_dict["gs_time"] * data_dict["timespan"]
        tgt_time = data_dict["target_time"] * data_dict["timespan"]
        if tgt_time.ndim == 3:
            tdiff_forward = tgt_time.unsqueeze(2) - ctx_time.unsqueeze(1)
            tdiff_forward = tdiff_forward.view(b * tgt_t, t * v, 1)
            tdiff_forward_batched = tdiff_forward.repeat_interleave(h * w, dim=1)
        else:
            tdiff_forward = tgt_time.unsqueeze(-1) - ctx_time.unsqueeze(-2)
            tdiff_forward = tdiff_forward.view(b * tgt_t, t, 1)
            tdiff_forward_batched = tdiff_forward.repeat_interleave(v * h * w, dim=1)
        if not static_only and not self.static:
            forward_translation = forward_v_batched[:, :ntk_dynamic] * tdiff_forward_batched
            means_batched[:, :ntk_dynamic] = means_batched[:, :ntk_dynamic] + forward_translation


        ### lifespan for opacity
        enable_lifespan = False
        if enable_lifespan:
            opacities_batched = opacities_batched * torch.exp(- 0.5 * tdiff_forward_batched ** 2 * lifespan_batched).squeeze(-1)
            


        if not self.training:  # mask out some noisy flow
            forward_v[forward_v.norm(dim=-1) < 1.0] = 0.0
            forward_v_batched = forward_v.repeat_interleave(tgt_t, dim=0)

        if not self.training:

            weights_batched = rearrange(probabilities, "b v n c -> b (v n) c").repeat(tgt_t, 1, 1)

            # Include lifespan for visualization (1 channel)
            colors_batched = torch.cat([color_batched, forward_v_batched, lifespan_batched, weights_batched], dim=-1)
        else:
            colors_batched = torch.cat([color_batched, forward_v_batched], dim=-1)

        camtoworlds_batched = data_dict["target_camtoworlds_global"].view(b * tgt_t, -1, 4, 4)
        viewmats_batched = torch.linalg.inv(camtoworlds_batched.float())
        Ks_batched = data_dict["target_intrinsics"].view(b * tgt_t, -1, 3, 3)

        motion_seg = None


        # replace means batched 
        means_batched = transformed_means



        if visualize_only:
            save_gaussians_to_ply(
                save_ply_name,
                means=means_batched[0].float(),  # Take first batch (first timestep)
                quats=quats_batched[0].float(),
                scales=scales_batched[0].float(),
                opacities=opacities_batched[0].float(),
                colors=-colors_batched[0, :, :3].float() * 0.5 + 0.5,
                opacity_threshold=0.05
            )
            return
        if self.use_latest_gsplat:
            raise Exception("deprecated")
            means_batched = means_batched.float()
            quats_batched = quats_batched.float()
            scales_batched = scales_batched.float()
            opacities_batched = opacities_batched.float()
            colors_batched = colors_batched.float()
            viewmats_batched = viewmats_batched.float()
            Ks_batched = Ks_batched.float()

            if not self.training:
                rendered_colors, rendered_alphas, rendered_flow, motion_seg = [], [], [], []
                rendered_depths, rendered_lifespans = [], []
                with torch.autocast("cuda", enabled=False):
                    for bid in range(means_batched.size(0)):
                        renderings, alpha, _ = rasterization(
                            means=means_batched[bid],
                            quats=quats_batched[bid],
                            scales=scales_batched[bid],
                            opacities=opacities_batched[bid],
                            colors=colors_batched[bid],
                            viewmats=viewmats_batched[bid],
                            Ks=Ks_batched[bid],
                            width=data_dict["width"],
                            height=data_dict["height"],
                            render_mode="RGB+ED",
                            near_plane=self.near,
                            far_plane=self.far,
                            packed=False,
                            radius_clip=radius_clip,
                        )
                        color, forward_flow, weights, rendered_lifespan, depth = renderings.split(
                            [self.gs_dim, 3, self.num_motion_tokens, 1, 1], dim=-1
                        )
                        rendered_colors.append(color)
                        rendered_alphas.append(alpha)
                        rendered_flow.append(forward_flow)
                        motion_seg.append(weights)
                        rendered_depths.append(depth)
                        rendered_lifespans.append(rendered_lifespan)
                color = torch.stack(rendered_colors, dim=0)
                rendered_alpha = torch.stack(rendered_alphas, dim=0)
                forward_flow = torch.stack(rendered_flow, dim=0)
                depth = torch.stack(rendered_depths, dim=0)
                rendered_lifespan = torch.stack(rendered_lifespans, dim=0)
                motion_seg = torch.stack(motion_seg, dim=0)
                if motion_seg.numel() > 0:
                    motion_seg = motion_seg.reshape(b, tgt_t, v, h, w, -1).argmax(dim=-1)
                else:
                    motion_seg = None
            else:
                rendered_colors, rendered_alphas, rendered_flow, rendered_depths = [], [], [], []
                with torch.autocast("cuda", enabled=False):
                    for bid in range(means_batched.size(0)):
                        renderings, alpha, _ = rasterization(
                            means=means_batched[bid],
                            quats=quats_batched[bid],
                            scales=scales_batched[bid],
                            opacities=opacities_batched[bid],
                            colors=colors_batched[bid],
                            viewmats=viewmats_batched[bid],
                            Ks=Ks_batched[bid],
                            width=data_dict["width"],
                            height=data_dict["height"],
                            render_mode="RGB+ED",
                            near_plane=self.near,
                            far_plane=self.far,
                            packed=False,
                            radius_clip=radius_clip,
                        )
                        color, forward_flow, depth = renderings.split([self.gs_dim, 3, 1], dim=-1)
                        rendered_colors.append(color)
                        rendered_alphas.append(alpha)
                        rendered_flow.append(forward_flow)
                        rendered_depths.append(depth)
                color = torch.stack(rendered_colors, dim=0)
                rendered_alpha = torch.stack(rendered_alphas, dim=0)
                forward_flow = torch.stack(rendered_flow, dim=0)
                depth = torch.stack(rendered_depths, dim=0)

        else:
            if not self.training:
                with torch.autocast("cuda", enabled=False):
                    rendered_color, rendered_alpha, _ = rasterization(
                        means=means_batched.float(),
                        quats=quats_batched.float(),
                        scales=scales_batched.float(),
                        opacities=opacities_batched.float(),
                        colors=(
                            colors_batched[..., : -(1 + self.num_bbox)].float()  # Exclude motion_seg AND lifespan
                            if True and render_motion_seg
                            else colors_batched.float()
                        ),
                        viewmats=viewmats_batched,
                        Ks=Ks_batched,
                        width=tgt_w,
                        height=tgt_h,
                        render_mode="RGB+ED",
                        near_plane=self.near,
                        far_plane=self.far,
                        packed=False,
                        radius_clip=radius_clip,
                    )
                    color, forward_flow, rendered_lifespan, depth = rendered_color.split([self.gs_dim, 3, 1, 1], dim=-1)
                    if True and render_motion_seg:
                        chunksize = 32
                        assignment_map = []
                        rendered_colors = colors_batched[..., -(1 + self.num_bbox) : -1]  # Get motion_seg but exclude lifespan (last channel)
                        for i in range(0, 1 + self.num_bbox, chunksize):
                            weights, _, _ = rasterization(
                                means=means_batched.float(),
                                quats=quats_batched.float(),
                                scales=scales_batched.float(),
                                opacities=opacities_batched.float(),
                                colors=rendered_colors[..., i : i + chunksize],
                                viewmats=viewmats_batched,
                                Ks=Ks_batched,
                                width=tgt_w,
                                height=tgt_h,
                                render_mode="RGB+ED",
                                near_plane=self.near,
                                far_plane=self.far,
                                packed=False,
                                radius_clip=radius_clip,
                            )
                            weights = weights.split([weights.size(-1) - 1, 1], dim=-1)[0]
                            assignment_map.append(weights)
                        motion_seg = torch.cat(assignment_map, dim=-1)


                        motion_seg = motion_seg.reshape(b, tgt_t, tgt_v, tgt_h, tgt_w, -1).argmax(
                            dim=-1
                        )
            else:
                with torch.autocast("cuda", enabled=False):
                    rendered_color, rendered_alpha, _ = rasterization(
                        means=means_batched.float(),
                        quats=quats_batched.float(),
                        scales=scales_batched.float(),
                        opacities=opacities_batched.float(),
                        colors=colors_batched.float(),
                        viewmats=viewmats_batched,
                        Ks=Ks_batched,
                        width=tgt_w,
                        height=tgt_h,
                        render_mode="RGB+ED",
                        near_plane=self.near,
                        far_plane=self.far,
                        packed=False,
                        radius_clip=radius_clip,
                    )
                color, forward_flow, depth = rendered_color.split([self.gs_dim, 3, 1], dim=-1)
        output_dict = {
            "rendered_image": color.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "rendered_depth": depth.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
            "rendered_alpha": rendered_alpha.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
            "rendered_flow": forward_flow.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "means_batched": means_batched,
        }
        if motion_seg is not None:
            output_dict["rendered_motion_seg"] = motion_seg.squeeze(-1)
        # Add rendered lifespan when not training
        if not self.training and 'rendered_lifespan' in locals():
            output_dict["rendered_lifespan"] = rendered_lifespan.view(b, tgt_t, tgt_v, tgt_h, tgt_w)
        return output_dict

    def get_ray_dict(self, data_dict):
        ray_dict = self.plucker_embedder(
            data_dict["context_intrinsics"],
            data_dict["context_camtoworlds"],
            image_size=data_dict["context_image"].shape[-2:],
        )

        ray_dict["target"] = self.plucker_embedder(
            data_dict["target_intrinsics"],
            data_dict["target_camtoworlds"],
            image_size=data_dict["context_image"].shape[-2:],
        )
        return data_dict, ray_dict

    def forward_1_feature(self, data_dict):

        context_imgs = data_dict["context_image"]
        b, t, v, c, h, w = context_imgs.size()


        with torch.autocast("cuda", enabled=False):
            bbox_embeds = corners_to_params(data_dict['context_instances_corner_local'])
            bbox_embeds = torch.cat([bbox_embeds, data_dict['context_instances_corner_local'].reshape(b, t, self.num_bbox, 24)], dim=-1)
            bbox_embeds = bbox_embeds.reshape(b, -1, bbox_embeds.shape[-1])

        bbox_feature = self.bbox_embed(bbox_embeds)
        bbox_time = data_dict['context_time'][:, :, 0].flatten() # (b x t)
        bbox_time_embedding = self.bbox_time_embed(self.time_embedder(bbox_time))

        bbox_time_embedding = rearrange(bbox_time_embedding, "(b t) c -> b t 1 c", b=b)
        bbox_feature = rearrange(bbox_feature, "b (t k) c -> b t k c", t=t)
        bbox_feature = bbox_feature + bbox_time_embedding
        bbox_feature = rearrange(bbox_feature, "b t k c -> b (t k) c")

        bbox_feature = torch.cat([self.background_token.expand(b, -1, -1), bbox_feature], dim=1)

        self.num_bbox_feature = bbox_feature.shape[1]
        # get plucker embedding for pixels
        data_dict, ray_dict = self.get_ray_dict(data_dict)

        data_dict["gs_time"] = data_dict["context_time"]
        # re-initailize mis feature every time
        mis_feature = self.init_feature_mis(batch_size=b)
        img_feature = self.init_feature_image(context_imgs, ray_dict["plucker"], data_dict["context_time"]) # [B, T_context, N_cam, 3, H, W] -> [B, L, D]

        if 'posterior_gs_origins' in data_dict:
            posterior_gs_feature = self.init_feature_from_patch(
                data_dict['posterior_image'],
                data_dict['posterior_gs_origins'],
                data_dict['posterior_gs_dirs'],
                data_dict['posterior_gs_time'],
                setnum=False,
            )

            num_posterior_features = posterior_gs_feature.shape[1]

            x_feature = torch.cat([bbox_feature, mis_feature, img_feature, posterior_gs_feature], dim=1)

            x = self.forward_features(x_feature)
            bbox_feature, mis_feature, gs_feature, updated_posterior = x.split([self.num_bbox_feature, self.num_mis_feature, self.num_image_feature, num_posterior_features], dim=1)
            data_dict['updated_posterior'] = updated_posterior
            # data_dict['updated_posterior'] = self.adaptor_posterior_output(updated_posterior) + data_dict['posterior_gs']

        else:
            x_feature = torch.cat([bbox_feature, mis_feature, img_feature], dim=1)
            x = self.forward_features(x_feature)
            bbox_feature, mis_feature, gs_feature = x.split([self.num_bbox_feature, self.num_mis_feature, self.num_image_feature], dim=1)



        # update state
        data_dict["gs_time"] = data_dict["context_time"]
        data_dict['gs_origins'] = ray_dict["origins"]
        data_dict['gs_dirs'] = ray_dict["dirs"]
        data_dict['img_feature'] = img_feature
        # data_dict['gs_c2w_global'] = data_dict['context_camtoworlds_global']
        data_dict['mis_state'] = mis_feature
        data_dict['gs_state'] = gs_feature
        data_dict['bbox_feature'] = bbox_feature


        sky_token, affine_tokens, motion_tokens = None, None, None
        if self.use_sky_token:
            sky_token = x[:, :1]
            x = x[:, 1:]

        if self.use_affine_token:
            affine_tokens = x[:, : self.num_cams]
            x = x[:, self.num_cams :]

        if self.num_motion_tokens > 0:
            motion_tokens = x[:, : self.num_motion_tokens]
            x = x[:, self.num_motion_tokens :]
        
        data_dict['sky_token'] = sky_token
        data_dict['affine_tokens'] = affine_tokens
        data_dict['motion_tokens'] = motion_tokens
        return data_dict

    def forward(self, data_dict, stage="all", motion=True):
        assert stage in ["all", 1, 2, 3]
        if stage == 1:
            return self.forward_1_feature(data_dict)
        elif stage == 2:
            return self.forward_2_gs(data_dict, motion=motion)
        elif stage == 3:
            return self.forward_3_render(data_dict)
        else:
            raise Exception("unexpected")

    def forward_2_gs(self, data_dict, motion=True):
        x = data_dict['gs_state']
        # bbox_tokens = data_dict['bbox_feature']

        gs_params = self.forward_gs_predictor(x, data_dict['gs_origins'], data_dict['gs_dirs'])

        # if self.num_motion_tokens > 0 and not self.static:
        if motion:
            data_dict = self.forward_motion_predictor_bbox(data_dict, gs_params)
        else:
            assert 'bbox_weights' in data_dict

        
        data_dict['gs_params'] = gs_params

        if self.num_memory_tokens > 0 and self.enable_mem_gs:
            mem_gs_params = self.forward_memory_gs(data_dict['mis_state'][:, -self.num_memory_tokens:])
            data_dict['gs_params']['mem_gs_params'] = mem_gs_params
        return data_dict

    def forward_3_export(self, data_dict, export_name):
        gs_params = data_dict['gs_params']
        self.forward_renderer(gs_params, data_dict, visualize_only=True, save_ply_name=export_name)

    def forward_3_render(self, data_dict):
        gs_params = data_dict['gs_params']
        # data_dict is target
        # render gs params
        # sometimes the number of views is too large, so we split the rendering into chunks
        step = 20
        if data_dict["target_camtoworlds"].shape[1] <= step:
            render_results = self.forward_renderer(gs_params, data_dict)
        else:
            chunk_data_dict = data_dict.copy()
            for chunk_start in range(0, data_dict["target_camtoworlds"].shape[1], step):
                chunk_end = min(chunk_start + step, data_dict["target_camtoworlds_global"].shape[1])
                chunk_data_dict["target_camtoworlds_global"] = data_dict["target_camtoworlds_global"][
                    :, chunk_start:chunk_end
                ]
                chunk_data_dict["target_intrinsics"] = data_dict["target_intrinsics"][
                    :, chunk_start:chunk_end
                ]
                chunk_data_dict["target_time"] = data_dict["target_time"][:, chunk_start:chunk_end]
                chunk_data_dict["target_instances_pose"] = data_dict['target_instances_pose'][:, chunk_start:chunk_end]
                chunk_render_results = self.forward_renderer(gs_params, chunk_data_dict)
                if chunk_start == 0:
                    render_results = chunk_render_results
                else:
                    for k, v in chunk_render_results.items():
                        render_results[k] = torch.cat([render_results[k], v], dim=1)
        images, opacities = render_results["rendered_image"], render_results["rendered_alpha"]
        if self.use_sky_token:
            sky_token = data_dict['sky_token']
            target_ray_dict = self.plucker_embedder(
                data_dict["target_intrinsics"],
                data_dict["target_camtoworlds"],
                image_size=(data_dict["height"], data_dict["width"]),
            )
            if data_dict["target_camtoworlds"].shape[1] <= step:
                sky = self.sky_head(target_ray_dict["dirs"], sky_token)
                images = images + (1 - opacities[..., None]) * sky
            else:
                for chunk_start in range(0, data_dict["target_camtoworlds"].shape[1], step):
                    dirs = target_ray_dict["dirs"][:, chunk_start : chunk_start + step]
                    chunk_sky = self.sky_head(dirs, sky_token)
                    images[:, chunk_start : chunk_start + step] += (
                        1 - opacities[:, chunk_start : chunk_start + step][..., None]
                    ) * chunk_sky
            data_dict["gs_params"]["sky_token"] = sky_token

        if self.use_affine_token:
            affine_tokens = data_dict['affine_tokens']
            affine = self.affine_linear(affine_tokens)  # b v (gs_dim * (gs_dim + 1))
            affine = rearrange(affine, "b v (p q) -> b v p q", p=self.gs_dim)
            images = torch.einsum("b t v h w p, b v p q -> b t v h w p", images, affine) # affine 居然是负的，无语了
            data_dict["gs_params"]["affine"] = affine
        else:
            images = -images
        render_results["rendered_image"] = images

        # currently dummy decoder
        render_results = self.forward_decoder(render_results)
        data_dict['render_results'] = render_results
        return data_dict

def UFO_B_8(**kwargs):
    return UFO(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)


def UFO_L_8(**kwargs):
    return UFO(patch_size=8, embed_dim=1024, depth=24, num_heads=16, **kwargs)


def UFO_B_16(**kwargs):
    return UFO(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)


def UFO_L_16(**kwargs):
    return UFO(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)


def UFO_XL_8(**kwargs):
    return UFO(patch_size=8, embed_dim=1152, depth=28, num_heads=16, **kwargs)


def UFO_H_8(**kwargs):
    return UFO(patch_size=8, embed_dim=1280, depth=32, num_heads=16, **kwargs)


def UFO_H_16(**kwargs):
    return UFO(patch_size=16, embed_dim=1280, depth=32, num_heads=16, **kwargs)


UFO_models = {
    "UFO-B/8": UFO_B_8,
    "UFO-L/8": UFO_L_8,
    "UFO-XL/8": UFO_XL_8,
    "UFO-H/8": UFO_H_8,
    "UFO-B/16": UFO_B_16,
    "UFO-L/16": UFO_L_16,
    "UFO-H/16": UFO_H_16,
}
