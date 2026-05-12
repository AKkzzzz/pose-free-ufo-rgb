# Copyright (c) Xiaomi Corporation.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT:   https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# --------------------------------------------------------

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


# ============================================================================
# LayerNorm2d — channel-first LayerNorm.
# UFO-original: a thin wrapper around ``F.layer_norm`` that internally permutes
# (N, C, H, W) → (N, H, W, C) so the normalization happens over C, then permutes
# back. Functionally equivalent to a manual mean/var implementation.
# ============================================================================
class LayerNorm2d(nn.Module):
    """LayerNorm applied along the channel dimension of a 4D ``(N, C, H, W)`` tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.normalized_shape = (num_channels,)

    def forward(self, x: Tensor) -> Tensor:
        # F.layer_norm normalizes over the last dim(s) — move channels there.
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


# ============================================================================
# modulate — adapted from DiT/models.py::modulate.
# AdaLN feature modulation: ``y = x * (1 + scale) + shift``.
# ============================================================================
def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================================
# TimestepEmbedder — adapted from DiT/models.py::TimestepEmbedder.
# Embeds a scalar continuous timestep into a vector via a sinusoidal frequency
# encoding followed by a 2-layer MLP with SiLU activation. The frequency formula
# itself is from OpenAI GLIDE (MIT-licensed) and is reproduced here verbatim.
# ============================================================================
class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """Sinusoidal timestep embeddings.

        :param t: a 1-D Tensor of N indices, one per batch element. May be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ============================================================================
# ModulatedLinearLayer — UFO-original.
#
# A small conditional output head that applies AdaLN modulation between an
# input projection and an output projection. Unlike DiT's FinalLayer (which is
# norm → modulate → linear), this module is:
#
#     linear → norm → modulate(shift, scale) → output_linear
#
# and additionally projects the conditioning vector ``c`` through its own
# linear layer ``condition_mapping`` so ``c`` and the embedding can have
# different feature widths (UFO uses this to condition a small MLP on the
# global ``sky_token``, where the token width and the embedding width differ).
#
# The five sub-modules and their state-dict keys are stable across the original
# UFO checkpoints — only the forward expression has been simplified here.
# ============================================================================
class ModulatedLinearLayer(nn.Module):
    """Two-layer conditional MLP with adaLN modulation between the layers."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        condition_channels: int = 768,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True),
        )
        self.condition_mapping = nn.Linear(condition_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        h = self.linear(x)                                            # (..., hidden)
        c_proj = self.condition_mapping(c.squeeze(1))                 # (B, hidden)
        shift, scale = self.adaLN_modulation(c_proj).chunk(2, dim=-1)

        # The input may have arbitrary leading dims; modulate operates on a
        # 3-D tensor (B, L, C). Flatten everything between the batch and the
        # feature axis, modulate, then restore the original leading shape.
        original_shape = h.shape
        h = h.reshape(original_shape[0], -1, original_shape[-1])
        h = modulate(self.norm(h), shift, scale)
        return self.output(h).reshape(*original_shape[:-1], -1)


# ============================================================================
# PluckerEmbedder — UFO-original.
#
# Computes Plücker-coordinate ray embeddings from camera intrinsics and
# extrinsics (or directly from explicit ray bundles via ``forward_from_rays``).
# The math is the standard inverse-pinhole projection composed with the
# camera-to-world rotation; no copyrightable expression is borrowed from any
# upstream codebase.
#
# Convention note: ``forward`` samples each pixel at center + 0.5 pixels in
# both axes (this matches the convention used by the rest of UFO downstream;
# changing it would require re-training).
# ============================================================================
class PluckerEmbedder(nn.Module):
    """Convert camera (intrinsics, c2w) — or explicit ray bundles — into
    Plücker coordinates ``[origin × direction, direction]``.
    """

    def __init__(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = 224,
        patch_size: int = 1,
    ) -> None:
        super().__init__()
        self.patch_size = _to_2tuple(patch_size)
        self.img_size = _to_2tuple(img_size)
        self.grid_size = tuple(s // p for s, p in zip(self.img_size, self.patch_size))

        # Cache the default-resolution pixel-center grid. Each cell sits at the
        # center of its patch (hence the +0.5). ``_x`` and ``_y`` are flat
        # ``(1, H'*W')`` row-major buffers consumed by ``forward``.
        ys_idx, xs_idx = torch.meshgrid(
            torch.arange(self.grid_size[0], dtype=torch.float32),
            torch.arange(self.grid_size[1], dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("x", (xs_idx.reshape(1, -1) + 0.5))
        self.register_buffer("y", (ys_idx.reshape(1, -1) + 0.5))

    @staticmethod
    def _scale_intrinsics_to_grid(K: Tensor, p_h: int, p_w: int) -> Tensor:
        """Rescale ``K`` so it maps into a grid that has been downsampled by
        ``(p_h, p_w)`` relative to the resolution K was authored for."""
        K = K.clone()
        K[:, 0, 0] = K[:, 0, 0] / p_w
        K[:, 0, 2] = K[:, 0, 2] / p_w
        K[:, 1, 1] = K[:, 1, 1] / p_h
        K[:, 1, 2] = K[:, 1, 2] / p_h
        return K

    def forward(
        self,
        intrinsics: Tensor,
        camtoworlds: Tensor,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> dict:
        """Project a pixel grid into world rays and emit their Plücker form.

        Returns a dict with keys ``origins``, ``viewdirs``, ``dirs``, ``plucker``,
        all shaped ``(*lead, H', W', 3 or 6)`` where ``lead`` is the leading
        batch shape of ``intrinsics``.
        """
        assert intrinsics.shape[-2:] == (3, 3), "intrinsics must be (..., 3, 3)"
        assert camtoworlds.shape[-2:] == (4, 4), "camtoworlds must be (..., 4, 4)"

        lead_shape = intrinsics.shape[:-2]
        K = intrinsics.reshape(-1, 3, 3)
        C = camtoworlds.reshape(-1, 4, 4)

        img_size = _to_2tuple(image_size) if image_size is not None else self.img_size
        ps = _to_2tuple(patch_size) if patch_size is not None else self.patch_size
        H, W = img_size[0] // ps[0], img_size[1] // ps[1]

        # Pixel-center grid + matching intrinsics. If the requested grid matches
        # the cached one, reuse the buffer; otherwise rebuild and rescale K.
        if (H, W) == self.grid_size:
            xs, ys = self.x, self.y
            K_eff = K
        else:
            ys_idx, xs_idx = torch.meshgrid(
                torch.arange(H, device=K.device, dtype=torch.float32),
                torch.arange(W, device=K.device, dtype=torch.float32),
                indexing="ij",
            )
            xs = xs_idx.reshape(1, -1) + 0.5
            ys = ys_idx.reshape(1, -1) + 0.5
            K_eff = self._scale_intrinsics_to_grid(K, ps[0], ps[1])

        # Inverse-pinhole projection: pixel (u, v) → camera-frame direction.
        # The "+0.5" in the numerator matches the historical UFO sampling
        # convention; preserving it keeps existing checkpoints valid.
        fx = K_eff[:, 0, 0].unsqueeze(-1)
        fy = K_eff[:, 1, 1].unsqueeze(-1)
        cx = K_eff[:, 0, 2].unsqueeze(-1)
        cy = K_eff[:, 1, 2].unsqueeze(-1)
        d_cam = torch.stack(
            [
                (xs - cx + 0.5) / fx,
                (ys - cy + 0.5) / fy,
                torch.ones_like(xs.expand(K_eff.size(0), -1)),
            ],
            dim=-1,
        )  # (B, H*W, 3)

        # Camera frame → world frame.
        R = C[:, :3, :3]
        t = C[:, :3, 3].unsqueeze(1)  # (B, 1, 3)
        d_world = torch.einsum("bij,bnj->bni", R, d_cam)
        origins = t.expand_as(d_world)

        # Plücker = [origin × unit_dir, unit_dir].
        d_unit = d_world / (torch.linalg.norm(d_world, dim=-1, keepdim=True) + 1e-8)
        moment = torch.cross(origins, d_unit, dim=-1)
        plucker = torch.cat([moment, d_unit], dim=-1)

        out_shape3 = (*lead_shape, H, W, 3)
        out_shape6 = (*lead_shape, H, W, 6)
        return {
            "origins":  origins.reshape(out_shape3),
            "viewdirs": d_unit.reshape(out_shape3),
            "dirs":     d_world.reshape(out_shape3),
            "plucker":  plucker.reshape(out_shape6),
        }

    def forward_from_rays(
        self,
        rays_origins: Tensor,
        rays_directions: Tensor,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> dict:
        """Compute Plücker embedding directly from ray origins and directions.

        Used by the recurrent posterior path, where each persisted token already
        carries its own ray bundle (potentially in a corrected frame after a
        dynamic-object pose update).
        """
        assert rays_origins.shape[-1] == 3, "rays_origins should have last dim 3"
        assert rays_directions.shape[-1] == 3, "rays_directions should have last dim 3"
        assert rays_origins.shape[:-1] == rays_directions.shape[:-1], "ray origins/dirs shape mismatch"

        original_shape = rays_origins.shape[:-1]
        batch_dims = original_shape[:-2] if len(original_shape) > 2 else ()

        if image_size is not None:
            image_size = _to_2tuple(image_size)
        elif len(original_shape) >= 2:
            image_size = (original_shape[-2], original_shape[-1])
        else:
            image_size = self.img_size

        ps = _to_2tuple(patch_size) if patch_size is not None else self.patch_size
        grid_size = tuple(s // p for s, p in zip(image_size, ps))

        flat_shape = (*batch_dims, -1, 3) if batch_dims else (-1, 3)
        origins_flat = rays_origins.reshape(flat_shape)
        directions_flat = rays_directions.reshape(flat_shape)
        viewdirs_flat = directions_flat / (
            torch.linalg.norm(directions_flat, dim=-1, keepdim=True) + 1e-8
        )
        cross_prod = torch.cross(origins_flat, viewdirs_flat, dim=-1)
        plucker_flat = torch.cat((cross_prod, viewdirs_flat), dim=-1)

        if len(original_shape) >= 2:
            target_shape = (*batch_dims, *grid_size, 3) if batch_dims else (*grid_size, 3)
            return {
                "origins":  origins_flat.reshape(target_shape),
                "viewdirs": viewdirs_flat.reshape(target_shape),
                "dirs":     directions_flat.reshape(target_shape),
                "plucker":  plucker_flat.reshape((*target_shape[:-1], 6)),
            }
        return {
            "origins":  origins_flat,
            "viewdirs": viewdirs_flat,
            "dirs":     directions_flat,
            "plucker":  plucker_flat,
        }


# ============================================================================
# Decoder utilities — UFO-original.
# ============================================================================
def check_results(result_dict: dict) -> bool:
    """Validate that a render-result dict carries the keys downstream code expects."""
    required = (
        "rgb_key",
        "depth_key",
        "alpha_key",
        "flow_key",
        "decoder_depth_key",
        "decoder_alpha_key",
        "decoder_flow_key",
    )
    for key in required:
        assert key in result_dict, f"{key} not found in result_dict"
    return True


class DummyDecoder(nn.Module):
    """No-op decoder: validates the rendered-results dict and returns it unchanged.

    Used when UFO renders directly in pixel space (``decoder_type == 'dummy'``).
    The ``**kwargs`` is preserved so this constructor stays signature-compatible
    with potential future decoder variants.
    """

    def __init__(self, **kwargs) -> None:  # noqa: D401 - kwargs intentionally unused
        super().__init__()

    def forward(self, render_results: dict) -> dict:
        if not check_results(render_results):
            raise ValueError("Invalid result dict")
        return render_results
