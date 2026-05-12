# Copyright (c) Xiaomi Corporation.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/huggingface/pytorch-image-models
# --------------------------------------------------------

import logging
import math
from functools import partial
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


logger = logging.getLogger("UFO")


def _to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


# ----------------------------------------------------------------------------
# MLP — adapted from timm/layers/mlp.py::Mlp
# ----------------------------------------------------------------------------
class Mlp(nn.Module):
    """Two-layer MLP as used in Vision Transformer / MLP-Mixer.

    Trimmed version of timm's ``Mlp``: drops the optional ``norm`` mid-layer,
    the configurable bias tuple, the use_conv path, and the dropouts (UFO
    never enables them). Behaviour matches timm's ``Mlp`` invoked with default
    bias=True, drop=0., norm_layer=None.

    The historical UFO call style ``Mlp(in_dim, hidden_dim, out_dim)`` continues
    to work because the first three positional arguments are unchanged.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ----------------------------------------------------------------------------
# Attention — adapted from timm/layers/attention.py::Attention
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    """Standard multi-head self-attention with QKV projection and SDPA.

    Adapted from timm's ``Attention`` with the following simplifications:
      * ``qkv_bias`` defaults to ``True`` (UFO/STORM checkpoints always had it).
      * No attention or projection dropout.
      * No fused/manual fallback — relies on torch's
        ``F.scaled_dot_product_attention`` (torch >= 2.0).
      * No optional output norm, no rope, no attn_mask.
    """

    _logged = False

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim must be divisible by num_heads, got {dim} and {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        if not Attention._logged:
            Attention._logged = True
            logger.info(f"[Attention]: Using {torch.__version__} F.scaled_dot_product_attention")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# ----------------------------------------------------------------------------
# Block — adapted from timm/models/vision_transformer.py::Block (pre-norm)
# ----------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm transformer block: x = x + Attn(LN(x)); x = x + MLP(LN(x))."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# ----------------------------------------------------------------------------
# Transformer — small wrapper around an ``nn.ModuleList`` of blocks.
# Kept to preserve state-dict keys ``transformer.blocks.X.*`` from earlier
# UFO checkpoints. Equivalent to looping over ``self.blocks`` directly.
# ----------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        norm_layer: Type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.grad_checkpointing = grad_checkpointing
        logger.info(f"[Transformer]: grad_checkpointing={grad_checkpointing}")

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# ----------------------------------------------------------------------------
# PatchEmbed — adapted from timm/layers/patch_embed.py::PatchEmbed
# ----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """2D image to patch embedding via a ``stride=patch_size`` Conv2d.

    Trimmed version of timm's ``PatchEmbed``: removes Format/torch.fx machinery
    and dynamic padding (UFO inputs are always sized to a multiple of patch
    size). The single output layout supported is **NHWC**, which is what the
    rest of UFO consumes; the flatten path is dropped.
    """

    def __init__(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        output_fmt: str = "NHWC",
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert output_fmt == "NHWC", "UFO only uses the NHWC output format from PatchEmbed."
        self.patch_size = _to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = _to_2tuple(img_size)
            self.grid_size = tuple(s // p for s, p in zip(self.img_size, self.patch_size))
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)                          # (B, C, H', W')
        return rearrange(x, "B C H W -> B H W C")


# ----------------------------------------------------------------------------
# resample_abs_pos_embed — adapted from timm/layers/pos_embed.py
# ----------------------------------------------------------------------------
def resample_abs_pos_embed(
    posemb: Tensor,
    new_size: List[int],
    old_size: Optional[List[int]] = None,
    n_prefix_tokens: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
    verbose: bool = False,
) -> Tensor:
    """Resample a learned 2D absolute positional embedding to a new grid.

    Functional copy of timm's ``resample_abs_pos_embed``. Difference: the
    keyword is ``n_prefix_tokens`` (matching the legacy UFO/STORM call sites)
    instead of timm's ``num_prefix_tokens``.
    """
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + n_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - n_prefix_tokens))
        old_size = [hw, hw]

    if n_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :n_prefix_tokens], posemb[:, n_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        logger.info(f"Resized position embedding: {old_size} to {new_size}.")

    return posemb


# ----------------------------------------------------------------------------
# VisionTransformer — minimal UFO-targeted ViT.
# Heavily trimmed compared to timm's ``VisionTransformer``: no class/register
# tokens, no classifier, no token droppage, no torchscript guards. Only what
# UFO actually exercises.
# ----------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    """ViT backbone used as the parent class of UFO."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        norm_layer: Type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            output_fmt="NHWC",
        )
        self.num_patches = self.patch_embed.num_patches
        self.img_size = self.patch_embed.img_size

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.transformer = Transformer(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            grad_checkpointing=grad_checkpointing,
        )
        self.norm = norm_layer(embed_dim)
        self.init_weights()

    def init_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

    def _pos_embed(self, x: Tensor) -> Tensor:
        """Add a learned absolute positional embedding, resampled if input
        resolution differs from the embedding's training-time grid size.
        """
        B, H, W, C = x.shape
        pos_embed = resample_abs_pos_embed(
            posemb=self.pos_embed,
            new_size=(H, W),
            old_size=self.patch_embed.grid_size,
            n_prefix_tokens=0,
        )
        return x.view(B, -1, C) + pos_embed

    def unpatchify(
        self,
        x: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        channel_first: bool = True,
        patch_size: Optional[int] = None,
    ) -> Tensor:
        """Inverse of patch embedding: ``(B, H*W, p^2 * C) -> (B, C, H*p, W*p)``."""
        hw = hw or self.img_size
        p = self.patch_size if patch_size is None else patch_size
        imgs = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=p,
            p2=p,
            h=hw[0] // p,
            w=hw[1] // p,
        )
        if not channel_first:
            imgs = rearrange(imgs, "b c h w -> b h w c")
        return imgs
