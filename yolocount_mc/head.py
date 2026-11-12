
"""
Multi-class density and classification prediction head.

Input:
    feat: [B, C, H, W]
    txt:  [B, K, D]

Output:
    density_map: [B, K, H, W]
    cls_logits:  [B, K, H, W]
    vis_den:     [B, D, H, W]

The visual feature extractor runs once for all K class queries.
Class-conditioned predictions are produced using batched similarity.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel dimensions must be positive.")

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number.")

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiClassHead(nn.Module):
    """
    Shared visual representation with class-conditioned prediction.

    Args:
        c_in: Input visual channels.
        c_hidden: Hidden feature channels.
        d: Shared visual-text embedding dimension.
        initial_density_scale: Initial positive density multiplier.
    """

    def __init__(
        self,
        c_in: int = 256,
        c_hidden: int = 256,
        d: int = 512,
        initial_density_scale: float = 0.002,
    ):
        super().__init__()

        if min(c_in, c_hidden, d) <= 0:
            raise ValueError("All channel dimensions must be positive.")

        if not math.isfinite(initial_density_scale) or initial_density_scale <= 0:
            raise ValueError("initial_density_scale must be finite and positive.")

        self.c_in = c_in
        self.output_dim = d

        # Shared spatial feature extraction.
        self.density_head_new = nn.Sequential(
            ConvBlock(c_in, c_hidden),
            ConvBlock(c_hidden, c_hidden),
        )

        # Visual embeddings for density and classification branches.
        self.density_proj = nn.Conv2d(c_hidden, d, kernel_size=1)
        self.cls_proj = nn.Conv2d(c_hidden, d, kernel_size=1)

        # Learnable similarity temperature.
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0)))

        # Learnable positive density calibration.
        self.log_density_scale = nn.Parameter(
            torch.tensor(math.log(initial_density_scale))
        )

    def forward(
        self,
        feat: torch.Tensor,
        txt: torch.Tensor,
    ):
        if feat.ndim != 4:
            raise ValueError(
                f"Expected feat [B,C,H,W], got {tuple(feat.shape)}"
            )

        if txt.ndim != 3:
            raise ValueError(
                f"Expected txt [B,K,D], got {tuple(txt.shape)}"
            )

        batch_size, channels, height, width = feat.shape
        text_batch, num_queries, text_dim = txt.shape

        if channels != self.c_in:
            raise ValueError(
                f"Expected {self.c_in} visual channels, got {channels}."
            )

        if batch_size != text_batch:
            raise ValueError("Image and text batch dimensions do not match.")

        if num_queries <= 0:
            raise ValueError("At least one class query is required.")

        if text_dim != self.output_dim:
            raise ValueError(
                f"Expected text dimension {self.output_dim}, got {text_dim}."
            )

        if height <= 0 or width <= 0:
            raise ValueError("Feature-map dimensions must be positive.")

        if feat.device != txt.device:
            raise ValueError("Visual features and text embeddings must share a device.")

        if not feat.is_floating_point() or not txt.is_floating_point():
            raise TypeError("Visual features and text embeddings must be floating-point tensors.")

        # 1. Extract shared visual features once.
        x = self.density_head_new(feat)

        # 2. Generate visual embeddings.
        vis_den = self.density_proj(x)
        vis_cls = self.cls_proj(x)

        # 3. Normalize embeddings before similarity.
        vis_den_n = F.normalize(vis_den.float(), p=2, dim=1)
        vis_cls_n = F.normalize(vis_cls.float(), p=2, dim=1)
        txt_n = F.normalize(txt.float(), p=2, dim=-1)

        # 4. Class-conditioned similarity.
        similarity_scale = self.log_scale.float().exp().clamp(
            min=0.01,
            max=100.0,
        )

        den_logits = torch.einsum(
            "bchw,bkc->bkhw",
            vis_den_n,
            txt_n,
        ) * similarity_scale

        cls_logits = torch.einsum(
            "bchw,bkc->bkhw",
            vis_cls_n,
            txt_n,
        ) * similarity_scale

        # 5. Positive density prediction.
        density_scale = self.log_density_scale.float().exp().clamp(
            min=1e-8,
            max=100.0,
        )

        density_map = F.softplus(den_logits) * density_scale

        # Preserve the original visual feature tensor for QVI.
        return density_map, cls_logits, vis_den
