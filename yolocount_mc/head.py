
"""
Multi-class density and classification prediction head.

Model B:
    Shared visual features + class-conditioned prediction.

Input:
    feat: [B, C, H, W]
    txt:  [B, K, D]

Output:
    density_map: [B, K, H, W]  Non-negative density predictions
    cls_logits:  [B, K, H, W]  Classification logits
    vis_den:     [B, D, H, W]  Visual embedding for QVI

Important:
    The visual feature map is processed once.
    The class dimension is introduced only through einsum.
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

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class MultiClassHead(nn.Module):
    """
    Shared visual representation with class-conditioned counting.

    Args:
        c_in:
            Input backbone channels.

        c_hidden:
            Hidden feature channels.

        d:
            Shared visual-text embedding dimension.

        initial_density_scale:
            Initial multiplier for the positive density predictions.
            This is a numerical initialization, not a fixed count scale.
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

        if initial_density_scale <= 0:
            raise ValueError("initial_density_scale must be positive.")

        # -----------------------------------------------------
        # SHARED SPATIAL FEATURE EXTRACTOR
        # -----------------------------------------------------
        self.density_head_new = nn.Sequential(
            ConvBlock(c_in, c_hidden),
            ConvBlock(c_hidden, c_hidden),
        )

        # -----------------------------------------------------
        # VISUAL EMBEDDING PROJECTIONS
        # -----------------------------------------------------
        self.density_proj = nn.Conv2d(
            c_hidden,
            d,
            kernel_size=1,
        )

        self.cls_proj = nn.Conv2d(
            c_hidden,
            d,
            kernel_size=1,
        )

        # -----------------------------------------------------
        # LEARNABLE SIMILARITY TEMPERATURE
        # -----------------------------------------------------
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0)))

        # -----------------------------------------------------
        # POSITIVE DENSITY CALIBRATION
        # -----------------------------------------------------
        # The density branch predicts non-negative values.
        # A learnable scale allows the model to calibrate
        # density magnitude through count supervision.
        self.log_density_scale = nn.Parameter(
            torch.tensor(math.log(initial_density_scale))
        )

        self.output_dim = d

    def forward(self, feat: torch.Tensor, txt: torch.Tensor):
        """
        Compute class-conditioned density and classification maps.

        Args:
            feat:
                [B, C, H, W] shared visual features.

            txt:
                [B, K, D] text embeddings.

        Returns:
            density_map:
                [B, K, H, W], non-negative.

            cls_logits:
                [B, K, H, W], classification logits.

            vis_den:
                [B, D, H, W], visual embedding for QVI.
        """

        if feat.ndim != 4:
            raise ValueError(
                f"Expected feat [B,C,H,W], got {tuple(feat.shape)}"
            )

        if txt.ndim != 3:
            raise ValueError(
                f"Expected txt [B,K,D], got {tuple(txt.shape)}"
            )

        batch_size, channels, height, width = feat.shape
        txt_batch, num_queries, text_dim = txt.shape

        if batch_size != txt_batch:
            raise ValueError(
                "Image and text batch dimensions do not match."
            )

        if text_dim != self.output_dim:
            raise ValueError(
                f"Text dimension {text_dim} does not match "
                f"head dimension {self.output_dim}."
            )

        if num_queries == 0:
            raise ValueError("At least one class query is required.")

        # -----------------------------------------------------
        # 1. SHARED VISUAL FEATURE EXTRACTION
        # -----------------------------------------------------
        x = self.density_head_new(feat)

        # -----------------------------------------------------
        # 2. VISUAL EMBEDDING PROJECTIONS
        # -----------------------------------------------------
        vis_den = self.density_proj(x)
        vis_cls = self.cls_proj(x)

        # -----------------------------------------------------
        # 3. L2 NORMALIZATION
        # -----------------------------------------------------
        vis_den_n = F.normalize(vis_den, p=2, dim=1)
        vis_cls_n = F.normalize(vis_cls, p=2, dim=1)
        txt_n = F.normalize(txt, p=2, dim=-1)

        # -----------------------------------------------------
        # 4. CLASS-CONDITIONED SIMILARITY
        # -----------------------------------------------------
        similarity_scale = self.log_scale.exp().clamp(
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

        # -----------------------------------------------------
        # 5. NON-NEGATIVE DENSITY PREDICTION
        # -----------------------------------------------------
        density_scale = self.log_density_scale.exp().clamp(
            min=1e-8,
            max=100.0,
        )

        density_map = F.softplus(den_logits) * density_scale

        # -----------------------------------------------------
        # OUTPUT CONTRACT
        # -----------------------------------------------------
        return density_map, cls_logits, vis_den