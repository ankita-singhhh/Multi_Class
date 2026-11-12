
"""
Query-Visual Interaction (QVI) for YOLO-Count.

Model C:
    Model B + query-conditioned visual interaction.

Input:
    vis: [B, C_in, H, W]
    txt: [B, K, D_txt]

Output:
    d_den: [B, K, H, W] log-density corrections
    d_cls: [B, K, H, W] classification-logit corrections

Design:
    1. Reduce visual channels.
    2. Generate query-conditioned modulation parameters.
    3. Modulate visual features for each query.
    4. Apply nonlinear spatial refinement.
    5. Predict zero-initialized residual corrections.

Important:
    d_den is a log-density correction, not an additive density value.

    Integrate it as:
        density_C = density_B * exp(clamp(d_den))

    This preserves non-negative density predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QVI(nn.Module):
    """
    Query-Visual Interaction module.

    Args:
        c_in:
            Input visual embedding dimension.

        d_txt:
            Text embedding dimension.

        c:
            Reduced interaction channels.

        mode:
            "modulation", "gated", or "none".

        query_chunk_size:
            Number of queries processed together to control memory.
            None processes all queries together.
    """

    SUPPORTED_MODES = {"modulation", "gated", "none"}

    def __init__(
        self,
        c_in: int = 512,
        d_txt: int = 512,
        c: int = 64,
        mode: str = "modulation",
        query_chunk_size: int = 2,
    ):
        super().__init__()

        if min(c_in, d_txt, c) <= 0:
            raise ValueError("QVI dimensions must be positive.")

        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported QVI mode: {mode}. "
                f"Supported modes: {sorted(self.SUPPORTED_MODES)}. "
                "Attention requires a separately implemented "
                "and validated interaction architecture."
            )

        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive or None.")

        self.mode = mode
        self.c_in = c_in
        self.d_txt = d_txt
        self.channels = c
        self.query_chunk_size = query_chunk_size

        # -----------------------------------------------------
        # VISUAL CHANNEL REDUCTION
        # -----------------------------------------------------
        self.reduce = nn.Sequential(
            nn.Conv2d(c_in, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        # -----------------------------------------------------
        # TEXT-CONDITIONED MODULATION
        # -----------------------------------------------------
        output_channels = c if mode == "gated" else 2 * c

        self.mlp = nn.Sequential(
            nn.Linear(d_txt, d_txt // 2),
            nn.GELU(),
            nn.Linear(d_txt // 2, output_channels),
        )

        # -----------------------------------------------------
        # NONLINEAR VISUAL REFINEMENT
        # -----------------------------------------------------
        self.refine = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )

        # -----------------------------------------------------
        # RESIDUAL OUTPUT PROJECTIONS
        # -----------------------------------------------------
        self.out_den = nn.Conv2d(c, 1, kernel_size=1)
        self.out_cls = nn.Conv2d(c, 1, kernel_size=1)

        # Preserve Model B exactly at initialization.
        nn.init.zeros_(self.out_den.weight)
        nn.init.zeros_(self.out_den.bias)

        nn.init.zeros_(self.out_cls.weight)
        nn.init.zeros_(self.out_cls.bias)

    def _process_query_chunk(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
    ):
        """
        Process a subset of queries.

        visual: [B, C, H, W]
        text:   [B, K_chunk, D]
        """

        batch_size, channels, height, width = visual.shape
        num_queries = text.shape[1]

        if self.mode == "gated":
            # Query-conditioned multiplicative gating.
            gate = torch.sigmoid(self.mlp(text))

            conditioned = (
                visual.unsqueeze(1)
                * gate.unsqueeze(-1).unsqueeze(-1)
            )

        else:
            # Query-conditioned affine modulation.
            gamma, beta = self.mlp(text).chunk(2, dim=-1)

            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)

            conditioned = (
                visual.unsqueeze(1) * (1.0 + gamma) + beta
            )

        # [B, K_chunk, C, H, W] -> [B*K_chunk, C, H, W]
        conditioned = conditioned.reshape(
            batch_size * num_queries,
            channels,
            height,
            width,
        )

        # Nonlinear refinement prevents a purely linear
        # reparameterization of the original similarity.
        refined = conditioned + self.refine(conditioned)

        # Predict query-specific corrections.
        d_den = self.out_den(refined)
        d_cls = self.out_cls(refined)

        d_den = d_den.reshape(
            batch_size,
            num_queries,
            height,
            width,
        )

        d_cls = d_cls.reshape(
            batch_size,
            num_queries,
            height,
            width,
        )

        return d_den, d_cls

    def forward(
        self,
        vis: torch.Tensor,
        txt: torch.Tensor,
    ):
        """
        Generate query-specific visual interaction corrections.

        Args:
            vis: [B, C_in, H, W]
            txt: [B, K, D_txt]

        Returns:
            d_den: [B, K, H, W]
            d_cls: [B, K, H, W]
        """

        if vis.ndim != 4:
            raise ValueError(
                f"Expected visual tensor [B,C,H,W], "
                f"got {tuple(vis.shape)}."
            )

        if txt.ndim != 3:
            raise ValueError(
                f"Expected text tensor [B,K,D], "
                f"got {tuple(txt.shape)}."
            )

        batch_size, channels, height, width = vis.shape
        txt_batch, num_queries, text_dim = txt.shape

        if batch_size != txt_batch:
            raise ValueError("Visual and text batch sizes do not match.")

        if channels != self.c_in:
            raise ValueError(
                f"Expected {self.c_in} visual channels, got {channels}."
            )

        if text_dim != self.d_txt:
            raise ValueError(
                f"Expected text dimension {self.d_txt}, got {text_dim}."
            )

        if num_queries == 0:
            raise ValueError("At least one text query is required.")

        # -----------------------------------------------------
        # NO-INTERACTION BASELINE
        # -----------------------------------------------------
        if self.mode == "none":
            zero = vis.new_zeros(
                batch_size,
                num_queries,
                height,
                width,
            )
            return zero, zero.clone()

        # -----------------------------------------------------
        # REDUCE VISUAL CHANNELS ONCE
        # -----------------------------------------------------
        visual = self.reduce(vis)

        # -----------------------------------------------------
        # QUERY CHUNKING FOR MEMORY CONTROL
        # -----------------------------------------------------
        chunk_size = self.query_chunk_size or num_queries

        density_chunks = []
        classification_chunks = []

        for start in range(0, num_queries, chunk_size):

            end = min(start + chunk_size, num_queries)

            text_chunk = txt[:, start:end, :]

            d_den, d_cls = self._process_query_chunk(
                visual,
                text_chunk,
            )

            density_chunks.append(d_den)
            classification_chunks.append(d_cls)

        d_den = torch.cat(density_chunks, dim=1)
        d_cls = torch.cat(classification_chunks, dim=1)

        return d_den, d_cls
