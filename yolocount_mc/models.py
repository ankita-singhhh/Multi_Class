
"""
Complete YOLO-Count multi-class architecture.

Model A:
    Single-query counting baseline.

Model B:
    Multi-class counting with shared visual features.

Model C:
    Model B + Query-Visual Interaction (QVI).

Common output contract:
    counts:       [B, K]
    density_maps: [B, K, H, W]
    cls_logits:   [B, K, H, W]
"""

from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn

from .config import Config
from .text_encoder import TextEncoder
from .backbone import YoloBackbonePAFPN
from .head import MultiClassHead
from .qvi import QVI


def validate_queries(
    queries: List[List[str]],
    batch_size: int,
) -> int:
    """Validate batched class-name queries."""

    if not isinstance(queries, list):
        raise TypeError("queries must be a list of query lists.")

    if len(queries) != batch_size:
        raise ValueError(
            f"Expected {batch_size} query lists, got {len(queries)}."
        )

    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    if not all(isinstance(row, list) and row for row in queries):
        raise ValueError(
            "Every image must contain at least one class query."
        )

    num_queries = len(queries[0])

    if any(len(row) != num_queries for row in queries):
        raise ValueError(
            "All images must have the same number of queries. "
            "Use the dataset collate function to pad query lists."
        )

    for row in queries:
        for query in row:
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    "Every class query must be a non-empty string."
                )

    return num_queries


def aggregate_counts(
    density_maps: torch.Tensor,
    cls_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Aggregate class-conditioned density maps into counts.

    Args:
        density_maps: [B, K, H, W], non-negative density predictions.
        cls_logits: [B, K, H, W], classification logits.

    Returns:
        counts: [B, K]
    """

    if density_maps.ndim != 4 or cls_logits.ndim != 4:
        raise ValueError(
            "Density maps and classification logits must have shape [B,K,H,W]."
        )

    if density_maps.shape != cls_logits.shape:
        raise ValueError(
            "Density maps and classification logits must have identical shapes."
        )

    if density_maps.device != cls_logits.device:
        raise ValueError("Both prediction tensors must share a device.")

    if not density_maps.is_floating_point():
        raise TypeError("Density maps must be floating-point tensors.")

    if not cls_logits.is_floating_point():
        raise TypeError("Classification logits must be floating-point tensors.")

    if not torch.isfinite(density_maps).all():
        raise ValueError("Density maps contain NaN or infinite values.")

    if not torch.isfinite(cls_logits).all():
        raise ValueError("Classification logits contain NaN or infinite values.")

    if torch.any(density_maps < 0):
        raise ValueError("Density predictions must be non-negative.")

    probabilities = torch.sigmoid(cls_logits)

    counts = (
        density_maps * probabilities
    ).sum(dim=(-2, -1))

    return counts


class YOLOCountBase(nn.Module):
    """Shared implementation for Models A, B and C."""

    def __init__(
        self,
        text_enc: TextEncoder,
        cfg: Optional[Config] = None,
    ):
        super().__init__()

        if text_enc is None:
            raise ValueError("A TextEncoder instance is required.")

        self.cfg = cfg if cfg is not None else Config()
        self.text_enc = text_enc

        self.backbone = YoloBackbonePAFPN(
            out_ch=self.cfg.backbone_channels,
            pretrained=self.cfg.pretrained_backbone,
            weights=getattr(
                self.cfg,
                "backbone_weights",
                "yolov8n.pt",
            ),
        )

        self.head = MultiClassHead(
            c_in=self.cfg.backbone_channels,
            c_hidden=self.cfg.backbone_channels,
            d=self.cfg.embed_dim,
        )

    def _extract_features_and_text(
        self,
        img: torch.Tensor,
        queries: List[List[str]],
    ):
        """Extract shared image features and text embeddings."""

        if not isinstance(img, torch.Tensor):
            raise TypeError("img must be a torch.Tensor.")

        if img.ndim != 4:
            raise ValueError(
                f"Expected image [B,C,H,W], got {tuple(img.shape)}."
            )

        if img.shape[0] == 0:
            raise ValueError("Image batch cannot be empty.")

        if img.shape[1] != 3:
            raise ValueError("Input images must have three RGB channels.")

        if not img.is_floating_point():
            raise TypeError("Input images must be floating-point tensors.")

        if not torch.isfinite(img).all():
            raise ValueError("Input images contain NaN or infinite values.")

        num_queries = validate_queries(queries, img.shape[0])

        # One visual backbone pass for all class queries.
        visual_features = self.backbone(img)

        # Encode all class queries.
        text_features = self.text_enc(queries)

        expected_shape = (
            img.shape[0],
            num_queries,
            self.cfg.embed_dim,
        )

        if tuple(text_features.shape) != expected_shape:
            raise RuntimeError(
                f"Text encoder returned {tuple(text_features.shape)}; "
                f"expected {expected_shape}."
            )

        if visual_features.ndim != 4:
            raise RuntimeError(
                "Backbone must return a feature tensor [B,C,H,W]."
            )

        if visual_features.shape[0] != img.shape[0]:
            raise RuntimeError("Backbone output batch size is incorrect.")

        return visual_features, text_features

    def _predict_base(self, img, queries):
        """Run the shared Model B prediction path."""

        visual_features, text_features = (
            self._extract_features_and_text(img, queries)
        )

        density_maps, cls_logits, visual_embedding = self.head(
            visual_features,
            text_features,
        )

        return (
            density_maps,
            cls_logits,
            visual_embedding,
            text_features,
        )


class ModelA(YOLOCountBase):
    """
    Model A: single-query counting baseline.

    Each image is evaluated with exactly one class query.
    """

    def forward(self, img, queries):
        if img.ndim != 4:
            raise ValueError("Expected images with shape [B,C,H,W].")

        num_queries = validate_queries(queries, img.shape[0])

        if num_queries != 1:
            raise ValueError(
                "Model A accepts exactly one query per image."
            )

        density_maps, cls_logits, _, _ = self._predict_base(
            img,
            queries,
        )

        counts = aggregate_counts(density_maps, cls_logits)

        return counts, density_maps, cls_logits


class ModelB(YOLOCountBase):
    """
    Model B: multi-class counting with shared visual features.

    The backbone runs once per image, regardless of the number
    of class queries.
    """

    def forward(self, img, queries):
        density_maps, cls_logits, _, _ = self._predict_base(
            img,
            queries,
        )

        counts = aggregate_counts(density_maps, cls_logits)

        return counts, density_maps, cls_logits


class ModelC(YOLOCountBase):
    """
    Model C: Model B enhanced with Query-Visual Interaction.

    QVI produces:
        d_den: log-density correction.
        d_cls: classification-logit correction.

    Integration:
        density_C = density_B * exp(clamp(d_den))
        logits_C  = logits_B + d_cls
    """

    def __init__(
        self,
        text_enc: TextEncoder,
        cfg: Optional[Config] = None,
    ):
        if text_enc is None:
            raise ValueError("A TextEncoder instance is required.")

        # Guarantee that Model C owns an independent text encoder.
        # This also handles callers that pass model_B.text_enc.
        independent_text_enc = deepcopy(text_enc)

        super().__init__(independent_text_enc, cfg)

        self.qvi = None

        if self.cfg.use_qvi:
            self.qvi = QVI(
                c_in=self.cfg.embed_dim,
                d_txt=self.cfg.embed_dim,
                c=self.cfg.qvi_channels,
                mode=self.cfg.qvi_type,
                query_chunk_size=getattr(
                    self.cfg,
                    "qvi_query_chunk_size",
                    2,
                ),
            )

    def initialize_from_model_b(
        self,
        model_b: ModelB,
        strict: bool = True,
    ):
        """
        Initialize Model C from Model B.

        Copies weights into independent modules.
        Does not share parameter objects.

        QVI retains its zero-initialized residual output layers.
        """

        if not isinstance(model_b, ModelB):
            raise TypeError("model_b must be an instance of ModelB.")

        if self.qvi is None:
            raise RuntimeError(
                "QVI is disabled in Model C. "
                "Enable QVI before transferring Model B weights."
            )

        module_names = (
            "backbone",
            "text_enc",
            "head",
        )

        for module_name in module_names:
            source = getattr(model_b, module_name)
            destination = getattr(self, module_name)

            destination.load_state_dict(
                source.state_dict(),
                strict=strict,
            )

        # Explicitly verify that no parameters are shared.
        source_parameters = {
            parameter.data_ptr()
            for parameter in model_b.parameters()
        }

        shared_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.data_ptr() in source_parameters
        ]

        if shared_parameters:
            raise RuntimeError(
                "Model B and Model C unexpectedly share parameter storage."
            )

        return self

    def forward(self, img, queries):
        (
            density_maps,
            cls_logits,
            visual_embedding,
            text_features,
        ) = self._predict_base(img, queries)

        if self.qvi is not None:
            d_den, d_cls = self.qvi(
                visual_embedding,
                text_features,
            )

            # Multiplicative residual preserves non-negative density.
            density_maps = density_maps * torch.exp(
                d_den.clamp(min=-5.0, max=5.0)
            )

            # Additive classification residual.
            cls_logits = cls_logits + d_cls

        counts = aggregate_counts(
            density_maps,
            cls_logits,
        )

        return counts, density_maps, cls_logits
