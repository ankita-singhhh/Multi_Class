
"""
Complete YOLO-Count multi-class architecture.

Model A:
    Single-query YOLO-Count baseline.

Model B:
    Multi-class counting with shared visual features.

Model C:
    Model B + Query-Visual Interaction (QVI).

Common output contract:
    counts:        [B, K]
    density_maps:  [B, K, H, W]
    cls_logits:    [B, K, H, W]
"""

from typing import List, Optional

import torch
import torch.nn as nn


# Support both package imports and direct root-level imports.
try:
    from .config import Config
    from .text_encoder import TextEncoder
    from .backbone import YoloBackbonePAFPN
    from .head import MultiClassHead
    from .qvi import QVI
except ImportError:
    from config import Config
    from text_encoder import TextEncoder
    from backbone import YoloBackbonePAFPN
    from head import MultiClassHead
    from qvi import QVI


def validate_queries(queries: List[List[str]], batch_size: int):
    """Validate batched text queries."""

    if not isinstance(queries, list) or len(queries) != batch_size:
        raise ValueError(
            f"Expected {batch_size} query lists, got "
            f"{len(queries) if isinstance(queries, list) else type(queries)}."
        )

    if not queries:
        raise ValueError("Queries cannot be empty.")

    if not all(isinstance(row, list) and len(row) > 0 for row in queries):
        raise ValueError(
            "Every image must have at least one class query."
        )

    num_queries = len(queries[0])

    if any(len(row) != num_queries for row in queries):
        raise ValueError(
            "All images must have the same number of queries. "
            "Use the dataset collate function."
        )

    if any(
        not isinstance(query, str) or not query.strip()
        for row in queries
        for query in row
    ):
        raise ValueError("All queries must be non-empty strings.")

    return num_queries


def aggregate_counts(
    density_maps: torch.Tensor,
    cls_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Convert density and classification maps into per-class counts.

    density_maps: [B, K, H, W], non-negative
    cls_logits:   [B, K, H, W]

    Returns:
        counts: [B, K]
    """

    if density_maps.shape != cls_logits.shape:
        raise ValueError(
            "Density and classification maps must have identical shapes."
        )

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

        self.cfg = cfg if cfg is not None else Config()

        if text_enc is None:
            raise ValueError("A TextEncoder instance is required.")

        self.text_enc = text_enc

        # -----------------------------------------------------
        # YOLO BACKBONE
        # -----------------------------------------------------
        self.backbone = YoloBackbonePAFPN(
            out_ch=self.cfg.backbone_channels,
            pretrained=self.cfg.pretrained_backbone,
            weights=getattr(
                self.cfg,
                "backbone_weights",
                "yolov8n.pt",
            ),
        )

        # -----------------------------------------------------
        # MULTI-CLASS COUNTING HEAD
        # -----------------------------------------------------
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
        """Run shared backbone and text encoder."""

        if img.ndim != 4:
            raise ValueError(
                f"Expected image tensor [B,C,H,W], got {tuple(img.shape)}."
            )

        if img.shape[1] != 3:
            raise ValueError("Input images must have three RGB channels.")

        num_queries = validate_queries(queries, img.shape[0])

        # One visual backbone pass per image.
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

        return visual_features, text_features

    def _predict_base(self, img, queries):
        """Run Model B prediction head."""

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
    Model A: original single-query YOLO-Count baseline.

    Each image receives exactly one class query.
    """

    def __init__(
        self,
        text_enc: TextEncoder,
        cfg: Optional[Config] = None,
    ):
        super().__init__(text_enc, cfg)

    def forward(
        self,
        img: torch.Tensor,
        queries: List[List[str]],
    ):
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
    Model B: multi-class YOLO-Count.

    The backbone runs once per image, regardless of K.
    Class-specific predictions are produced through text-visual
    interaction in the counting head.
    """

    def __init__(
        self,
        text_enc: TextEncoder,
        cfg: Optional[Config] = None,
    ):
        super().__init__(text_enc, cfg)

    def forward(
        self,
        img: torch.Tensor,
        queries: List[List[str]],
    ):
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
        d_den: log-density correction
        d_cls: classification-logit correction

    Density correction is multiplicative:
        density_C = density_B * exp(d_den)

    This guarantees non-negative density predictions.
    """

    def __init__(
        self,
        text_enc: TextEncoder,
        cfg: Optional[Config] = None,
    ):
        super().__init__(text_enc, cfg)

        self.qvi = None

        if self.cfg.use_qvi:
            self.qvi = QVI(
                c_in=self.cfg.embed_dim,
                d_txt=self.cfg.embed_dim,
                c=self.cfg.qvi_channels,
                mode=self.cfg.qvi_type,
            )

    def initialize_from_model_b(
        self,
        model_b: ModelB,
        strict: bool = True,
    ):
        """
        Transfer Model B weights into Model C.

        This initializes the shared backbone, text encoder and
        counting head identically before QVI fine-tuning.

        QVI retains its zero-initialized residual output layers.
        """

        if self.qvi is None:
            raise RuntimeError(
                "Cannot initialize QVI weights because QVI is disabled."
            )

        modules = (
            "backbone",
            "text_enc",
            "head",
        )

        for module_name in modules:
            source = getattr(model_b, module_name)
            destination = getattr(self, module_name)

            destination.load_state_dict(
                source.state_dict(),
                strict=strict,
            )

        return self

    def forward(
        self,
        img: torch.Tensor,
        queries: List[List[str]],
    ):
        density_maps, cls_logits, visual_embedding, text_features = (
            self._predict_base(img, queries)
        )

        # -----------------------------------------------------
        # QUERY-VISUAL INTERACTION
        # -----------------------------------------------------
        if self.qvi is not None:

            d_den, d_cls = self.qvi(
                visual_embedding,
                text_features,
            )

            # -------------------------------------------------
            # MULTIPLICATIVE DENSITY RESIDUAL
            # -------------------------------------------------
            density_maps = density_maps * torch.exp(
                d_den.clamp(min=-5.0, max=5.0)
            )

            # -------------------------------------------------
            # ADDITIVE CLASSIFICATION RESIDUAL
            # -------------------------------------------------
            cls_logits = cls_logits + d_cls

        counts = aggregate_counts(
            density_maps,
            cls_logits,
        )

        return counts, density_maps, cls_logits
