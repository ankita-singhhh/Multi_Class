
"""
Multi-class YOLO-Count utility functions.

Responsibilities:
    1. Apply spatial masks across multiple classes.
    2. Aggregate density and classification maps into counts.
    3. Convert predictions into Python lists.
    4. Handle padded query masks safely.

Tensor contracts:
    density:    [B, K, H, W]
    cls_logits: [B, K, H, W]
    counts:     [B, K]
"""

from typing import List, Optional

import torch


def apply_mask_broadcast(
    density: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a spatial validity mask to every class density map.

    Args:
        density: [B, K, H, W]
        mask:    [B, 1, H, W]

    Returns:
        Masked density: [B, K, H, W]
    """

    if density.ndim != 4:
        raise ValueError(
            f"Expected density [B,K,H,W], got {tuple(density.shape)}"
        )

    if mask.ndim != 4:
        raise ValueError(
            f"Expected mask [B,1,H,W], got {tuple(mask.shape)}"
        )

    if mask.shape[1] != 1:
        raise ValueError(
            "Spatial mask must have exactly one channel."
        )

    if density.shape[0] != mask.shape[0]:
        raise ValueError("Batch dimensions do not match.")

    if density.shape[-2:] != mask.shape[-2:]:
        raise ValueError("Spatial dimensions do not match.")

    if density.device != mask.device:
        raise ValueError("Density and mask must be on the same device.")

    if not torch.is_floating_point(mask):
        mask = mask.float()

    return density * mask


def aggregate_counts(
    density: torch.Tensor,
    cls_logits: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert density and classification maps into per-class counts.

    Formula:
        count[b,k] = sum(
            density[b,k,h,w] * sigmoid(cls_logits[b,k,h,w])
        )

    Args:
        density:
            Non-negative density maps [B,K,H,W].

        cls_logits:
            Classification logits [B,K,H,W].

        query_mask:
            Optional validity mask [B,K].
            Padded class queries receive zero counts.

    Returns:
        counts: [B,K]
    """

    if density.ndim != 4 or cls_logits.ndim != 4:
        raise ValueError(
            "Density and classification maps must be 4D tensors."
        )

    if density.shape != cls_logits.shape:
        raise ValueError(
            f"Shape mismatch: density {tuple(density.shape)}, "
            f"classification {tuple(cls_logits.shape)}."
        )

    if density.device != cls_logits.device:
        raise ValueError(
            "Density and classification maps must be on the same device."
        )

    if torch.any(density < 0):
        raise ValueError(
            "Density predictions must be non-negative."
        )

    confidence = torch.sigmoid(cls_logits)

    # Integrate spatial density weighted by class confidence.
    counts = (density * confidence).sum(dim=(-2, -1))

    # Mask padded query positions.
    if query_mask is not None:

        if query_mask.ndim != 2:
            raise ValueError("query_mask must have shape [B,K].")

        if query_mask.shape != counts.shape:
            raise ValueError(
                f"Expected query mask {tuple(counts.shape)}, "
                f"got {tuple(query_mask.shape)}."
            )

        counts = counts * query_mask.to(
            device=counts.device,
            dtype=counts.dtype,
        )

    return counts


def predictions_to_list(
    counts: torch.Tensor,
) -> List[List[float]]:
    """
    Convert count predictions to nested Python lists.

    Preserves [B,K], including the single-class case.
    """

    if counts.ndim != 2:
        raise ValueError(
            f"Expected counts [B,K], got {tuple(counts.shape)}"
        )

    return counts.detach().float().cpu().tolist()


def masked_count_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    query_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute masked count error.

    Padded query positions do not contribute to the loss.

    Args:
        predictions: [B,K]
        targets:      [B,K]
        query_mask:   [B,K]

    Returns:
        Scalar loss for reduction='mean' or 'sum'.
    """

    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes must match.")

    if predictions.shape != query_mask.shape:
        raise ValueError("Prediction and mask shapes must match.")

    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'.")

    mask = query_mask.to(
        device=predictions.device,
        dtype=predictions.dtype,
    )

    errors = torch.abs(predictions - targets)

    masked_errors = errors * mask

    if reduction == "sum":
        return masked_errors.sum()

    valid_count = mask.sum().clamp_min(1.0)

    return masked_errors.sum() / valid_count
