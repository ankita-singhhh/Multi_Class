
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


def _validate_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise when a tensor contains NaN or infinite values."""
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains NaN or infinite values.")


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
    if not isinstance(density, torch.Tensor):
        raise TypeError("density must be a torch.Tensor.")

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")

    if density.ndim != 4:
        raise ValueError(
            f"Expected density [B,K,H,W], got {tuple(density.shape)}."
        )

    if mask.ndim != 4:
        raise ValueError(
            f"Expected mask [B,1,H,W], got {tuple(mask.shape)}."
        )

    if density.shape[1] <= 0:
        raise ValueError("Density must contain at least one query channel.")

    if mask.shape[1] != 1:
        raise ValueError("Spatial mask must have exactly one channel.")

    if density.shape[0] != mask.shape[0]:
        raise ValueError("Batch dimensions do not match.")

    if density.shape[-2:] != mask.shape[-2:]:
        raise ValueError("Spatial dimensions do not match.")

    if density.device != mask.device:
        raise ValueError("Density and mask must be on the same device.")

    if not density.is_floating_point():
        raise TypeError("Density must have a floating-point dtype.")

    if not mask.is_floating_point() and mask.dtype != torch.bool:
        raise TypeError("Mask must be floating point or boolean.")

    _validate_finite("density", density)

    mask = mask.to(dtype=density.dtype)

    _validate_finite("mask", mask)

    if torch.any(mask < 0).item() or torch.any(mask > 1).item():
        raise ValueError("Spatial mask values must be in [0, 1].")

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
    if not isinstance(density, torch.Tensor):
        raise TypeError("density must be a torch.Tensor.")

    if not isinstance(cls_logits, torch.Tensor):
        raise TypeError("cls_logits must be a torch.Tensor.")

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

    if not density.is_floating_point():
        raise TypeError("Density must have a floating-point dtype.")

    if not cls_logits.is_floating_point():
        raise TypeError("Classification logits must be floating point.")

    _validate_finite("density", density)
    _validate_finite("cls_logits", cls_logits)

    if torch.any(density < 0).item():
        raise ValueError("Density predictions must be non-negative.")

    # Accumulate in float32 for improved numerical stability.
    density_fp32 = density.float()
    logits_fp32 = cls_logits.float()

    confidence = torch.sigmoid(logits_fp32)

    counts = (
        density_fp32 * confidence
    ).sum(dim=(-2, -1))

    if query_mask is not None:
        if not isinstance(query_mask, torch.Tensor):
            raise TypeError("query_mask must be a torch.Tensor.")

        if query_mask.ndim != 2:
            raise ValueError("query_mask must have shape [B,K].")

        if query_mask.shape != counts.shape:
            raise ValueError(
                f"Expected query mask {tuple(counts.shape)}, "
                f"got {tuple(query_mask.shape)}."
            )

        if query_mask.device != counts.device:
            raise ValueError(
                "query_mask and predictions must be on the same device."
            )

        query_mask = query_mask.bool()

        # where() prevents invalid padded positions from contaminating
        # the output through multiplication with zero.
        counts = torch.where(
            query_mask,
            counts,
            torch.zeros_like(counts),
        )

    return counts


def predictions_to_list(
    counts: torch.Tensor,
) -> List[List[float]]:
    """
    Convert count predictions to nested Python lists.

    Preserves [B,K], including the single-class case.
    """
    if not isinstance(counts, torch.Tensor):
        raise TypeError("counts must be a torch.Tensor.")

    if counts.ndim != 2:
        raise ValueError(
            f"Expected counts [B,K], got {tuple(counts.shape)}."
        )

    _validate_finite("counts", counts)

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
    if not isinstance(predictions, torch.Tensor):
        raise TypeError("predictions must be a torch.Tensor.")

    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor.")

    if not isinstance(query_mask, torch.Tensor):
        raise TypeError("query_mask must be a torch.Tensor.")

    if predictions.ndim != 2:
        raise ValueError("Predictions must have shape [B,K].")

    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes must match.")

    if predictions.shape != query_mask.shape:
        raise ValueError("Prediction and mask shapes must match.")

    if predictions.device != targets.device:
        raise ValueError(
            "Predictions and targets must be on the same device."
        )

    if predictions.device != query_mask.device:
        raise ValueError(
            "Predictions and query_mask must be on the same device."
        )

    if not predictions.is_floating_point():
        raise TypeError("Predictions must be floating point.")

    if not targets.is_floating_point():
        raise TypeError("Targets must be floating point.")

    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'.")

    valid_mask = query_mask.bool()

    if not valid_mask.any().item():
        # Differentiable zero loss when a batch has no valid queries.
        return predictions.sum() * 0.0

    # Select valid positions BEFORE arithmetic.
    # This avoids NaN * 0 contamination from padded positions.
    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]

    _validate_finite("valid predictions", valid_predictions)
    _validate_finite("valid targets", valid_targets)

    errors = torch.abs(
        valid_predictions.float() - valid_targets.float()
    )

    if reduction == "sum":
        return errors.sum()

    return errors.mean()
