
"""
Evaluation utilities for YOLO-Count multi-class counting.

Metrics:
    - Overall MAE
    - Overall RMSE
    - Per-class MAE
    - Per-class RMSE
    - Per-image MAE
    - Mean per-image RMSE

Supports:
    - Model A: single-query counting
    - Model B: multi-class counting
    - Model C: QVI-enhanced counting
    - Variable query counts with padding masks
    - Category-name-based metric aggregation
"""

from collections import defaultdict
from typing import Dict, List, Optional

import torch


class CountingMetrics:
    """Accumulate count errors using class names."""

    def __init__(self):
        self.class_abs_errors = defaultdict(float)
        self.class_squared_errors = defaultdict(float)
        self.class_counts = defaultdict(int)

        self.total_abs_error = 0.0
        self.total_squared_error = 0.0
        self.total_valid_queries = 0

        self.image_abs_errors = []
        self.image_rmse_values = []

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        queries: List[List[str]],
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Update metrics from one batch.

        Args:
            predictions: [B,K]
            targets: [B,K]
            queries: List[List[str]]
            mask: Optional [B,K] validity mask.
        """

        if not isinstance(predictions, torch.Tensor):
            raise TypeError("predictions must be a torch.Tensor.")

        if not isinstance(targets, torch.Tensor):
            raise TypeError("targets must be a torch.Tensor.")

        if predictions.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                "Predictions and targets must have shape [B,K]."
            )

        if predictions.shape != targets.shape:
            raise ValueError(
                "Prediction and target shapes do not match."
            )

        if predictions.device != targets.device:
            raise ValueError(
                "Predictions and targets must share a device."
            )

        batch_size, num_queries = predictions.shape

        if not isinstance(queries, list) or len(queries) != batch_size:
            raise ValueError(
                "Query batch size does not match predictions."
            )

        if mask is None:
            mask = torch.ones_like(predictions, dtype=torch.bool)

        if mask.shape != predictions.shape:
            raise ValueError("Mask shape does not match predictions.")

        predictions = predictions.detach().double().cpu()
        targets = targets.detach().double().cpu()
        mask = mask.detach().cpu().bool()

        for b in range(batch_size):

            if not isinstance(queries[b], list):
                raise TypeError(
                    f"Queries for batch item {b} must be a list."
                )

            if len(queries[b]) != num_queries:
                raise ValueError(
                    f"Query count mismatch in batch item {b}."
                )

            image_abs = []
            image_squared = []

            for k, class_name in enumerate(queries[b]):

                if not mask[b, k]:
                    continue

                if not isinstance(class_name, str) or not class_name.strip():
                    raise ValueError(
                        "Every valid query must have a non-empty class name."
                    )

                if not torch.isfinite(predictions[b, k]):
                    raise ValueError(
                        f"Non-finite prediction for class {class_name}."
                    )

                if not torch.isfinite(targets[b, k]):
                    raise ValueError(
                        f"Non-finite target for class {class_name}."
                    )

                error = (
                    predictions[b, k] - targets[b, k]
                ).item()

                abs_error = abs(error)
                squared_error = error ** 2

                class_name = class_name.strip()

                self.class_abs_errors[class_name] += abs_error
                self.class_squared_errors[class_name] += squared_error
                self.class_counts[class_name] += 1

                self.total_abs_error += abs_error
                self.total_squared_error += squared_error
                self.total_valid_queries += 1

                image_abs.append(abs_error)
                image_squared.append(squared_error)

            if image_abs:
                self.image_abs_errors.append(
                    sum(image_abs) / len(image_abs)
                )

                self.image_rmse_values.append(
                    (sum(image_squared) / len(image_squared)) ** 0.5
                )

    def compute(self) -> Dict:
        """Calculate accumulated evaluation metrics."""

        if self.total_valid_queries == 0:
            raise ValueError("No valid queries were evaluated.")

        mae_per_class = {}
        rmse_per_class = {}

        for class_name, count in self.class_counts.items():

            mae_per_class[class_name] = (
                self.class_abs_errors[class_name] / count
            )

            rmse_per_class[class_name] = (
                self.class_squared_errors[class_name] / count
            ) ** 0.5

        return {
            "mae_overall": (
                self.total_abs_error / self.total_valid_queries
            ),
            "rmse_overall": (
                self.total_squared_error / self.total_valid_queries
            ) ** 0.5,
            "mae_per_class": mae_per_class,
            "rmse_per_class": rmse_per_class,
            "image_mae": (
                sum(self.image_abs_errors) / len(self.image_abs_errors)
            ),
            "image_rmse": (
                sum(self.image_rmse_values) / len(self.image_rmse_values)
            ),
            "num_valid_queries": self.total_valid_queries,
            "num_classes_evaluated": len(self.class_counts),
            "num_images_evaluated": len(self.image_abs_errors),
            "class_counts": dict(self.class_counts),
        }


@torch.no_grad()
def evaluate(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Evaluate pre-collected predictions and targets.

    Args:
        pred: [N,K]
        gt: [N,K]
        mask: Optional [N,K]

    Note:
        This function uses positional class columns.
        Use evaluate_model() for class-name-based aggregation.
    """

    if not isinstance(pred, torch.Tensor) or not isinstance(gt, torch.Tensor):
        raise TypeError("pred and gt must be torch.Tensor objects.")

    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("pred and gt must have shape [N,K].")

    if pred.shape != gt.shape:
        raise ValueError("Prediction and target shapes must match.")

    if pred.device != gt.device:
        raise ValueError("pred and gt must share a device.")

    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)

    if mask.shape != pred.shape:
        raise ValueError("Mask shape must match predictions.")

    valid = mask.bool()

    if not valid.any():
        raise ValueError("No valid evaluation entries.")

    # Select valid values before arithmetic. This avoids NaN * 0
    # contaminating metrics from padded positions.
    valid_pred = pred[valid].double()
    valid_gt = gt[valid].double()

    if not torch.isfinite(valid_pred).all():
        raise ValueError("Valid predictions contain NaN or infinity.")

    if not torch.isfinite(valid_gt).all():
        raise ValueError("Valid targets contain NaN or infinity.")

    error = valid_pred - valid_gt

    mae_overall = error.abs().mean()
    rmse_overall = error.square().mean().sqrt()

    # Per-column metrics.
    column_mae = []
    column_rmse = []

    for k in range(pred.shape[1]):

        column_valid = valid[:, k]

        if not column_valid.any():
            column_mae.append(float("nan"))
            column_rmse.append(float("nan"))
            continue

        column_error = (
            pred[:, k][column_valid].double()
            - gt[:, k][column_valid].double()
        )

        column_mae.append(
            column_error.abs().mean()
        )

        column_rmse.append(
            column_error.square().mean().sqrt()
        )

    mae_per_class = torch.tensor(
        column_mae,
        dtype=torch.float64,
        device=pred.device,
    )

    rmse_per_class = torch.tensor(
        column_rmse,
        dtype=torch.float64,
        device=pred.device,
    )

    return {
        "mae_per_class": mae_per_class,
        "rmse_per_class": rmse_per_class,
        "mae_overall": mae_overall,
        "rmse_overall": rmse_overall,
    }


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
) -> Dict:
    """
    Evaluate Model A, B or C on a real validation dataset.

    Metrics are aggregated by actual class names.
    """

    if dataloader is None:
        raise ValueError("A validation DataLoader is required.")

    device = torch.device(device)

    model.eval()

    metrics = CountingMetrics()

    for images, queries, targets, mask in dataloader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        mask = mask.to(
            device,
            non_blocking=True,
        )

        predictions, density_maps, cls_logits = model(
            images,
            queries,
        )

        metrics.update(
            predictions=predictions,
            targets=targets,
            queries=queries,
            mask=mask,
        )

    return metrics.compute()
