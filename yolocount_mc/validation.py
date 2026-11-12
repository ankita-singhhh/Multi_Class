
"""
Evaluation utilities for YOLO-Count multi-class counting.

Metrics:
    - Overall MAE
    - Overall RMSE
    - Per-class MAE
    - Per-class RMSE
    - Per-image MAE
    - Per-image RMSE

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
    """
    Accumulates count errors across batches.

    Per-class metrics are grouped by class name, not query index.
    """

    def __init__(self):
        self.class_abs_errors = defaultdict(float)
        self.class_squared_errors = defaultdict(float)
        self.class_counts = defaultdict(int)

        self.total_abs_error = 0.0
        self.total_squared_error = 0.0
        self.total_valid_queries = 0

        self.image_abs_errors = []
        self.image_squared_errors = []

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

        if predictions.ndim != 2 or targets.ndim != 2:
            raise ValueError("Predictions and targets must have shape [B,K].")

        if predictions.shape != targets.shape:
            raise ValueError("Prediction and target shapes do not match.")

        batch_size, num_queries = predictions.shape

        if len(queries) != batch_size:
            raise ValueError("Query batch size does not match predictions.")

        if mask is None:
            mask = torch.ones_like(predictions)

        if mask.shape != predictions.shape:
            raise ValueError("Mask shape does not match predictions.")

        predictions = predictions.detach().double().cpu()
        targets = targets.detach().double().cpu()
        mask = mask.detach().cpu().bool()

        for b in range(batch_size):

            if len(queries[b]) != num_queries:
                raise ValueError(
                    f"Query count mismatch in batch item {b}."
                )

            image_abs = []
            image_squared = []

            for k, class_name in enumerate(queries[b]):

                if not mask[b, k]:
                    continue

                if not torch.isfinite(predictions[b, k]):
                    raise ValueError(
                        f"Non-finite prediction for class {class_name}."
                    )

                if not torch.isfinite(targets[b, k]):
                    raise ValueError(
                        f"Non-finite target for class {class_name}."
                    )

                error = predictions[b, k] - targets[b, k]

                abs_error = abs(error.item())
                squared_error = error.item() ** 2

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

                self.image_squared_errors.append(
                    sum(image_squared) / len(image_squared)
                )

    def compute(self) -> Dict:
        """Calculate all accumulated metrics."""

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

        mae_overall = (
            self.total_abs_error / self.total_valid_queries
        )

        rmse_overall = (
            self.total_squared_error / self.total_valid_queries
        ) ** 0.5

        return {
            "mae_overall": mae_overall,
            "rmse_overall": rmse_overall,
            "mae_per_class": mae_per_class,
            "rmse_per_class": rmse_per_class,
            "image_mae": (
                sum(self.image_abs_errors)
                / len(self.image_abs_errors)
                if self.image_abs_errors else 0.0
            ),
            "image_rmse": (
                sum(self.image_squared_errors)
                / len(self.image_squared_errors)
            ) ** 0.5 if self.image_squared_errors else 0.0,
            "num_valid_queries": self.total_valid_queries,
            "num_classes_evaluated": len(self.class_counts),
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
        This function uses positional class columns. Use
        evaluate_model() for class-name-based aggregation when
        query ordering changes across batches.
    """

    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("pred and gt must have shape [N,K].")

    if pred.shape != gt.shape:
        raise ValueError("Prediction and target shapes must match.")

    if mask is None:
        mask = torch.ones_like(pred)

    if mask.shape != pred.shape:
        raise ValueError("Mask shape must match predictions.")

    valid = mask.bool()

    if not valid.any():
        raise ValueError("No valid evaluation entries.")

    error = pred - gt
    abs_error = error.abs()
    squared_error = error.square()

    valid_float = valid.to(pred.dtype)
    count = valid_float.sum(dim=0)

    mae_per_class = (
        (abs_error * valid_float).sum(dim=0)
        / count.clamp_min(1)
    )

    rmse_per_class = (
        (squared_error * valid_float).sum(dim=0)
        / count.clamp_min(1)
    ).sqrt()

    # Mark columns without observations as NaN.
    mae_per_class = mae_per_class.masked_fill(count == 0, float("nan"))
    rmse_per_class = rmse_per_class.masked_fill(count == 0, float("nan"))

    total = valid_float.sum().clamp_min(1)

    mae_overall = (
        abs_error * valid_float
    ).sum() / total

    rmse_overall = (
        squared_error * valid_float
    ).sum().div(total).sqrt()

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
    Evaluate a complete model on a real validation dataset.

    Supports Model A, Model B and Model C.
    """

    model.eval()

    metrics = CountingMetrics()

    device = torch.device(device)

    for images, queries, targets, mask in dataloader:

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

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
