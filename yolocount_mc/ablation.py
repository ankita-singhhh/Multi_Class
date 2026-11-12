
"""
Controlled QVI ablation experiments for Multi-Class YOLO-Count.

Variants:
    B          : Multi-class YOLO-Count baseline
    QVI-gated  : Model B + gated QVI
    QVI-mod    : Model B + modulation QVI

Protocol:
    - Real training and validation datasets only.
    - At least three independent random seeds.
    - One trained Model B per seed.
    - Both QVI variants start from the same seed-specific Model B.
    - Baseline weights are checked for unintended modification.
    - Identical validation dataset and evaluation protocol.
    - Per-seed MAE/RMSE and aggregate mean/sample standard deviation.
    - Results saved as JSON and CSV.

This module does not create toy datasets.
"""

import csv
import hashlib
import json
import random
import statistics
from pathlib import Path
from dataclasses import replace
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .train import train_model_B, train_model_C
from .dataset import collate


# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------

def set_experiment_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Improve repeatability of CUDA/cuDNN execution.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _model_state_digest(model) -> str:
    """
    Calculate a digest of model state without retaining a second
    complete copy of its parameters in memory.
    """
    digest = hashlib.sha256()

    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))

        value = tensor.detach().contiguous().cpu()

        # Byte view supports dtypes such as bfloat16.
        digest.update(value.view(torch.uint8).numpy().tobytes())

    return digest.hexdigest()


def _assert_model_unchanged(model, original_digest: str) -> None:
    """Raise if a QVI experiment unexpectedly modified Model B."""
    current_digest = _model_state_digest(model)

    if current_digest != original_digest:
        raise RuntimeError(
            "The baseline Model B weights changed during QVI training. "
            "Check Model C initialization and parameter independence."
        )


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

def _validate_seeds(seeds: Sequence[int]) -> tuple:
    seeds = tuple(seeds)

    if len(seeds) < 3:
        raise ValueError(
            "Research ablations require at least 3 independent seeds."
        )

    if any(
        not isinstance(seed, int) or isinstance(seed, bool)
        for seed in seeds
    ):
        raise TypeError("Every experiment seed must be an integer.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Experiment seeds must be unique.")

    return seeds


def _validate_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_experiment_inputs(
    train_dataset,
    val_dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    qvi_fine_tune_epochs: int,
) -> None:
    if train_dataset is None or val_dataset is None:
        raise ValueError(
            "Real training and validation datasets are required."
        )

    _validate_positive_integer("epochs", epochs)
    _validate_positive_integer("batch_size", batch_size)

    if (
        not isinstance(qvi_fine_tune_epochs, int)
        or isinstance(qvi_fine_tune_epochs, bool)
        or qvi_fine_tune_epochs < 0
    ):
        raise ValueError(
            "qvi_fine_tune_epochs must be a non-negative integer."
        )

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

@torch.inference_mode()
def evaluate_counting(
    model,
    dataset,
    cfg: Config,
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate a trained counting model.

    Metrics are calculated over valid, non-padded class-image pairs.
    MAE and RMSE use the same pair-level aggregation.
    """
    model.eval()

    if batch_size is None:
        batch_size = getattr(cfg, "batch_size", 4)

    _validate_positive_integer("batch_size", batch_size)

    device = torch.device(cfg.device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 0),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )

    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    valid_count = 0

    for images, queries, targets, query_mask in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        query_mask = query_mask.to(
            device, non_blocking=True
        ).bool()

        predictions, _, _ = model(images, queries)

        if predictions.shape != targets.shape:
            raise ValueError(
                "Prediction and target shapes differ: "
                f"{tuple(predictions.shape)} vs "
                f"{tuple(targets.shape)}"
            )

        if query_mask.shape != targets.shape:
            raise ValueError(
                "Query mask and target shapes differ: "
                f"{tuple(query_mask.shape)} vs "
                f"{tuple(targets.shape)}"
            )

        valid_predictions = predictions.float()[query_mask]
        valid_targets = targets.float()[query_mask]

        if valid_predictions.numel() == 0:
            continue

        if not torch.isfinite(valid_predictions).all():
            raise ValueError(
                "Model produced NaN or infinite valid predictions."
            )

        if not torch.isfinite(valid_targets).all():
            raise ValueError(
                "Validation dataset contains NaN or infinite valid targets."
            )

        errors = valid_predictions - valid_targets

        absolute_error_sum += errors.abs().sum().item()
        squared_error_sum += errors.square().sum().item()
        valid_count += errors.numel()

    if valid_count == 0:
        raise RuntimeError(
            "Validation produced no valid class-query samples."
        )

    mae = absolute_error_sum / valid_count
    rmse = (squared_error_sum / valid_count) ** 0.5

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "num_evaluated": int(valid_count),
    }


# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

def summarize_results(
    results: Dict[str, list],
) -> Dict[str, Dict]:
    """Calculate mean and sample standard deviation across seeds."""
    summary = {}

    for variant, runs in results.items():
        if not runs:
            raise ValueError(
                f"No experiment results found for variant {variant}."
            )

        maes = [float(run["mae"]) for run in runs]
        rmses = [float(run["rmse"]) for run in runs]

        if not all(np.isfinite(value) for value in maes + rmses):
            raise ValueError(
                f"Non-finite metrics found for variant {variant}."
            )

        summary[variant] = {
            "mae_mean": float(statistics.mean(maes)),
            "mae_std": float(
                statistics.stdev(maes) if len(maes) > 1 else 0.0
            ),
            "rmse_mean": float(statistics.mean(rmses)),
            "rmse_std": float(
                statistics.stdev(rmses) if len(rmses) > 1 else 0.0
            ),
            "num_seeds": len(runs),
        }

    return summary


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_ablation_results(
    results: Dict[str, list],
    summary: Dict[str, Dict],
    output_dir,
    metadata: Optional[Dict] = None,
) -> Dict[str, str]:
    """Save per-seed results, aggregate results and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata or {},
        "per_seed_results": results,
        "summary": summary,
    }

    json_path = output_dir / "ablation_results.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, allow_nan=False)

    csv_path = output_dir / "ablation_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "variant",
            "seed",
            "mae",
            "rmse",
            "num_evaluated",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for variant, runs in results.items():
            for run in runs:
                writer.writerow({
                    "variant": variant,
                    **run,
                })

    summary_path = output_dir / "ablation_summary.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "variant",
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "rmse_std",
            "num_seeds",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for variant, metrics in summary.items():
            writer.writerow({
                "variant": variant,
                **metrics,
            })

    return {
        "json": str(json_path),
        "detailed_csv": str(csv_path),
        "summary_csv": str(summary_path),
    }


# ---------------------------------------------------------
# MAIN ABLATION EXPERIMENT
# ---------------------------------------------------------

def run_ablation(
    train_dataset,
    val_dataset,
    seeds: Sequence[int] = (0, 1, 2),
    cfg: Optional[Config] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    qvi_fine_tune_epochs: int = 0,
):
    """
    Run controlled Model B vs QVI ablation experiments.

    Each seed trains one baseline Model B. Both QVI variants
    are initialized independently from that same baseline.

    Args:
        train_dataset: Real training dataset.
        val_dataset: Real validation dataset.
        seeds: At least three distinct integer seeds.
        cfg: Project configuration.
        epochs: Model B and QVI-only training epochs.
        batch_size: Training and validation batch size.
        learning_rate: Learning rate.
        qvi_fine_tune_epochs: Optional QVI fine-tuning epochs.

    Returns:
        Per-seed results, aggregate statistics and saved file paths.
    """
    if cfg is None:
        cfg = Config()

    seeds = _validate_seeds(seeds)

    epochs = (
        getattr(cfg, "epochs", 20)
        if epochs is None else epochs
    )

    batch_size = (
        getattr(cfg, "batch_size", 4)
        if batch_size is None else batch_size
    )

    learning_rate = (
        getattr(cfg, "lr", 1e-4)
        if learning_rate is None else learning_rate
    )

    _validate_experiment_inputs(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        qvi_fine_tune_epochs=qvi_fine_tune_epochs,
    )

    variants = {
        "B": [],
        "QVI-gated": [],
        "QVI-mod": [],
    }

    print("\n" + "=" * 75)
    print("YOLO-COUNT CONTROLLED ABLATION EXPERIMENT")
    print("=" * 75)
    print(f"Seeds: {seeds}")
    print(f"Model B epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"QVI fine-tuning epochs: {qvi_fine_tune_epochs}")
    print("=" * 75)

    for seed in seeds:
        print(f"\n\n{'#' * 65}")
        print(f"EXPERIMENT SEED: {seed}")
        print(f"{'#' * 65}")

        set_experiment_seed(seed)
        seed_cfg = replace(cfg, seed=seed)

        # -------------------------------------------------
        # 1. TRAIN BASELINE MODEL B
        # -------------------------------------------------

        print("\n[1/3] Training baseline Model B...")

        model_B = train_model_B(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            bs=batch_size,
            lr=learning_rate,
            cfg=seed_cfg,
        )

        if model_B is None:
            raise RuntimeError(
                "train_model_B must return the trained model."
            )

        model_B.eval()

        baseline_metrics = evaluate_counting(
            model_B,
            val_dataset,
            seed_cfg,
            batch_size=batch_size,
        )

        variants["B"].append({
            "seed": int(seed),
            **baseline_metrics,
        })

        print(
            f"Model B | MAE: {baseline_metrics['mae']:.4f} "
            f"| RMSE: {baseline_metrics['rmse']:.4f}"
        )

        # Save a digest to verify that Model B is not modified.
        baseline_digest = _model_state_digest(model_B)

        # -------------------------------------------------
        # 2. TRAIN BOTH QVI VARIANTS FROM THE SAME MODEL B
        # -------------------------------------------------

        for mode, variant_name in (
            ("gated", "QVI-gated"),
            ("modulation", "QVI-mod"),
        ):
            print(
                f"\nTraining {variant_name} "
                f"from seed {seed} Model B..."
            )

            # Reset RNG so the variant setup is reproducible.
            set_experiment_seed(seed)

            qvi_cfg = replace(
                seed_cfg,
                use_qvi=True,
                qvi_type=mode,
            )

            model_C = train_model_C(
                model_B=model_B,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=epochs,
                bs=batch_size,
                lr=learning_rate,
                cfg=qvi_cfg,
                fine_tune_epochs=qvi_fine_tune_epochs,
            )

            if model_C is None:
                raise RuntimeError(
                    f"train_model_C returned None for {variant_name}."
                )

            # Confirm that QVI training did not alter the baseline.
            _assert_model_unchanged(model_B, baseline_digest)

            model_C.eval()

            metrics = evaluate_counting(
                model_C,
                val_dataset,
                qvi_cfg,
                batch_size=batch_size,
            )

            variants[variant_name].append({
                "seed": int(seed),
                **metrics,
            })

            print(
                f"{variant_name} | "
                f"MAE: {metrics['mae']:.4f} "
                f"| RMSE: {metrics['rmse']:.4f}"
            )

            del model_C

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model_B

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------
    # AGGREGATE RESULTS
    # -----------------------------------------------------

    summary = summarize_results(variants)

    print("\n\n" + "=" * 85)
    print("FINAL ABLATION RESULTS")
    print("=" * 85)

    print(
        f"{'Variant':<16}"
        f"{'MAE Mean':>14}"
        f"{'MAE Std':>14}"
        f"{'RMSE Mean':>14}"
        f"{'RMSE Std':>14}"
    )

    print("-" * 85)

    for variant, metrics in summary.items():
        print(
            f"{variant:<16}"
            f"{metrics['mae_mean']:>14.4f}"
            f"{metrics['mae_std']:>14.4f}"
            f"{metrics['rmse_mean']:>14.4f}"
            f"{metrics['rmse_std']:>14.4f}"
        )

    print("=" * 85)

    # -----------------------------------------------------
    # INTERPRETATION
    # -----------------------------------------------------

    baseline_mae = summary["B"]["mae_mean"]

    print("\nQVI comparison against Model B:")

    for variant in ("QVI-gated", "QVI-mod"):
        qvi_mae = summary[variant]["mae_mean"]
        improvement = baseline_mae - qvi_mae

        improvement_percent = (
            improvement / baseline_mae * 100.0
            if baseline_mae > 0 else 0.0
        )

        print(
            f"{variant}: "
            f"{improvement:+.4f} MAE improvement "
            f"({improvement_percent:+.2f}%)"
        )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    output_dir = Path(cfg.output_dir) / "ablation"

    metadata = {
        "seeds": list(seeds),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "qvi_fine_tune_epochs": qvi_fine_tune_epochs,
        "validation_metric": "class-image-pair MAE and RMSE",
        "standard_deviation": "sample standard deviation",
        "variants": ["B", "QVI-gated", "QVI-mod"],
    }

    saved_files = save_ablation_results(
        variants,
        summary,
        output_dir,
        metadata=metadata,
    )

    print("\nSaved experiment results:")

    for name, path in saved_files.items():
        print(f"{name}: {path}")

    return {
        "per_seed_results": variants,
        "summary": summary,
        "saved_files": saved_files,
        "metadata": metadata,
    }
