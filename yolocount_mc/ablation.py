
"""
Step 11: Controlled QVI ablation experiments.

Variants:
    B             : Multi-class YOLO-Count baseline
    QVI-gated     : Model B + gated QVI
    QVI-mod       : Model B + modulation QVI

Protocol:
    - Real training and validation datasets only.
    - At least three independent random seeds.
    - Each QVI variant starts from the same seed-specific
      trained Model B initialization.
    - Identical validation data and evaluation protocol.
    - Report per-seed MAE/RMSE, mean and standard deviation.
    - Save experiment results as JSON and CSV.

No toy datasets are used in this module.
"""

import csv
import json
import random
import statistics
from pathlib import Path
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .config import Config
    from .train import train_model_B, train_model_C
    from .dataset import collate
except ImportError:
    from config import Config
    from train import train_model_B, train_model_C
    from dataset import collate


# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------

def set_experiment_seed(seed: int):
    """Set random seeds across Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
    Evaluate a trained counting model on a real dataset.

    Computes metrics only over valid, non-padded queries.

    Returns:
        MAE
        RMSE
        Number of evaluated class-image pairs
    """

    model.eval()

    batch_size = batch_size or getattr(cfg, "batch_size", 4)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 0),
        pin_memory=(
            cfg.device.type == "cuda"
            if isinstance(cfg.device, torch.device)
            else str(cfg.device).startswith("cuda")
        ),
        collate_fn=collate,
    )

    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    valid_count = 0

    for images, queries, targets, query_mask in loader:

        images = images.to(cfg.device, non_blocking=True)
        targets = targets.to(
            cfg.device, non_blocking=True
        )
        query_mask = query_mask.to(
            cfg.device, non_blocking=True
        ).bool()

        predictions, _, _ = model(images, queries)

        if predictions.shape != targets.shape:
            raise ValueError(
                "Prediction and target shapes differ: "
                f"{predictions.shape} vs {targets.shape}"
            )

        errors = predictions.float() - targets.float()

        valid_errors = errors[query_mask]

        if valid_errors.numel() == 0:
            continue

        absolute_error_sum += (
            valid_errors.abs().sum().item()
        )

        squared_error_sum += (
            valid_errors.square().sum().item()
        )

        valid_count += valid_errors.numel()

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
    results: Dict[str, List[Dict]],
) -> Dict[str, Dict]:
    """
    Calculate mean and sample standard deviation
    across independent random seeds.
    """

    summary = {}

    for variant, runs in results.items():

        maes = [run["mae"] for run in runs]
        rmses = [run["rmse"] for run in runs]

        summary[variant] = {
            "mae_mean": statistics.mean(maes),
            "mae_std": (
                statistics.stdev(maes)
                if len(maes) > 1 else 0.0
            ),
            "rmse_mean": statistics.mean(rmses),
            "rmse_std": (
                statistics.stdev(rmses)
                if len(rmses) > 1 else 0.0
            ),
            "num_seeds": len(runs),
        }

    return summary


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_ablation_results(
    results: Dict[str, List[Dict]],
    summary: Dict[str, Dict],
    output_dir,
):
    """Save detailed and aggregated results."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detailed JSON
    payload = {
        "per_seed_results": results,
        "summary": summary,
    }

    json_path = output_dir / "ablation_results.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)

    # Detailed CSV
    csv_path = output_dir / "ablation_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "variant",
                "seed",
                "mae",
                "rmse",
                "num_evaluated",
            ],
        )

        writer.writeheader()

        for variant, runs in results.items():
            for run in runs:

                writer.writerow({
                    "variant": variant,
                    **run,
                })

    # Summary CSV
    summary_path = output_dir / "ablation_summary.csv"

    with open(
        summary_path, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "variant",
                "mae_mean",
                "mae_std",
                "rmse_mean",
                "rmse_std",
                "num_seeds",
            ],
        )

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
    Run controlled Model B vs QVI ablations.

    Args:
        train_dataset:
            Actual training dataset, e.g. COCO or LVIS.

        val_dataset:
            Actual validation dataset.

        seeds:
            Independent random seeds. Use at least 3.

        cfg:
            Project configuration.

        epochs:
            Number of Model B training epochs.

        batch_size:
            Training batch size.

        learning_rate:
            Learning rate.

        qvi_fine_tune_epochs:
            Optional fine-tuning epochs after QVI-only training.

    Returns:
        Per-seed results and aggregate statistics.
    """

    if cfg is None:
        cfg = Config()

    seeds = tuple(seeds)

    if len(seeds) < 3:
        raise ValueError(
            "Research ablations require at least 3 independent seeds."
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError("Experiment seeds must be unique.")

    if train_dataset is None or val_dataset is None:
        raise ValueError(
            "Real training and validation datasets are required."
        )

    epochs = epochs or getattr(cfg, "epochs", 20)

    batch_size = batch_size or getattr(
        cfg, "batch_size", 4
    )

    learning_rate = learning_rate or getattr(
        cfg, "lr", 1e-4
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

    # -----------------------------------------------------
    # RUN EACH INDEPENDENT SEED
    # -----------------------------------------------------

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

        # -------------------------------------------------
        # 2. TRAIN QVI VARIANTS FROM THE SAME MODEL B
        # -------------------------------------------------

        for mode, variant_name in (
            ("gated", "QVI-gated"),
            ("modulation", "QVI-mod"),
        ):

            print(
                f"\nTraining {variant_name} "
                f"from seed {seed} Model B..."
            )

            # Reinitialize RNG before each variant.
            # This makes the experimental setup reproducible.
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

    output_dir = (
        Path(cfg.output_dir) / "ablation"
    )

    saved_files = save_ablation_results(
        variants,
        summary,
        output_dir,
    )

    print("\nSaved experiment results:")

    for name, path in saved_files.items():
        print(f"{name}: {path}")

    return {
        "per_seed_results": variants,
        "summary": summary,
        "saved_files": saved_files,
    }
