
"""
YOLO-Count Multi-Class + QVI research entry point.

This module provides:
    1. Model shape checks.
    2. Single-query vs multi-query consistency checks.
    3. QVI zero-initialization checks.
    4. Benchmark execution for trained models.
    5. Real-dataset ablation execution.

Training and evaluation are intentionally explicit.
No toy datasets or random training data are used.
"""

from typing import Sequence

import torch

from yolocount_mc.config import Config
from yolocount_mc.models import ModelA, ModelB, ModelC
from yolocount_mc.benchmark import benchmark_A_B_C
from yolocount_mc.ablation import run_ablation


# ---------------------------------------------------------
# MODEL SHAPE CHECK
# ---------------------------------------------------------

@torch.inference_mode()
def test_shapes(
    model,
    image: torch.Tensor,
    class_names: Sequence[str],
):
    """Verify output dimensions for multiple query counts."""

    model.eval()

    if image.ndim != 4:
        raise ValueError("Image must have shape [B,3,H,W].")

    batch_size = image.shape[0]

    for k in (1, 2, 5, 10, 20):

        if k > len(class_names):
            continue

        queries = [
            list(class_names[:k])
            for _ in range(batch_size)
        ]

        counts, density, cls_logits = model(
            image, queries
        )

        assert counts.shape == (batch_size, k), (
            f"Unexpected count shape: {counts.shape}"
        )

        assert density.shape[:2] == (
            batch_size, k
        )

        assert cls_logits.shape[:2] == (
            batch_size, k
        )

        assert torch.isfinite(counts).all()
        assert torch.isfinite(density).all()
        assert torch.isfinite(cls_logits).all()

        assert (density >= 0).all(), (
            "Density predictions must be nonnegative."
        )

    print("Model shape checks: PASSED")


# ---------------------------------------------------------
# SINGLE QUERY VS MULTI-QUERY CONSISTENCY
# ---------------------------------------------------------

@torch.inference_mode()
def test_consistency(
    model,
    image: torch.Tensor,
    class_names: Sequence[str],
    atol: float = 1e-4,
):
    """
    Check whether multi-query inference matches separate
    single-query inference for the same model.

    This check is meaningful when the model shares its
    visual backbone across queries.
    """

    if len(class_names) < 2:
        raise ValueError("At least two class names are required.")

    model.eval()

    queries = [list(class_names[:3])]

    multi_counts, _, _ = model(image, queries)

    separate_counts = []

    for class_name in queries[0]:

        count, _, _ = model(
            image, [[class_name]]
        )

        separate_counts.append(count[:, 0])

    separate_counts = torch.stack(
        separate_counts,
        dim=1,
    )

    if not torch.allclose(
        multi_counts,
        separate_counts,
        atol=atol,
        rtol=1e-4,
    ):
        max_error = (
            multi_counts - separate_counts
        ).abs().max().item()

        raise AssertionError(
            "Multi-query consistency failed. "
            f"Maximum difference: {max_error:.6f}"
        )

    print("Multi-query consistency: PASSED")


# ---------------------------------------------------------
# QVI ZERO INITIALIZATION
# ---------------------------------------------------------

@torch.inference_mode()
def test_qvi_zero_init(
    model_B: ModelB,
    model_C: ModelC,
    image: torch.Tensor,
    class_names: Sequence[str],
    atol: float = 1e-5,
):
    """
    Verify that a newly initialized zero-correction QVI
    model reproduces Model B predictions.

    Model C must have been initialized from the exact
    Model B weights, with zero-initialized QVI outputs.
    """

    if not class_names:
        raise ValueError("At least one class name is required.")

    model_B.eval()
    model_C.eval()

    queries = [list(class_names[:2])]

    counts_B, _, _ = model_B(image, queries)
    counts_C, _, _ = model_C(image, queries)

    if not torch.allclose(
        counts_B,
        counts_C,
        atol=atol,
        rtol=1e-4,
    ):
        max_error = (
            counts_B - counts_C
        ).abs().max().item()

        raise AssertionError(
            "QVI zero-initialization check failed. "
            f"Maximum difference: {max_error:.6f}"
        )

    print("QVI zero-initialization: PASSED")


# ---------------------------------------------------------
# RUN MODEL CHECKS
# ---------------------------------------------------------

def run_model_checks(
    model_B,
    model_C,
    image,
    class_names,
):
    """Run architecture checks on supplied models."""

    test_shapes(
        model_B,
        image,
        class_names,
    )

    test_consistency(
        model_B,
        image,
        class_names,
    )

    test_qvi_zero_init(
        model_B,
        model_C,
        image,
        class_names,
    )

    return True


# ---------------------------------------------------------
# BENCHMARK TRAINED MODELS
# ---------------------------------------------------------

def run_benchmark(
    model_A,
    model_B,
    model_C,
    image,
    class_names,
    Ks=(1, 2, 5, 10, 20),
    warmup=20,
    iterations=100,
):
    """Benchmark three already-trained models."""

    return benchmark_A_B_C(
        model_A=model_A,
        model_B=model_B,
        model_C=model_C,
        class_names=class_names,
        image=image,
        Ks=Ks,
        warmup=warmup,
        iterations=iterations,
    )


# ---------------------------------------------------------
# RUN REAL-DATASET ABLATION
# ---------------------------------------------------------

def run_research_ablation(
    train_dataset,
    val_dataset,
    cfg: Config,
    seeds=(0, 1, 2),
):
    """Run the controlled QVI ablation experiment."""

    return run_ablation(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
        seeds=seeds,
    )


# ---------------------------------------------------------
# ENVIRONMENT INFORMATION
# ---------------------------------------------------------

def print_environment():

    print("\nYOLO-COUNT RESEARCH ENVIRONMENT")
    print("-" * 45)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA version:",
            torch.version.cuda,
        )

    else:
        print("GPU: Not available")

    print("-" * 45)


if __name__ == "__main__":

    print_environment()

    print(
        "\nThis module provides research utilities."
        "\nUse the GPU Jupyter notebook to supply real datasets,"
        "\nload trained checkpoints and execute experiments."
    )
