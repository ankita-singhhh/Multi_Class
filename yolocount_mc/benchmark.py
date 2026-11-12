
"""
Research benchmarking for YOLO-Count Models A, B and C.

Compares:
    Model A: K separate single-query forward passes.
    Model B: K queries in one shared-backbone forward pass.
    Model C: K queries in one shared-backbone + QVI pass.

Metrics:
    - Mean inference latency (ms)
    - Median inference latency (ms)
    - P95 inference latency (ms)
    - Image throughput (images/second)
    - Query throughput (queries/second)
    - Peak total GPU memory (MB)
    - Incremental peak GPU memory (MB)
    - Reserved GPU memory (MB)

Important:
    Pass trained models, real class names and a representative image.
    This module does not initialize or train models.
    Input transfer is excluded from latency measurements.
"""

import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch


DEFAULT_QUERY_COUNTS = (1, 2, 5, 10, 20)


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA operations before or after timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_class_names(class_names: Sequence[str]) -> None:
    if not class_names:
        raise ValueError("class_names cannot be empty.")

    if any(not isinstance(name, str) or not name.strip()
           for name in class_names):
        raise ValueError("Every class name must be a non-empty string.")

    if len(set(name.strip() for name in class_names)) != len(class_names):
        raise ValueError("Benchmark class names must be unique.")


def _prepare_queries(class_names: Sequence[str], k: int):
    """Prepare one image batch containing K distinct class queries."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("K must be a positive integer.")

    if k > len(class_names):
        raise ValueError(
            f"Requested K={k}, but only {len(class_names)} "
            "class names were supplied."
        )

    return [list(class_names[:k])]


def _validate_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Image must have shape [1, 3, H, W].")

    if image.shape[1] != 3:
        raise ValueError("Image must have 3 RGB channels.")

    if image.shape[2] <= 0 or image.shape[3] <= 0:
        raise ValueError("Image height and width must be positive.")

    if not image.is_floating_point():
        raise TypeError("Image tensor must have a floating-point dtype.")

    if not torch.isfinite(image).all().item():
        raise ValueError("Image contains NaN or infinite values.")


def _validate_model_device(model, device: torch.device) -> None:
    """Ensure model parameters and buffers are on the benchmark device."""
    tensors = list(model.parameters()) + list(model.buffers())

    if not tensors:
        raise ValueError("Model has no parameters or buffers to validate.")

    mismatched = [
        tensor.device for tensor in tensors
        if tensor.device != device
    ]

    if mismatched:
        raise ValueError(
            f"Model tensors must be on {device}; found {mismatched[0]}."
        )


def _run_model_a(model, image, queries):
    """
    Model A executes one forward pass per query.

    Returns concatenated counts with shape [1, K].
    """
    predictions = []

    for query in queries[0]:
        counts, _, _ = model(image, [[query]])
        predictions.append(counts)

    return torch.cat(predictions, dim=1)


def _run_model_b_or_c(model, image, queries):
    """Models B and C process all K queries in one forward pass."""
    counts, _, _ = model(image, queries)
    return counts


def _validate_counts(counts: torch.Tensor, k: int) -> None:
    if not isinstance(counts, torch.Tensor):
        raise TypeError("Model must return counts as a torch.Tensor.")

    if counts.shape != (1, k):
        raise ValueError(
            f"Expected count output shape [1, {k}], "
            f"received {tuple(counts.shape)}."
        )

    if not torch.isfinite(counts).all().item():
        raise ValueError("Model produced NaN or infinite counts.")


@torch.inference_mode()
def benchmark_single(
    model,
    image: torch.Tensor,
    class_names: Sequence[str],
    k: int,
    model_name: str,
    warmup: int = 20,
    iterations: int = 100,
    model_a: bool = False,
) -> Dict:
    """
    Benchmark one trained model at a specified query count.

    The image must already be on the model's device.
    Data transfer is excluded from the timing.

    Returns:
        Dictionary containing latency, throughput and memory metrics.
    """
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer.")

    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer.")

    _validate_class_names(class_names)
    _validate_image(image)

    device = image.device

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA image supplied but CUDA is unavailable.")

    _validate_model_device(model, device)

    queries = _prepare_queries(class_names, k)

    model.eval()

    if model_a:
        def forward():
            counts = _run_model_a(model, image, queries)
            _validate_counts(counts, k)
            return counts
    else:
        def forward():
            counts = _run_model_b_or_c(model, image, queries)
            _validate_counts(counts, k)
            return counts

    # ---------------------------------------------------------
    # WARM-UP
    # ---------------------------------------------------------
    for _ in range(warmup):
        forward()

    _synchronize(device)

    # ---------------------------------------------------------
    # MEMORY BASELINE
    # ---------------------------------------------------------
    if device.type == "cuda":
        baseline_allocated = torch.cuda.memory_allocated(device)

        torch.cuda.reset_peak_memory_stats(device)

    else:
        baseline_allocated = 0

    # ---------------------------------------------------------
    # LATENCY MEASUREMENT
    # ---------------------------------------------------------
    latencies_ms = []

    for _ in range(iterations):
        _synchronize(device)

        start = time.perf_counter()

        forward()

        _synchronize(device)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    latency_array = np.asarray(latencies_ms, dtype=np.float64)

    mean_latency = float(np.mean(latency_array))
    median_latency = float(np.median(latency_array))
    p95_latency = float(np.percentile(latency_array, 95))

    # ---------------------------------------------------------
    # THROUGHPUT
    # ---------------------------------------------------------
    images_per_second = 1000.0 / mean_latency
    queries_per_second = (1000.0 * k) / mean_latency

    # ---------------------------------------------------------
    # GPU MEMORY
    # ---------------------------------------------------------
    if device.type == "cuda":
        _synchronize(device)

        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        peak_memory_mb = peak_allocated / (1024 ** 2)

        incremental_peak_memory_mb = max(
            0,
            peak_allocated - baseline_allocated
        ) / (1024 ** 2)

        reserved_memory_mb = peak_reserved / (1024 ** 2)

    else:
        peak_memory_mb = float("nan")
        incremental_peak_memory_mb = float("nan")
        reserved_memory_mb = float("nan")

    return {
        "model": model_name,
        "K": k,
        "image_height": int(image.shape[2]),
        "image_width": int(image.shape[3]),
        "device": str(device),
        "warmup": warmup,
        "iterations": iterations,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "images_per_second": images_per_second,
        "queries_per_second": queries_per_second,
        "peak_memory_mb": peak_memory_mb,
        "incremental_peak_memory_mb": incremental_peak_memory_mb,
        "reserved_memory_mb": reserved_memory_mb,
    }


@torch.inference_mode()
def benchmark(
    model,
    class_names: Sequence[str],
    Ks=DEFAULT_QUERY_COUNTS,
    img_size: int = 640,
    device: str = "cuda",
    warmup: int = 20,
    iterations: int = 100,
    model_name: str = "Model",
    model_a: bool = False,
    image: Optional[torch.Tensor] = None,
):
    """
    Benchmark a trained model over multiple query counts.

    A real representative image is required. The img_size argument
    is retained for API compatibility; image dimensions come from
    the supplied image.
    """
    if image is None:
        raise ValueError(
            "A real representative image is required. "
            "Pass image=... instead of benchmarking a zero tensor."
        )

    target_device = torch.device(device)

    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    image = image.to(target_device)

    _validate_image(image)
    _validate_class_names(class_names)

    Ks = list(Ks)

    if not Ks:
        raise ValueError("Ks cannot be empty.")

    if len(set(Ks)) != len(Ks):
        raise ValueError("Ks must not contain duplicate query counts.")

    rows = []

    for k in Ks:
        result = benchmark_single(
            model=model,
            image=image,
            class_names=class_names,
            k=k,
            model_name=model_name,
            warmup=warmup,
            iterations=iterations,
            model_a=model_a,
        )

        rows.append(result)

    return rows


def benchmark_A_B_C(
    model_A,
    model_B,
    model_C,
    class_names: Sequence[str],
    image: torch.Tensor,
    Ks=DEFAULT_QUERY_COUNTS,
    warmup: int = 20,
    iterations: int = 100,
):
    """
    Compare already-trained Models A, B and C.

    All models must be loaded with trained weights.
    The same image and class-name ordering are used for every model.
    """
    _validate_image(image)
    _validate_class_names(class_names)

    models = [
        ("A", model_A, True),
        ("B", model_B, False),
        ("C", model_C, False),
    ]

    device = image.device

    print("\nYOLO-COUNT INFERENCE BENCHMARK")
    print("=" * 132)

    results = []

    for name, model, is_model_a in models:
        if model is None:
            continue

        rows = benchmark(
            model=model,
            class_names=class_names,
            Ks=Ks,
            device=str(device),
            warmup=warmup,
            iterations=iterations,
            model_name=name,
            model_a=is_model_a,
            image=image,
        )

        results.extend(rows)

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------
    header = (
        f"{'Model':<8}"
        f"{'K':>5}"
        f"{'Mean(ms)':>12}"
        f"{'Median(ms)':>13}"
        f"{'P95(ms)':>11}"
        f"{'Images/s':>12}"
        f"{'Queries/s':>12}"
        f"{'Peak MB':>12}"
        f"{'Extra MB':>12}"
    )

    print(header)
    print("-" * 132)

    for row in results:
        peak = row["peak_memory_mb"]
        extra = row["incremental_peak_memory_mb"]

        peak_str = f"{peak:.1f}" if np.isfinite(peak) else "N/A"
        extra_str = f"{extra:.1f}" if np.isfinite(extra) else "N/A"

        print(
            f"{row['model']:<8}"
            f"{row['K']:>5}"
            f"{row['mean_latency_ms']:>12.2f}"
            f"{row['median_latency_ms']:>13.2f}"
            f"{row['p95_latency_ms']:>11.2f}"
            f"{row['images_per_second']:>12.2f}"
            f"{row['queries_per_second']:>12.2f}"
            f"{peak_str:>12}"
            f"{extra_str:>12}"
        )

    print("=" * 132)

    return results
