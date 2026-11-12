
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
    - Peak GPU memory (MB)

Important:
    Pass trained models and real class names.
    This module does not initialize or train models.
"""

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


DEFAULT_QUERY_COUNTS = (1, 2, 5, 10, 20)


def _synchronize(device):
    """Synchronize CUDA before and after timing."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prepare_queries(class_names: Sequence[str], k: int):
    """Prepare a single-image batch containing K distinct queries."""

    if k > len(class_names):
        raise ValueError(
            f"Requested K={k}, but only "
            f"{len(class_names)} class names were supplied."
        )

    selected = list(class_names[:k])

    if len(set(selected)) != len(selected):
        raise ValueError("Benchmark class names must be unique.")

    return [selected]


def _run_model_a(model, image, queries):
    """
    Model A requires one forward pass per query.

    Returns the concatenated class counts.
    """

    predictions = []

    for query in queries[0]:
        count, _, _ = model(image, [[query]])
        predictions.append(count)

    return torch.cat(predictions, dim=1)


def _run_model_b_or_c(model, image, queries):
    """Models B and C process all queries in one forward pass."""

    counts, _, _ = model(image, queries)
    return counts


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
    Benchmark one model at a specified query count.

    Args:
        model:
            Already-trained model in evaluation mode.

        image:
            Real or representative image tensor [1,3,H,W].

        class_names:
            Distinct class names from the evaluation dataset.

        k:
            Number of class queries.

        model_name:
            Model identifier.

        warmup:
            Number of warm-up iterations.

        iterations:
            Number of timed iterations.

        model_a:
            Whether to execute K separate Model A passes.

    Returns:
        Dictionary containing latency, throughput and memory.
    """

    if warmup < 0 or iterations <= 0:
        raise ValueError("Invalid warmup or iteration count.")

    device = image.device

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Benchmark image must have shape [1,3,H,W].")

    if image.shape[1] != 3:
        raise ValueError("Benchmark image must have 3 RGB channels.")

    if len(class_names) < k:
        raise ValueError("Not enough class names for requested K.")

    queries = _prepare_queries(class_names, k)

    model.eval()

    if model_a:
        forward = lambda: _run_model_a(model, image, queries)
    else:
        forward = lambda: _run_model_b_or_c(model, image, queries)

    # ---------------------------------------------------------
    # WARM-UP
    # ---------------------------------------------------------
    for _ in range(warmup):
        forward()

    _synchronize(device)

    # ---------------------------------------------------------
    # RESET GPU MEMORY STATISTICS
    # ---------------------------------------------------------
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---------------------------------------------------------
    # LATENCY MEASUREMENT
    # ---------------------------------------------------------
    latencies_ms = []

    for _ in range(iterations):

        _synchronize(device)

        start = time.perf_counter()

        forward()

        _synchronize(device)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        latencies_ms.append(elapsed_ms)

    latency_array = np.asarray(latencies_ms)

    mean_latency = float(np.mean(latency_array))
    median_latency = float(np.median(latency_array))
    p95_latency = float(np.percentile(latency_array, 95))

    # ---------------------------------------------------------
    # THROUGHPUT
    # ---------------------------------------------------------
    images_per_second = 1000.0 / mean_latency

    queries_per_second = (
        images_per_second * k
    )

    # ---------------------------------------------------------
    # GPU MEMORY
    # ---------------------------------------------------------
    if device.type == "cuda":

        _synchronize(device)

        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

        reserved_memory_mb = (
            torch.cuda.max_memory_reserved(device)
            / (1024 ** 2)
        )

    else:
        peak_memory_mb = float("nan")
        reserved_memory_mb = float("nan")

    return {
        "model": model_name,
        "K": k,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "images_per_second": images_per_second,
        "queries_per_second": queries_per_second,
        "peak_memory_mb": peak_memory_mb,
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

    Supply a representative image when possible. If image is
    omitted, a zero-valued input is created for shape-only testing.
    """

    device = torch.device(device)

    if image is None:
        image = torch.zeros(
            1, 3, img_size, img_size,
            device=device,
        )
    else:
        image = image.to(device)

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
    """

    models = [
        ("A", model_A, True),
        ("B", model_B, False),
        ("C", model_C, False),
    ]

    device = image.device

    print("\nYOLO-COUNT INFERENCE BENCHMARK")
    print("=" * 105)

    results = []

    for name, model, is_model_a in models:

        if model is None:
            continue

        model.eval()

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
    )

    print(header)
    print("-" * 105)

    for row in results:

        memory = row["peak_memory_mb"]

        memory_str = (
            f"{memory:.1f}"
            if np.isfinite(memory)
            else "N/A"
        )

        print(
            f"{row['model']:<8}"
            f"{row['K']:>5}"
            f"{row['mean_latency_ms']:>12.2f}"
            f"{row['median_latency_ms']:>13.2f}"
            f"{row['p95_latency_ms']:>11.2f}"
            f"{row['images_per_second']:>12.2f}"
            f"{row['queries_per_second']:>12.2f}"
            f"{memory_str:>12}"
        )

    print("=" * 105)

    return results
