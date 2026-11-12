
# YOLO-Count: Multi-Class Counting with Query-Visual Interaction (QVI)

An independent research implementation exploring multi-class visual counting with shared visual features, text-conditioned class queries, and Query-Visual Interaction (QVI).

The project investigates whether multiple object categories can be counted efficiently in a single forward pass and whether query-visual interaction improves counting accuracy.

> **Research Status:** This repository is under active development. Architecture, dataset adapters, density supervision, and training workflows must be validated before experimental results are reported.

## Architecture

The project studies three model variants:

| Model | Description |
|---|---|
| **Model A** | Single-query counting baseline: one class query per forward pass. |
| **Model B** | Multi-class counting: processes K class queries using shared visual features. |
| **Model C** | Model B enhanced with Query-Visual Interaction (QVI). |

### Model Architecture

```text
                         Input Image
                              |
                              v
                       YOLO Backbone
                              |
                              v
                        PAFPN Features
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
              Text Encoder          Visual Features
                  |
                  v
             K Class Embeddings
                  |
                  v
          Multi-Class Counting Head
                  |
                  v
             Model B Outputs
                  |
                  v
               QVI Module
                  |
                  v
             Model C Outputs
```

Model A is the single-query reference. Model B introduces shared multi-class visual processing. Model C investigates query-conditioned visual refinement.

## Repository Structure

```text
Multi_Class/
│
├── yolocount_mc/
│   ├── __init__.py
│   ├── config.py
│   ├── backbone.py
│   ├── text_encoder.py
│   ├── dataset.py
│   ├── head.py
│   ├── qvi.py
│   ├── models.py
│   ├── yolocount.py
│   ├── train.py
│   ├── validation.py
│   ├── benchmark.py
│   └── ablation.py
│
├── tests/
│   └── test_models.py
│
├── notebooks/
│   └── multi_class_training.ipynb
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Core Components

| File | Responsibility |
|---|---|
| `backbone.py` | YOLO-based visual feature extraction and multi-scale feature fusion |
| `text_encoder.py` | Converts class-name queries into text embeddings using CLIP |
| `head.py` | Produces class-conditioned density and classification maps |
| `models.py` | Implements Models A, B, and C |
| `qvi.py` | Implements Query-Visual Interaction |
| `dataset.py` | Dataset loading, category queries, and batching |
| `yolocount.py` | Counting utilities and prediction aggregation |
| `train.py` | Training loops, optimization, and checkpoint management |
| `validation.py` | Counting evaluation using MAE and RMSE |
| `benchmark.py` | Inference latency, throughput, and GPU memory comparison |
| `ablation.py` | Multi-seed QVI ablation experiments |
| `config.py` | Model, dataset, and training configuration |

## Tensor Interface

### Inputs

- Images: `[B, 3, H, W]`
- Class queries: K category names per image.

### Outputs

| Output | Shape |
|---|---|
| Predicted counts | `[B, K]` |
| Density maps | `[B, K, Hf, Wf]` |
| Classification logits | `[B, K, Hf, Wf]` |

Where:

- B = Batch size
- K = Number of queried classes
- Hf and Wf = Output feature-map dimensions

Padded queries must be excluded from training losses and evaluation metrics using the query mask.

## Datasets

The research plan considers real annotated datasets. Each dataset requires its own annotation interpretation and validation rules.

### 1. COCO Dataset

COCO instance annotations contain object categories and bounding boxes from which image-level instance counts can be derived.

Planned experiments:

- Multi-class counting.
- Per-category evaluation.
- Comparison of Models A, B, and C.

### 2. LVIS Dataset

LVIS provides a large and diverse category vocabulary.

Its metadata distinguishes positive, negative, and not-exhaustively annotated categories. These distinctions must be respected when selecting valid class queries and computing losses and metrics.

### 3. FSC147 Dataset

FSC147 is a class-agnostic counting dataset with point annotations and exemplar information.

It requires a dedicated adapter and target-generation strategy. Full FSC147 support is not claimed until the official annotation format has been implemented and validated.

> Dataset compatibility does not imply that all datasets share the same annotation format or supervision strategy.

## Training Strategy

The intended experimental workflow is:

1. Configure the dataset and experiment settings.
2. Initialize the pretrained visual backbone.
3. Initialize the text encoder.
4. Train Model B using real training data.
5. Initialize Model C from the corresponding trained Model B checkpoint.
6. Train QVI using the same training and validation splits.
7. Restore the best validation checkpoints.
8. Evaluate all variants under identical evaluation protocols.

Model A serves as a single-query computational baseline.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| MAE | Average absolute counting error |
| RMSE | Penalizes larger counting errors |
| Inference Latency | Time required per image |
| Images/Second | Image processing throughput |
| Queries/Second | Class-query processing throughput |
| Peak GPU Memory | Maximum allocated GPU memory |

For fair comparisons, keep validation images, class queries, image resolution, GPU hardware, precision settings, and timing protocol consistent.

## QVI Ablation Experiments

The planned controlled ablation compares:

| Variant | Description |
|---|---|
| B | Multi-class baseline without QVI |
| QVI-Gated | Model B with gated QVI |
| QVI-Modulation | Model B with modulation-based QVI |

Use at least three independent random seeds and report per-seed metrics, mean, and standard deviation.

```text
Mean MAE ± Standard Deviation
Mean RMSE ± Standard Deviation
```

A lower mean MAE alone is not sufficient evidence that QVI improves performance. Results must be interpreted in light of seed-to-seed variability and consistent experimental conditions.

## Installation

Clone this repository:

```bash
git clone https://github.com/ankita-singhhh/Multi_Class.git

cd Multi_Class
```

Install the project and dependencies:

```bash
pip install -r requirements.txt

pip install -e .
```

For GPU experiments, install a PyTorch build compatible with the CUDA runtime provided by the training platform.

## Testing

Run the architecture unit tests:

```bash
pytest -q
```

The intended unit tests use lightweight mock components to validate model contracts without downloading pretrained weights.

Real YOLO, CLIP, dataset, and GPU integration tests must be performed separately in the GPU notebook.

## Benchmarking

The benchmark compares already-trained Models A, B, and C for different query counts:

```python
Ks = (1, 2, 5, 10, 20)
```

It is designed to include:

- Warm-up iterations.
- CUDA synchronization.
- Mean inference latency.
- Median inference latency.
- P95 inference latency.
- Image throughput.
- Query throughput.
- Peak GPU memory.

Benchmark results should be collected from trained checkpoints on the same hardware and under the same precision and input conditions.

Randomly initialized models are not research results.

## Experimental Environment

Training is intended to run in a GPU-enabled Jupyter environment.

The notebook workflow will cover:

- GPU and CUDA verification.
- Dataset preparation and validation.
- Model initialization.
- Training and validation.
- Checkpoint management.
- Benchmarking.
- Ablation experiments.
- Result export.

Source-code development and architecture changes are maintained in VS Code.

## Research Scope and Current Limitations

This is an evolving research implementation.

The following must be independently implemented or verified before experimental claims are made:

- Compatibility with the selected Ultralytics YOLO architecture and feature taps.
- Pretrained CLIP integration and query batching.
- Correct COCO and LVIS annotation handling.
- Density-target generation and spatial supervision.
- Training stability and best-checkpoint restoration.
- FSC147-specific annotation support.
- Reproducible multi-seed experiments.

A nonnegative predicted density map does not, by itself, establish spatially supervised density estimation. Density targets and their associated losses must be implemented and validated before claiming density-supervised training.

## Research Objective

**Can shared visual computation and Query-Visual Interaction improve the efficiency and accuracy of multi-class object counting compared with processing each class independently?**

The project aims to investigate this question through controlled experiments, real datasets, reproducible training, and quantitative comparisons.

---

## Repository

**GitHub:** https://github.com/ankita-singhhh/Multi_Class

**Owner:** ankita-singhhh

**Project:** Multi_Class

**Primary Branch:** main

**Research Focus:** Multi-Class Object Counting + Query-Visual Interaction
