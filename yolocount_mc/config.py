
"""Global configuration for the YOLO-Count multi-class + QVI research pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
import os
import random

import numpy as np
import torch


@dataclass
class Config:
    # ---------------------------------------------------------
    # PROJECT PATHS
    # ---------------------------------------------------------
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
    )

    dataset_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "datasets"
    )

    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "outputs"
    )

    checkpoint_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "checkpoints"
    )

    # ---------------------------------------------------------
    # MODEL CONFIGURATION
    # ---------------------------------------------------------
    img_size: int = 640
    backbone_channels: int = 256
    embed_dim: int = 512

    # Multi-scale feature configuration
    feature_strides: tuple = (8, 16, 32)

    # Number of object classes.
    # Set dynamically from the selected dataset.
    num_classes: int = 80

    # ---------------------------------------------------------
    # QVI CONFIGURATION
    # ---------------------------------------------------------
    use_qvi: bool = False
    qvi_type: str = "modulation"
    qvi_channels: int = 64

    # QVI initialization and regularization
    qvi_dropout: float = 0.1

    # ---------------------------------------------------------
    # BACKBONE / TEXT ENCODER
    # ---------------------------------------------------------
    backbone_name: str = "yolov8"
    pretrained_backbone: bool = True

    text_encoder_name: str = "openai/clip-vit-base-patch32"
    freeze_text_encoder: bool = True

    # ---------------------------------------------------------
    # DATASET CONFIGURATION
    # ---------------------------------------------------------
    dataset_name: str = "coco"

    coco_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
        / "datasets" / "COCO"
    )

    lvis_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
        / "datasets" / "LVIS"
    )

    fsc147_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
        / "datasets" / "FSC147"
    )

    train_split: str = "train"
    val_split: str = "val"

    # ---------------------------------------------------------
    # TRAINING CONFIGURATION
    # ---------------------------------------------------------
    epochs: int = 50
    batch_size: int = 8
    num_workers: int = 4

    learning_rate: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4

    optimizer: str = "adamw"
    scheduler: str = "cosine"

    warmup_epochs: int = 3
    gradient_clip_norm: float = 1.0

    # ---------------------------------------------------------
    # LOSS CONFIGURATION
    # ---------------------------------------------------------
    count_loss_weight: float = 1.0
    density_loss_weight: float = 1.0
    classification_loss_weight: float = 1.0

    # ---------------------------------------------------------
    # GPU / MIXED PRECISION
    # ---------------------------------------------------------
    device: str = field(
        default_factory=lambda: (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    use_amp: bool = True
    amp_dtype: str = "float16"

    pin_memory: bool = True

    # ---------------------------------------------------------
    # VALIDATION / CHECKPOINTING
    # ---------------------------------------------------------
    validate_every: int = 1
    save_every: int = 5

    monitor_metric: str = "mae"
    early_stopping_patience: int = 10

    resume_training: bool = False
    checkpoint_path: str = ""

    # ---------------------------------------------------------
    # REPRODUCIBILITY
    # ---------------------------------------------------------
    seed: int = 42
    deterministic: bool = True

    # ---------------------------------------------------------
    # EXPERIMENT CONFIGURATION
    # ---------------------------------------------------------
    experiment_name: str = "multiclass_yolocount"

    log_interval: int = 20

    def __post_init__(self):
        """Validate configuration and prepare output directories."""

        if self.img_size <= 0:
            raise ValueError("img_size must be positive")

        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.qvi_type not in {
            "none", "gated", "modulation", "attention"
        }:
            raise ValueError(
                f"Unsupported qvi_type: {self.qvi_type}"
            )

        if self.dataset_name.lower() not in {
            "coco", "lvis", "fsc147"
        }:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}"
            )

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but no CUDA GPU is available."
            )

        self.device = torch.device(self.device)

        # AMP is useful only on supported CUDA hardware.
        self.use_amp = bool(self.use_amp and self.device.type == "cuda")

        # Create output folders.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def set_seed(self):
        """Set random seeds for reproducible experiments."""

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_amp_dtype(self):
        """Return the configured automatic mixed precision dtype."""

        if self.amp_dtype == "bfloat16":
            return torch.bfloat16

        return torch.float16

    def summary(self):
        """Print the active experiment configuration."""

        print("\n========== YOLO-COUNT CONFIGURATION ==========")
        print(f"Experiment       : {self.experiment_name}")
        print(f"Dataset          : {self.dataset_name}")
        print(f"Device           : {self.device}")
        print(f"Image size       : {self.img_size}")
        print(f"Embedding dim    : {self.embed_dim}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Batch size       : {self.batch_size}")
        print(f"Epochs           : {self.epochs}")
        print(f"Learning rate    : {self.learning_rate}")
        print(f"AMP enabled      : {self.use_amp}")
        print(f"QVI enabled      : {self.use_qvi}")
        print(f"QVI type         : {self.qvi_type}")
        print("==============================================\n")
