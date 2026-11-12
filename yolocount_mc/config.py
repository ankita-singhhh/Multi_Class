
"""Global configuration for the YOLO-Count multi-class + QVI pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
import random

import numpy as np
import torch


def get_project_root() -> Path:
    """Return the repository root (parent of yolocount_mc)."""
    return Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---------------------------------------------------------
    # PROJECT PATHS
    # ---------------------------------------------------------
    project_root: Path = field(default_factory=get_project_root)

    dataset_root: Path = field(
        default_factory=lambda: get_project_root() / "datasets"
    )

    output_dir: Path = field(
        default_factory=lambda: get_project_root() / "outputs"
    )

    checkpoint_dir: Path = field(
        default_factory=lambda: get_project_root() / "checkpoints"
    )

    # ---------------------------------------------------------
    # MODEL CONFIGURATION
    # ---------------------------------------------------------
    img_size: int = 640
    backbone_channels: int = 256
    embed_dim: int = 512

    feature_strides: tuple = (8, 16, 32)

    # Dataset category count; actual query count can vary per image.
    num_classes: int = 80

    # ---------------------------------------------------------
    # QVI CONFIGURATION
    # ---------------------------------------------------------
    use_qvi: bool = False
    qvi_type: str = "modulation"
    qvi_channels: int = 64
    qvi_dropout: float = 0.1

    # ---------------------------------------------------------
    # BACKBONE / TEXT ENCODER
    # ---------------------------------------------------------
    backbone_name: str = "yolov8"
    backbone_weights: str = "yolov8n.pt"
    pretrained_backbone: bool = True

    text_encoder_name: str = "openai/clip-vit-base-patch32"
    freeze_text_encoder: bool = True

    # ---------------------------------------------------------
    # DATASET CONFIGURATION
    # ---------------------------------------------------------
    dataset_name: str = "coco"

    coco_root: Path = field(
        default_factory=lambda: get_project_root()
        / "datasets" / "COCO"
    )

    lvis_root: Path = field(
        default_factory=lambda: get_project_root()
        / "datasets" / "LVIS"
    )

    fsc147_root: Path = field(
        default_factory=lambda: get_project_root()
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

        # Normalize paths.
        path_fields = (
            "project_root",
            "dataset_root",
            "output_dir",
            "checkpoint_dir",
            "coco_root",
            "lvis_root",
            "fsc147_root",
        )

        for name in path_fields:
            setattr(self, name, Path(getattr(self, name)))

        # Basic model validation.
        if self.img_size <= 0:
            raise ValueError("img_size must be positive.")

        if self.backbone_channels <= 0:
            raise ValueError("backbone_channels must be positive.")

        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")

        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        if any(stride <= 0 for stride in self.feature_strides):
            raise ValueError("All feature strides must be positive.")

        # QVI validation.
        supported_qvi_modes = {
            "none",
            "gated",
            "modulation",
        }

        if self.qvi_type not in supported_qvi_modes:
            raise ValueError(
                f"Unsupported qvi_type: {self.qvi_type}. "
                f"Choose from {sorted(supported_qvi_modes)}."
            )

        if self.use_qvi and self.qvi_type == "none":
            raise ValueError(
                "use_qvi=True requires qvi_type='gated' "
                "or qvi_type='modulation'."
            )

        if self.qvi_channels <= 0:
            raise ValueError("qvi_channels must be positive.")

        if not 0.0 <= self.qvi_dropout < 1.0:
            raise ValueError("qvi_dropout must be in [0, 1).")

        # Dataset validation.
        self.dataset_name = self.dataset_name.lower()

        if self.dataset_name not in {"coco", "lvis", "fsc147"}:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}."
            )

        # Training validation.
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        if self.learning_rate <= 0 or self.backbone_lr <= 0:
            raise ValueError("Learning rates must be positive.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")

        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative.")

        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")

        if self.optimizer.lower() not in {"adamw", "adam", "sgd"}:
            raise ValueError(
                "optimizer must be adamw, adam or sgd."
            )

        if self.scheduler.lower() not in {"cosine", "none"}:
            raise ValueError(
                "scheduler must be cosine or none."
            )

        # Loss validation.
        loss_weights = (
            self.count_loss_weight,
            self.density_loss_weight,
            self.classification_loss_weight,
        )

        if any(weight < 0 for weight in loss_weights):
            raise ValueError("Loss weights cannot be negative.")

        if not any(weight > 0 for weight in loss_weights):
            raise ValueError("At least one loss weight must be positive.")

        # Device validation.
        self.device = torch.device(self.device)

        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but no CUDA GPU is available."
                )

            if (
                self.device.index is not None
                and self.device.index >= torch.cuda.device_count()
            ):
                raise RuntimeError(
                    f"CUDA device {self.device.index} does not exist."
                )

        elif self.device.type not in {"cpu", "mps"}:
            raise ValueError(
                f"Unsupported device type: {self.device.type}."
            )

        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")

        # Mixed precision validation.
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError(
                "amp_dtype must be float16 or bfloat16."
            )

        self.use_amp = bool(
            self.use_amp and self.device.type == "cuda"
        )

        # Checkpoint and validation configuration.
        if self.validate_every <= 0:
            raise ValueError("validate_every must be positive.")

        if self.save_every <= 0:
            raise ValueError("save_every must be positive.")

        if self.early_stopping_patience < 0:
            raise ValueError(
                "early_stopping_patience cannot be negative."
            )

        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive.")

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

        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def get_amp_dtype(self):
        """Return the configured automatic mixed precision dtype."""

        if self.amp_dtype == "bfloat16":
            return torch.bfloat16

        return torch.float16

    def summary(self):
        """Print the active experiment configuration."""

        print("\n========== YOLO-COUNT CONFIGURATION ==========")
        print(f"Project root     : {self.project_root}")
        print(f"Dataset          : {self.dataset_name}")
        print(f"Dataset root     : {self.dataset_root}")
        print(f"Output directory : {self.output_dir}")
        print(f"Checkpoint dir   : {self.checkpoint_dir}")
        print(f"Device           : {self.device}")
        print(f"Image size       : {self.img_size}")
        print(f"Embedding dim    : {self.embed_dim}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Batch size       : {self.batch_size}")
        print(f"Epochs           : {self.epochs}")
        print(f"Learning rate    : {self.learning_rate}")
        print(f"AMP enabled      : {self.use_amp}")
        print(f"AMP dtype        : {self.amp_dtype}")
        print(f"QVI enabled      : {self.use_qvi}")
        print(f"QVI type         : {self.qvi_type}")
        print("==============================================\n")
