
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

        # -----------------------------------------------------
        # NORMALIZE PATHS
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # MODEL VALIDATION
        # -----------------------------------------------------
        for name in (
            "img_size",
            "backbone_channels",
            "embed_dim",
            "num_classes",
        ):
            value = getattr(self, name)

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")

        self.feature_strides = tuple(self.feature_strides)

        if not self.feature_strides:
            raise ValueError("feature_strides cannot be empty.")

        if any(
            not isinstance(stride, int)
            or isinstance(stride, bool)
            or stride <= 0
            for stride in self.feature_strides
        ):
            raise ValueError(
                "All feature strides must be positive integers."
            )

        # -----------------------------------------------------
        # QVI VALIDATION
        # -----------------------------------------------------
        supported_qvi_modes = {"none", "gated", "modulation"}

        self.qvi_type = str(self.qvi_type).lower()

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

        if (
            not isinstance(self.qvi_channels, int)
            or isinstance(self.qvi_channels, bool)
            or self.qvi_channels <= 0
        ):
            raise ValueError("qvi_channels must be a positive integer.")

        if (
            not np.isfinite(self.qvi_dropout)
            or not 0.0 <= self.qvi_dropout < 1.0
        ):
            raise ValueError("qvi_dropout must be in [0, 1).")

        # -----------------------------------------------------
        # DATASET VALIDATION
        # -----------------------------------------------------
        self.dataset_name = str(self.dataset_name).lower()

        if self.dataset_name not in {"coco", "lvis", "fsc147"}:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}."
            )

        # -----------------------------------------------------
        # TRAINING VALIDATION
        # -----------------------------------------------------
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")

        if (
            not isinstance(self.num_workers, int)
            or isinstance(self.num_workers, bool)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer.")

        for name in ("learning_rate", "backbone_lr"):
            value = getattr(self, name)

            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")

        if (
            not np.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be finite and non-negative.")

        if (
            not isinstance(self.warmup_epochs, int)
            or isinstance(self.warmup_epochs, bool)
            or self.warmup_epochs < 0
        ):
            raise ValueError("warmup_epochs must be non-negative.")

        if (
            not np.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive and finite."
            )

        self.optimizer = str(self.optimizer).lower()
        self.scheduler = str(self.scheduler).lower()

        if self.optimizer not in {"adamw", "adam", "sgd"}:
            raise ValueError(
                "optimizer must be adamw, adam or sgd."
            )

        if self.scheduler not in {"cosine", "none"}:
            raise ValueError(
                "scheduler must be cosine or none."
            )

        # -----------------------------------------------------
        # LOSS VALIDATION
        # -----------------------------------------------------
        loss_weights = (
            self.count_loss_weight,
            self.density_loss_weight,
            self.classification_loss_weight,
        )

        if any(
            not np.isfinite(weight) or weight < 0
            for weight in loss_weights
        ):
            raise ValueError(
                "Loss weights must be finite and non-negative."
            )

        if not any(weight > 0 for weight in loss_weights):
            raise ValueError(
                "At least one loss weight must be positive."
            )

        # -----------------------------------------------------
        # DEVICE VALIDATION
        # -----------------------------------------------------
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

        if (
            self.device.type == "mps"
            and not torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is unavailable.")

        # -----------------------------------------------------
        # MIXED PRECISION VALIDATION
        # -----------------------------------------------------
        self.amp_dtype = str(self.amp_dtype).lower()

        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError(
                "amp_dtype must be float16 or bfloat16."
            )

        if not isinstance(self.use_amp, bool):
            raise TypeError("use_amp must be a boolean.")

        self.use_amp = self.use_amp and self.device.type == "cuda"

        if (
            self.use_amp
            and self.amp_dtype == "bfloat16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "bfloat16 AMP was requested, but this CUDA GPU "
                "does not report bfloat16 support. Use float16 instead."
            )

        # -----------------------------------------------------
        # CHECKPOINT / VALIDATION CONFIGURATION
        # -----------------------------------------------------
        for name in ("validate_every", "save_every", "log_interval"):
            value = getattr(self, name)

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")

        if (
            not isinstance(self.early_stopping_patience, int)
            or isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience < 0
        ):
            raise ValueError(
                "early_stopping_patience must be non-negative."
            )

        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer.")

        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be a boolean.")

        if not isinstance(self.pin_memory, bool):
            raise TypeError("pin_memory must be a boolean.")

        if not isinstance(self.use_qvi, bool):
            raise TypeError("use_qvi must be a boolean.")

        if not isinstance(self.freeze_text_encoder, bool):
            raise TypeError("freeze_text_encoder must be a boolean.")

        if not isinstance(self.pretrained_backbone, bool):
            raise TypeError("pretrained_backbone must be a boolean.")

        if not isinstance(self.resume_training, bool):
            raise TypeError("resume_training must be a boolean.")

        # -----------------------------------------------------
        # CREATE OUTPUT FOLDERS
        # -----------------------------------------------------
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def lr(self) -> float:
        """
        Backward-compatible alias for learning_rate.

        Prefer using learning_rate in new code.
        """
        return self.learning_rate

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
        print(f"Backbone LR      : {self.backbone_lr}")
        print(f"AMP enabled      : {self.use_amp}")
        print(f"AMP dtype        : {self.amp_dtype}")
        print(f"QVI enabled      : {self.use_qvi}")
        print(f"QVI type         : {self.qvi_type}")
        print(f"Deterministic    : {self.deterministic}")
        print("==============================================\n")
