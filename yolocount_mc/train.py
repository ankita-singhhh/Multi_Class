
"""
Research training pipeline for YOLO-Count.

Model B:
    Multi-class counting baseline trained on real datasets.

Model C:
    Initialized from Model B.
    Stage 1: Freeze shared network and train QVI.
    Stage 2: Optionally fine-tune the complete trainable network.

Training objective:
    Masked count-level MAE.

Important:
    This implementation does not provide spatial density-map
    supervision or spatial classification supervision.
"""

from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .text_encoder import TextEncoder
from .models import ModelB, ModelC
from .dataset import collate


# -------------------------------------------------------------
# DEVICE AND MIXED PRECISION
# -------------------------------------------------------------

def _get_device(cfg):
    device = torch.device(cfg.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    return device


def _get_amp_settings(cfg, device):
    enabled = bool(cfg.use_amp and device.type == "cuda")
    dtype = cfg.get_amp_dtype()

    return enabled, dtype


# -------------------------------------------------------------
# DATA LOADER
# -------------------------------------------------------------

def create_data_loader(
    dataset,
    cfg: Config,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
):
    """Create a DataLoader for a real counting dataset."""

    if dataset is None:
        raise ValueError("A real dataset must be supplied.")

    if batch_size is None:
        batch_size = cfg.batch_size

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    workers = cfg.num_workers

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=(
            cfg.pin_memory and torch.cuda.is_available()
        ),
        persistent_workers=(workers > 0),
        drop_last=False,
    )


# -------------------------------------------------------------
# LOSS
# -------------------------------------------------------------

def masked_count_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Mean absolute count error over valid query positions.

    predictions: [B,K]
    targets:     [B,K]
    mask:        [B,K]
    """

    if predictions.ndim != 2:
        raise ValueError("Predictions must have shape [B,K].")

    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes do not match.")

    if predictions.shape != mask.shape:
        raise ValueError("Query mask shape does not match predictions.")

    if not torch.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN or infinite values.")

    valid_mask = mask.to(
        device=predictions.device,
        dtype=torch.bool,
    )

    if not valid_mask.any():
        raise ValueError("Batch contains no valid query positions.")

    targets = targets.to(
        device=predictions.device,
        dtype=predictions.dtype,
    )

    # Ignore padded target positions safely.
    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]

    if not torch.isfinite(valid_targets).all():
        raise ValueError("Valid targets contain NaN or infinite values.")

    return F.l1_loss(
        valid_predictions,
        valid_targets,
        reduction="mean",
    )


# -------------------------------------------------------------
# CHECKPOINTING
# -------------------------------------------------------------

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    loss,
    path,
    scaler=None,
):
    """Save a complete training checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "scaler_state_dict": (
            scaler.state_dict() if scaler is not None else None
        ),
        "loss": float(loss),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    path,
    device,
    scaler=None,
):
    """Restore model and optional training states."""

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if (
        optimizer is not None
        and checkpoint.get("optimizer_state_dict") is not None
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if (
        scaler is not None
        and checkpoint.get("scaler_state_dict") is not None
    ):
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    return checkpoint.get("epoch", 0) + 1


# -------------------------------------------------------------
# OPTIMIZER
# -------------------------------------------------------------

def build_optimizer(
    model,
    cfg,
    learning_rate,
    backbone_lr=None,
):
    """AdamW with separate backbone and other parameter groups."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if backbone_lr is None:
        backbone_lr = cfg.backbone_lr

    backbone_params = []
    other_params = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if name.startswith("backbone."):
            backbone_params.append(parameter)
        else:
            other_params.append(parameter)

    param_groups = []

    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
        })

    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": learning_rate,
        })

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.weight_decay,
    )


# -------------------------------------------------------------
# FROZEN MODULE HANDLING
# -------------------------------------------------------------

def _keep_frozen_modules_eval(model):
    """
    Keep fully frozen major modules in evaluation mode.

    This prevents frozen BatchNorm running statistics from
    changing during QVI-only training.
    """

    for module_name in ("backbone", "head", "text_enc"):
        module = getattr(model, module_name, None)

        if module is None:
            continue

        has_trainable_parameters = any(
            parameter.requires_grad
            for parameter in module.parameters()
        )

        if not has_trainable_parameters:
            module.eval()


# -------------------------------------------------------------
# EPOCH EXECUTION
# -------------------------------------------------------------

def run_epoch(
    model,
    loader,
    optimizer,
    device,
    cfg,
    scaler,
    training=True,
):
    """Execute one training or validation epoch."""

    if training and optimizer is None:
        raise ValueError("An optimizer is required for training.")

    model.train(training)

    if training:
        _keep_frozen_modules_eval(model)

    amp_enabled, amp_dtype = _get_amp_settings(cfg, device)

    total_loss = 0.0
    total_absolute_error = 0.0
    total_valid_queries = 0

    for step, batch in enumerate(loader):

        images, queries, targets, mask = batch

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):

                predictions, density_maps, cls_logits = model(
                    images,
                    queries,
                )

                loss = masked_count_loss(
                    predictions,
                    targets,
                    mask,
                )

            if training:

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                        and parameter.grad is not None
                    ],
                    max_norm=cfg.gradient_clip_norm,
                )

                scaler.step(optimizer)
                scaler.update()

        valid_mask = mask.to(
            device=predictions.device,
            dtype=torch.bool,
        )

        valid = int(valid_mask.sum().item())

        total_loss += loss.detach().item() * valid

        with torch.no_grad():
            absolute_error = torch.abs(
                predictions.float() - targets.float()
            )

            total_absolute_error += (
                absolute_error[valid_mask].sum().item()
            )

            total_valid_queries += valid

        if training and (step + 1) % cfg.log_interval == 0:
            print(
                f"  Step {step + 1}/{len(loader)} "
                f"Loss: {loss.item():.4f}"
            )

    if total_valid_queries == 0:
        raise RuntimeError("Epoch contained no valid query positions.")

    return {
        "loss": total_loss / total_valid_queries,
        "mae": total_absolute_error / total_valid_queries,
    }


# -------------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------------

def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs,
    cfg,
    stage_name,
    checkpoint_name,
    start_epoch=1,
):
    """Train a model and restore its best checkpoint."""

    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    if len(train_loader) == 0:
        raise ValueError("Training DataLoader is empty.")

    device = _get_device(cfg)

    amp_enabled, amp_dtype = _get_amp_settings(cfg, device)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            amp_enabled and amp_dtype == torch.float16
        ),
    )

    best_metric = float("inf")

    checkpoint_path = (
        Path(cfg.checkpoint_dir) / checkpoint_name
    )

    for epoch in range(start_epoch, epochs + 1):

        print(
            f"\n[{stage_name}] Epoch {epoch}/{epochs}"
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            scaler=scaler,
            training=True,
        )

        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:

            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                cfg=cfg,
                scaler=scaler,
                training=False,
            )

            monitored_metric = val_metrics["mae"]

            print(
                f"Train loss: {train_metrics['loss']:.4f} | "
                f"Train MAE: {train_metrics['mae']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f}"
            )

        else:

            monitored_metric = train_metrics["mae"]

            print(
                f"Train loss: {train_metrics['loss']:.4f} | "
                f"Train MAE: {train_metrics['mae']:.4f}"
            )

        if monitored_metric < best_metric:

            best_metric = monitored_metric

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=monitored_metric,
                path=checkpoint_path,
                scaler=scaler,
            )

            print(f"Best checkpoint saved: {checkpoint_path}")

        if epoch % cfg.save_every == 0:

            periodic_path = (
                Path(cfg.checkpoint_dir)
                / f"{stage_name.lower().replace(' ', '_')}_epoch_{epoch}.pt"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=monitored_metric,
                path=periodic_path,
                scaler=scaler,
            )

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"No best checkpoint was produced: {checkpoint_path}"
        )

    # IMPORTANT: return the best model, not the final epoch model.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"\nRestored best {stage_name} checkpoint "
        f"from epoch {checkpoint['epoch']}."
    )

    return model


# -------------------------------------------------------------
# MODEL B TRAINING
# -------------------------------------------------------------

def train_model_B(
    train_dataset,
    val_dataset=None,
    epochs=None,
    bs=None,
    lr=None,
    cfg=None,
):
    """Train Model B using a real counting dataset."""

    cfg = cfg if cfg is not None else Config()

    cfg.set_seed()

    device = _get_device(cfg)

    epochs = cfg.epochs if epochs is None else epochs
    bs = cfg.batch_size if bs is None else bs
    lr = cfg.learning_rate if lr is None else lr

    if train_dataset is None:
        raise ValueError("A real training dataset is required.")

    print("\nInitializing Model B...")

    encoder = TextEncoder(
        vocab=None,
        d=cfg.embed_dim,
        model_name=cfg.text_encoder_name,
        freeze=cfg.freeze_text_encoder,
    )

    model = ModelB(
        text_enc=encoder,
        cfg=cfg,
    ).to(device)

    train_loader = create_data_loader(
        train_dataset,
        cfg,
        batch_size=bs,
        shuffle=True,
    )

    val_loader = (
        create_data_loader(
            val_dataset,
            cfg,
            batch_size=bs,
            shuffle=False,
        )
        if val_dataset is not None
        else None
    )

    optimizer = build_optimizer(
        model,
        cfg,
        learning_rate=lr,
        backbone_lr=cfg.backbone_lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        cfg=cfg,
        stage_name="Model B",
        checkpoint_name="model_B_best.pt",
    )

    return model


# -------------------------------------------------------------
# MODEL C TRAINING
# -------------------------------------------------------------

def train_model_C(
    model_B,
    train_dataset,
    val_dataset=None,
    epochs=None,
    bs=None,
    lr=None,
    cfg=None,
    fine_tune_epochs=0,
):
    """
    Train Model C in two stages.

    Stage 1:
        Freeze shared modules and train QVI.

    Stage 2:
        Optionally fine-tune the shared network.
    """

    if not isinstance(model_B, ModelB):
        raise TypeError("model_B must be a trained ModelB instance.")

    if train_dataset is None:
        raise ValueError("A real training dataset is required.")

    if fine_tune_epochs < 0:
        raise ValueError("fine_tune_epochs cannot be negative.")

    cfg = cfg if cfg is not None else Config()

    cfg.set_seed()

    device = _get_device(cfg)

    epochs = cfg.epochs if epochs is None else epochs
    bs = cfg.batch_size if bs is None else bs
    lr = 1e-4 if lr is None else lr

    if not cfg.use_qvi:
        cfg_C = replace(
            cfg,
            use_qvi=True,
            qvi_type="modulation",
        )
    else:
        cfg_C = replace(cfg, use_qvi=True)

    print("\nInitializing Model C from Model B...")

    model = ModelC(
        text_enc=model_B.text_enc,
        cfg=cfg_C,
    ).to(device)

    model.initialize_from_model_b(
        model_B,
        strict=True,
    )

    if model.qvi is None:
        raise RuntimeError("Model C was created without QVI.")

    train_loader = create_data_loader(
        train_dataset,
        cfg_C,
        batch_size=bs,
        shuffle=True,
    )

    val_loader = (
        create_data_loader(
            val_dataset,
            cfg_C,
            batch_size=bs,
            shuffle=False,
        )
        if val_dataset is not None
        else None
    )

    # ---------------------------------------------------------
    # STAGE 1: TRAIN QVI ONLY
    # ---------------------------------------------------------

    print("\nStage 1: Training QVI only...")

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.qvi.parameters():
        parameter.requires_grad = True

    optimizer = build_optimizer(
        model,
        cfg_C,
        learning_rate=lr,
        backbone_lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        cfg=cfg_C,
        stage_name="Model C - QVI",
        checkpoint_name="model_C_qvi_best.pt",
    )

    # ---------------------------------------------------------
    # STAGE 2: OPTIONAL FINE-TUNING
    # ---------------------------------------------------------

    if fine_tune_epochs > 0:

        print("\nStage 2: Fine-tuning Model C...")

        for module in (
            model.backbone,
            model.head,
            model.qvi,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

        # Preserve CLIP's configured freezing policy.
        if cfg_C.freeze_text_encoder:

            for parameter in model.text_enc.clip.parameters():
                parameter.requires_grad = False

            # The optional output projection remains trainable.
            for parameter in model.text_enc.output_projection.parameters():
                parameter.requires_grad = True

        else:

            for parameter in model.text_enc.parameters():
                parameter.requires_grad = True

        fine_tune_lr = lr * 0.1

        optimizer = build_optimizer(
            model,
            cfg_C,
            learning_rate=fine_tune_lr,
            backbone_lr=fine_tune_lr * 0.1,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=fine_tune_epochs,
        )

        model = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=fine_tune_epochs,
            cfg=cfg_C,
            stage_name="Model C - Fine-tuning",
            checkpoint_name="model_C_finetuned_best.pt",
        )

    return model
