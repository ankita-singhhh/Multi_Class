"""
Research training pipeline for YOLO-Count.

Model B: multi-class counting baseline trained on real datasets.
Model C: initialized from Model B; Stage 1 trains QVI only,
Stage 2 optionally fine-tunes the configured trainable network.

Objective: masked count-level MAE only.
No spatial density-map or spatial classification supervision is implemented.
"""

from dataclasses import replace
from pathlib import Path
from typing import Optional
import math
import os

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

def _get_device(cfg) -> torch.device:
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return device


def _get_amp_settings(cfg, device):
    return bool(cfg.use_amp and device.type == "cuda"), cfg.get_amp_dtype()


def _validate_training_arguments(epochs, batch_size, learning_rate):
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    if (
        not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be a finite positive number.")


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

    batch_size = cfg.batch_size if batch_size is None else batch_size
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    workers = int(cfg.num_workers)
    if workers < 0:
        raise ValueError("cfg.num_workers cannot be negative.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=bool(cfg.pin_memory and torch.cuda.is_available()),
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
    """Mean absolute count error over valid [B,K] query positions."""
    if not all(isinstance(x, torch.Tensor) for x in (predictions, targets, mask)):
        raise TypeError("predictions, targets, and mask must be tensors.")
    if predictions.ndim != 2:
        raise ValueError("Predictions must have shape [B,K].")
    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes do not match.")
    if predictions.shape != mask.shape:
        raise ValueError("Query mask shape does not match predictions.")
    if not predictions.is_floating_point():
        raise TypeError("Predictions must use a floating-point dtype.")

    if mask.dtype != torch.bool:
        if mask.is_floating_point() and not bool(torch.isfinite(mask).all().item()):
            raise ValueError("Query mask contains NaN or infinite values.")
        binary_mask = (mask == 0) | (mask == 1)
        if not bool(binary_mask.all().item()):
            raise ValueError("Query mask must contain only 0 and 1.")

    valid_mask = mask.to(device=predictions.device, dtype=torch.bool)
    if not bool(valid_mask.any().item()):
        raise ValueError("Batch contains no valid query positions.")

    targets = targets.to(device=predictions.device, dtype=predictions.dtype)
    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]

    if not bool(torch.isfinite(valid_predictions).all().item()):
        raise ValueError("Valid predictions contain NaN or infinite values.")
    if not bool(torch.isfinite(valid_targets).all().item()):
        raise ValueError("Valid targets contain NaN or infinite values.")

    return F.l1_loss(valid_predictions, valid_targets, reduction="mean")


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
    """Save model and optional optimizer/scheduler/scaler states."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "loss": float(loss),
    }

    # Atomic replacement avoids leaving a partially written checkpoint.
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    path,
    device,
    scaler=None,
):
    """
    Restore model and optional training states; return the next epoch.

    A complete resume workflow must additionally restore RNG and sampler
    state and configure the caller's epoch/scheduler loop consistently.
    """
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return int(checkpoint.get("epoch", 0)) + 1


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
    for value, name in (
        (learning_rate, "learning_rate"),
        (cfg.backbone_lr if backbone_lr is None else backbone_lr, "backbone_lr"),
    ):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number.")

    backbone_lr = cfg.backbone_lr if backbone_lr is None else backbone_lr
    backbone_params, other_params = [], []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(parameter)
        else:
            other_params.append(parameter)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if other_params:
        groups.append({"params": other_params, "lr": learning_rate})
    if not groups:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


# -------------------------------------------------------------
# FROZEN MODULE HANDLING
# -------------------------------------------------------------

def _keep_frozen_modules_eval(model):
    """Prevent frozen modules' BatchNorm running statistics from changing."""
    for module_name in ("backbone", "head", "text_enc"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        if not any(p.requires_grad for p in module.parameters()):
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
        if len(batch) != 4:
            raise ValueError(
                "Each batch must contain images, queries, targets, and mask."
            )

        images, queries, targets, mask = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                predictions, density_maps, cls_logits = model(images, queries)
                loss = masked_count_loss(predictions, targets, mask)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                gradients = [
                    p for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                torch.nn.utils.clip_grad_norm_(
                    gradients,
                    max_norm=cfg.gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()

        valid_mask = mask.to(device=predictions.device, dtype=torch.bool)
        valid = int(valid_mask.sum().item())
        total_loss += float(loss.detach().item()) * valid

        with torch.no_grad():
            valid_errors = torch.abs(
                predictions.detach().float()[valid_mask]
                - targets.to(
                    device=predictions.device,
                    dtype=torch.float32,
                )[valid_mask]
            )
            total_absolute_error += float(valid_errors.sum().item())
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
    """Train a model and restore the best checkpoint saved in this run."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if (
        not isinstance(start_epoch, int)
        or isinstance(start_epoch, bool)
        or start_epoch <= 0
    ):
        raise ValueError("start_epoch must be a positive integer.")
    if start_epoch > epochs:
        raise ValueError("start_epoch cannot be greater than epochs.")
    if len(train_loader) == 0:
        raise ValueError("Training DataLoader is empty.")

    device = _get_device(cfg)
    amp_enabled, amp_dtype = _get_amp_settings(cfg, device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(amp_enabled and amp_dtype == torch.float16),
    )

    best_metric = float("inf")
    best_checkpoint_saved = False
    checkpoint_path = Path(cfg.checkpoint_dir) / checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n[{stage_name}] Epoch {epoch}/{epochs}")

        train_metrics = run_epoch(
            model, train_loader, optimizer, device, cfg, scaler, training=True
        )

        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            val_metrics = run_epoch(
                model, val_loader, None, device, cfg, scaler, training=False
            )
            monitored_metric = val_metrics["mae"]
            print(
                f"Train loss: {train_metrics['loss']:.4f} | "
                f"Train MAE: {train_metrics['mae']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f}"
            )
        else:
            # This fallback is training-MAE selection, not validation selection.
            monitored_metric = train_metrics["mae"]
            print(
                f"Train loss: {train_metrics['loss']:.4f} | "
                f"Train MAE: {train_metrics['mae']:.4f} "
                "(no validation loader; selecting by training MAE)"
            )

        if not math.isfinite(float(monitored_metric)):
            raise RuntimeError(
                f"{stage_name} produced a non-finite monitored MAE "
                f"at epoch {epoch}: {monitored_metric}"
            )

        if monitored_metric < best_metric:
            best_metric = float(monitored_metric)
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                checkpoint_path,
                scaler,
            )
            best_checkpoint_saved = True
            print(f"Best checkpoint saved: {checkpoint_path}")

        if epoch % cfg.save_every == 0:
            periodic_path = (
                Path(cfg.checkpoint_dir)
                / f"{stage_name.lower().replace(' ', '_')}_epoch_{epoch}.pt"
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                monitored_metric,
                periodic_path,
                scaler,
            )

    # An old checkpoint from another run must never satisfy this condition.
    if not best_checkpoint_saved:
        raise RuntimeError(
            f"No best checkpoint was saved during this {stage_name} run."
        )

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
    _validate_training_arguments(epochs, bs, lr)

    if train_dataset is None:
        raise ValueError("A real training dataset is required.")

    print("\nInitializing Model B...")
    encoder = TextEncoder(
        vocab=None,
        d=cfg.embed_dim,
        model_name=cfg.text_encoder_name,
        freeze=cfg.freeze_text_encoder,
    )
    model = ModelB(text_enc=encoder, cfg=cfg).to(device)

    train_loader = create_data_loader(
        train_dataset, cfg, batch_size=bs, shuffle=True
    )
    val_loader = (
        create_data_loader(val_dataset, cfg, batch_size=bs, shuffle=False)
        if val_dataset is not None else None
    )

    optimizer = build_optimizer(
        model, cfg, learning_rate=lr, backbone_lr=cfg.backbone_lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    return fit(
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
    Train Model C in two stages:
      1. Freeze shared modules and train QVI.
      2. Optionally fine-tune the configured trainable network.
    """
    if not isinstance(model_B, ModelB):
        raise TypeError("model_B must be a trained ModelB instance.")
    if train_dataset is None:
        raise ValueError("A real training dataset is required.")

    cfg = cfg if cfg is not None else Config()
    cfg.set_seed()
    device = _get_device(cfg)

    epochs = cfg.epochs if epochs is None else epochs
    bs = cfg.batch_size if bs is None else bs
    lr = cfg.learning_rate if lr is None else lr
    _validate_training_arguments(epochs, bs, lr)

    if (
        not isinstance(fine_tune_epochs, int)
        or isinstance(fine_tune_epochs, bool)
        or fine_tune_epochs < 0
    ):
        raise ValueError("fine_tune_epochs must be a non-negative integer.")

    cfg_C = (
        replace(cfg, use_qvi=True, qvi_type="modulation")
        if not cfg.use_qvi
        else replace(cfg, use_qvi=True)
    )

    print("\nInitializing Model C from Model B...")
    model = ModelC(text_enc=model_B.text_enc, cfg=cfg_C).to(device)
    model.initialize_from_model_b(model_B, strict=True)

    if model.qvi is None:
        raise RuntimeError("Model C was created without QVI.")

    train_loader = create_data_loader(
        train_dataset, cfg_C, batch_size=bs, shuffle=True
    )
    val_loader = (
        create_data_loader(val_dataset, cfg_C, batch_size=bs, shuffle=False)
        if val_dataset is not None else None
    )

    # Stage 1: QVI only.
    print("\nStage 1: Training QVI only...")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.qvi.parameters():
        parameter.requires_grad = True

    optimizer = build_optimizer(
        model, cfg_C, learning_rate=lr, backbone_lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
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

    # Stage 2: optional fine-tuning.
    if fine_tune_epochs > 0:
        print("\nStage 2: Fine-tuning Model C...")

        for module in (model.backbone, model.head, model.qvi):
            for parameter in module.parameters():
                parameter.requires_grad = True

        if cfg_C.freeze_text_encoder:
            for parameter in model.text_enc.clip.parameters():
                parameter.requires_grad = False
            # Keep the encoder's optional learned projection trainable.
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
            optimizer, T_max=fine_tune_epochs
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
