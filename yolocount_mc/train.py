
"""
Research training pipeline for YOLO-Count.

Model B:
    Multi-class counting baseline trained on real datasets.

Model C:
    Initialized from Model B.
    Stage 1: Freeze shared network and train QVI.
    Stage 2: Optionally fine-tune the complete trainable network.

No toy datasets or randomly generated training targets.

Expected dataset batch:
    images:  [B, 3, H, W]
    queries: List[List[str]]
    targets: [B, K]
    mask:    [B, K]
"""

from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .config import Config
    from .text_encoder import TextEncoder
    from .models import ModelB, ModelC
    from .dataset import collate
except ImportError:
    from config import Config
    from text_encoder import TextEncoder
    from models import ModelB, ModelC
    from dataset import collate


# -------------------------------------------------------------
# DEVICE AND MIXED PRECISION
# -------------------------------------------------------------

def _get_device(cfg):
    return torch.device(cfg.device)


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
    """
    Create a DataLoader for the real counting datasets.

    Dataset must return:
        image, queries, counts
    """

    if dataset is None:
        raise ValueError("A real dataset must be supplied.")

    workers = cfg.num_workers

    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.batch_size,
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

    predictions: [B, K]
    targets:     [B, K]
    mask:        [B, K]
    """

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} "
            f"does not match target shape {targets.shape}."
        )

    if predictions.shape != mask.shape:
        raise ValueError("Query mask shape does not match predictions.")

    mask = mask.to(
        device=predictions.device,
        dtype=predictions.dtype,
    )

    errors = F.l1_loss(
        predictions,
        targets,
        reduction="none",
    )

    valid_queries = mask.sum().clamp_min(1.0)

    return (errors * mask).sum() / valid_queries


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
):
    """Save a complete training checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
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
):
    """Resume model and optimizer training state."""

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
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
    """
    AdamW with separate backbone and remaining parameter groups.
    """

    backbone_lr = backbone_lr or cfg.backbone_lr

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
    """
    Execute one training or validation epoch.

    Returns:
        Average masked count loss.
        MAE over valid query positions.
    """

    model.train(training)

    amp_enabled, amp_dtype = _get_amp_settings(cfg, device)

    total_loss = 0.0
    total_absolute_error = 0.0
    total_valid_queries = 0

    for step, batch in enumerate(loader):

        images, queries, targets, mask = batch

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        mask = mask.to(
            device,
            non_blocking=True,
        )

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
                        p for p in model.parameters()
                        if p.requires_grad and p.grad is not None
                    ],
                    max_norm=cfg.gradient_clip_norm,
                )

                scaler.step(optimizer)
                scaler.update()

        valid = mask.sum().item()

        total_loss += loss.detach().item() * valid

        with torch.no_grad():
            absolute_error = (
                torch.abs(predictions.float() - targets.float())
                * mask
            )

            total_absolute_error += absolute_error.sum().item()
            total_valid_queries += valid

        if training and (step + 1) % cfg.log_interval == 0:
            print(
                f"  Step {step + 1}/{len(loader)} "
                f"Loss: {loss.item():.4f}"
            )

    denominator = max(total_valid_queries, 1)

    return {
        "loss": total_loss / denominator,
        "mae": total_absolute_error / denominator,
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
    """Common training engine for Model B and Model C."""

    device = _get_device(cfg)

    amp_enabled, amp_dtype = _get_amp_settings(cfg, device)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    best_val_mae = float("inf")
    best_train_loss = float("inf")

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

            monitored_mae = val_metrics["mae"]

            print(
                f"Train loss: {train_metrics['loss']:.4f} | "
                f"Train MAE: {train_metrics['mae']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f}"
            )

        else:
            monitored_mae = train_metrics["mae"]

            print(
                f"Train loss: {train_metrics['loss']:.4f} | "
                f"Train MAE: {train_metrics['mae']:.4f}"
            )

        improved = monitored_mae < best_val_mae

        if improved:
            best_val_mae = monitored_mae

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                monitored_mae,
                checkpoint_path,
            )

            print(f"Best checkpoint saved: {checkpoint_path}")

        if train_metrics["loss"] < best_train_loss:
            best_train_loss = train_metrics["loss"]

        if (
            epoch % cfg.save_every == 0
            and not improved
        ):
            periodic_path = (
                Path(cfg.checkpoint_dir)
                / f"{stage_name.lower()}_epoch_{epoch}.pt"
            )

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                monitored_mae,
                periodic_path,
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
    """
    Train Model B using a real counting dataset.

    Required:
        train_dataset: COCO or LVIS dataset instance.

    Optional:
        val_dataset: validation dataset instance.
    """

    cfg = cfg if cfg is not None else Config()

    cfg.set_seed()

    device = _get_device(cfg)

    epochs = epochs or cfg.epochs
    bs = bs or cfg.batch_size
    lr = lr or cfg.learning_rate

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
        Initialize from Model B.
        Freeze backbone, text encoder and counting head.
        Train QVI only.

    Stage 2 (optional):
        Unfreeze the trainable shared network.
        Fine-tune with lower learning rates.

    Args:
        model_B:
            Previously trained Model B.

        train_dataset:
            Real training dataset.

        val_dataset:
            Optional real validation dataset.

        fine_tune_epochs:
            Number of full-network fine-tuning epochs.
            Zero disables Stage 2.
    """

    cfg = cfg if cfg is not None else Config()

    cfg.set_seed()

    device = _get_device(cfg)

    epochs = epochs or cfg.epochs
    bs = bs or cfg.batch_size
    lr = lr or 1e-4

    if not cfg.use_qvi:
        cfg_C = replace(
            cfg,
            use_qvi=True,
            qvi_type="modulation",
        )
    else:
        cfg_C = replace(cfg, use_qvi=True)

    print("\nInitializing Model C from Model B...")

    # Reuse the trained Model B text encoder.
    model = ModelC(
        text_enc=model_B.text_enc,
        cfg=cfg_C,
    ).to(device)

    # Explicitly copy shared weights.
    model.initialize_from_model_b(
        model_B,
        strict=True,
    )

    # ---------------------------------------------------------
    # STAGE 1: FREEZE SHARED MODEL, TRAIN QVI
    # ---------------------------------------------------------
    print("\nStage 1: Training QVI only...")

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.qvi.parameters():
        parameter.requires_grad = True

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
    # STAGE 2: OPTIONAL FULL FINE-TUNING
    # ---------------------------------------------------------
    if fine_tune_epochs > 0:

        print("\nStage 2: Fine-tuning Model C...")

        # Unfreeze shared visual and counting network.
        # Keep frozen CLIP parameters frozen.
        for module in (
            model.backbone,
            model.head,
            model.qvi,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

        # Restore original CLIP freezing policy.
        if cfg_C.freeze_text_encoder:
            for parameter in model.text_enc.parameters():
                parameter.requires_grad = False
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
