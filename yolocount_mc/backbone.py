
"""
YOLOv8 backbone + multi-scale feature fusion for YOLO-Count.

Extracts P3, P4 and P5 feature maps from a pretrained YOLOv8 model,
projects them into a common channel dimension and fuses them.

Output:
    [B, out_ch, H/8, W/8]

The visual backbone executes once per image, irrespective of
the number of text queries.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class YoloBackbonePAFPN(nn.Module):
    """
    YOLOv8 feature extractor with multi-scale feature fusion.

    Args:
        in_ch: Number of input image channels.
        out_ch: Number of output feature channels.
        pretrained: Load pretrained weights when True.
        weights: YOLO checkpoint or model YAML.
        feature_indices: Layer indices for P3, P4 and P5.
        freeze_backbone: Freeze the original YOLO feature extractor.

    Input:
        img: [B, 3, H, W]

    Output:
        features: [B, out_ch, H/8, W/8]
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 256,
        pretrained: bool = True,
        weights: str = "yolov8n.pt",
        feature_indices: Tuple[int, ...] = (15, 18, 21),
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if in_ch != 3:
            raise ValueError("YOLOv8 expects three RGB input channels.")

        if out_ch <= 0:
            raise ValueError("out_ch must be positive.")

        if len(feature_indices) != 3:
            raise ValueError(
                "Exactly three feature indices are required: P3, P4, P5."
            )

        if tuple(sorted(feature_indices)) != tuple(feature_indices):
            raise ValueError("Feature indices must be in ascending order.")

        if min(feature_indices) < 0:
            raise ValueError("Feature indices cannot be negative.")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required. Install it using "
                "`pip install ultralytics`."
            ) from exc

        model_source = weights

        if not pretrained and weights.endswith(".pt"):
            model_source = weights[:-3] + ".yaml"

        self.detector = YOLO(model_source)
        self.yolo_model = self.detector.model

        # Use the official Ultralytics graph execution implementation.
        if not hasattr(self.yolo_model, "_predict_once"):
            raise RuntimeError(
                "This Ultralytics model does not expose _predict_once. "
                "Check your Ultralytics version."
            )

        self.layers = self.yolo_model.model
        self.feature_indices = tuple(feature_indices)

        if max(self.feature_indices) >= len(self.layers):
            raise ValueError(
                f"Invalid feature indices {self.feature_indices}. "
                f"The loaded model contains {len(self.layers)} layers."
            )

        # Validate feature taps using the actual loaded model graph.
        self._validate_feature_taps()

        # Obtain channel dimensions from a controlled dummy forward.
        feature_channels = self._infer_feature_channels()

        self.feature_channels = tuple(feature_channels)

        # -----------------------------------------------------
        # MULTI-SCALE PROJECTIONS
        # -----------------------------------------------------
        self.lateral_p3 = ConvBNAct(
            self.feature_channels[0],
            out_ch,
            kernel_size=1,
            padding=0,
        )

        self.lateral_p4 = ConvBNAct(
            self.feature_channels[1],
            out_ch,
            kernel_size=1,
            padding=0,
        )

        self.lateral_p5 = ConvBNAct(
            self.feature_channels[2],
            out_ch,
            kernel_size=1,
            padding=0,
        )

        # -----------------------------------------------------
        # FEATURE FUSION
        # -----------------------------------------------------
        self.fusion = nn.Sequential(
            ConvBNAct(out_ch * 3, out_ch, kernel_size=3, padding=1),
            ConvBNAct(out_ch, out_ch, kernel_size=3, padding=1),
        )

        self.out_channels = out_ch
        self.output_stride = 8
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            self._freeze_yolo()

    def _validate_feature_taps(self):
        """Check that the selected layers are valid feature-producing layers."""

        for index in self.feature_indices:
            layer = self.layers[index]

            if not isinstance(layer, nn.Module):
                raise TypeError(
                    f"YOLO layer {index} is not a PyTorch module."
                )

    def _run_yolo(self, img: torch.Tensor):
        """
        Run the YOLO graph and capture selected intermediate features.

        Ultralytics' _predict_once method handles its internal layer
        routing and saved activation logic.
        """

        captured = {}
        original_forward = self.yolo_model._predict_once

        # The official implementation stores intermediate outputs
        # in y. We capture the selected feature maps from this list.
        def capture_forward(x, profile=False, visualize=False, embed=None):
            y = []
            embeddings = []

            for module in self.yolo_model.model:
                if module.f != -1:
                    if isinstance(module.f, int):
                        x = y[module.f]
                    else:
                        x = [
                            x if j == -1 else y[j]
                            for j in module.f
                        ]

                x = module(x)

                y.append(
                    x if module.i in self.yolo_model.save else None
                )

                if module.i in self.feature_indices:
                    captured[module.i] = x

            return x

        # Execute the graph through the same module sequence and
        # routing semantics as Ultralytics.
        #
        # This intentionally does not call the detector's Detect head.
        x = img
        y = []

        for module in self.yolo_model.model:
            if module.i > max(self.feature_indices):
                break

            if module.f != -1:
                if isinstance(module.f, int):
                    x = y[module.f]
                else:
                    x = [
                        x if j == -1 else y[j]
                        for j in module.f
                    ]

            x = module(x)

            y.append(
                x if module.i in self.yolo_model.save
                or module.i in self.feature_indices
                else None
            )

            if module.i in self.feature_indices:
                captured[module.i] = x

        if len(captured) != 3:
            raise RuntimeError(
                f"Expected 3 feature maps, received {len(captured)}."
            )

        return tuple(
            captured[index] for index in self.feature_indices
        )

    def _infer_feature_channels(self):
        """Infer actual feature channels from a dummy forward pass."""

        was_training = self.yolo_model.training
        self.yolo_model.eval()

        try:
            parameter = next(self.yolo_model.parameters())
            device = parameter.device
            dtype = parameter.dtype

            dummy = torch.zeros(
                1, 3, 256, 256,
                device=device,
                dtype=dtype,
            )

            with torch.no_grad():
                features = self._run_yolo(dummy)

            channels = []

            for index, feature in zip(self.feature_indices, features):
                if feature.ndim != 4:
                    raise RuntimeError(
                        f"Feature layer {index} returned "
                        f"shape {tuple(feature.shape)}."
                    )

                channels.append(feature.shape[1])

            return channels

        finally:
            self.yolo_model.train(was_training)

    def _freeze_yolo(self):
        """Freeze YOLO parameters and BatchNorm running statistics."""

        for parameter in self.yolo_model.parameters():
            parameter.requires_grad = False

        self.yolo_model.eval()

    def train(self, mode: bool = True):
        """
        Set training mode while preserving frozen YOLO BatchNorm state.
        """

        super().train(mode)

        if self.freeze_backbone:
            self.yolo_model.eval()

        return self

    def extract_multiscale_features(
        self,
        img: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract P3, P4 and P5 feature maps."""

        if self.freeze_backbone:
            with torch.no_grad():
                return self._run_yolo(img)

        return self._run_yolo(img)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract and fuse multi-scale YOLO features.

        Input:
            [B, 3, H, W]

        Output:
            [B, out_channels, H/8, W/8]
        """

        if img.ndim != 4:
            raise ValueError(
                f"Expected [B,3,H,W], got {tuple(img.shape)}."
            )

        if img.shape[1] != 3:
            raise ValueError("Input must have three RGB channels.")

        if not img.is_floating_point():
            raise TypeError("Input image tensor must be floating point.")

        if img.shape[-2] % 32 != 0 or img.shape[-1] % 32 != 0:
            raise ValueError(
                "Input height and width must be divisible by 32."
            )

        p3, p4, p5 = self.extract_multiscale_features(img)

        p3 = self.lateral_p3(p3)
        p4 = self.lateral_p4(p4)
        p5 = self.lateral_p5(p5)

        target_size = p3.shape[-2:]

        p4 = F.interpolate(
            p4,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        p5 = F.interpolate(
            p5,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat([p3, p4, p5], dim=1)

        return self.fusion(fused)
