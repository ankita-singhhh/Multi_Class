
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

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel dimensions must be positive.")

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

        if (
            not isinstance(out_ch, int)
            or isinstance(out_ch, bool)
            or out_ch <= 0
        ):
            raise ValueError("out_ch must be a positive integer.")

        if not isinstance(weights, str) or not weights.strip():
            raise ValueError("weights must be a non-empty string.")

        if not isinstance(pretrained, bool):
            raise TypeError("pretrained must be a boolean.")

        if not isinstance(freeze_backbone, bool):
            raise TypeError("freeze_backbone must be a boolean.")

        feature_indices = tuple(feature_indices)

        if len(feature_indices) != 3:
            raise ValueError(
                "Exactly three feature indices are required: P3, P4, P5."
            )

        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            for index in feature_indices
        ):
            raise ValueError(
                "Feature indices must be non-negative integers."
            )

        if tuple(sorted(feature_indices)) != feature_indices:
            raise ValueError("Feature indices must be in ascending order.")

        if len(set(feature_indices)) != 3:
            raise ValueError("Feature indices must be unique.")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required. Install it using "
                "`pip install ultralytics`."
            ) from exc

        model_source = weights

        if not pretrained and weights.lower().endswith(".pt"):
            model_source = weights[:-3] + ".yaml"

        self.detector = YOLO(model_source)
        self.yolo_model = self.detector.model

        if not hasattr(self.yolo_model, "model"):
            raise RuntimeError(
                "Loaded Ultralytics model does not expose its layer graph."
            )

        self.layers = self.yolo_model.model
        self.feature_indices = feature_indices

        if max(feature_indices) >= len(self.layers):
            raise ValueError(
                f"Invalid feature indices {feature_indices}. "
                f"The loaded model contains {len(self.layers)} layers."
            )

        if not hasattr(self.yolo_model, "save"):
            raise RuntimeError(
                "Loaded Ultralytics model does not expose graph save indices."
            )

        self._validate_feature_taps()

        # Infer channels and verify P3/P4/P5 strides.
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

    def _validate_feature_taps(self) -> None:
        """Validate selected layers and their graph metadata."""

        for index in self.feature_indices:
            layer = self.layers[index]

            if not isinstance(layer, nn.Module):
                raise TypeError(
                    f"YOLO layer {index} is not a PyTorch module."
                )

            if not hasattr(layer, "f"):
                raise RuntimeError(
                    f"YOLO layer {index} does not expose graph routing "
                    "metadata 'f'."
                )

            if not hasattr(layer, "i"):
                raise RuntimeError(
                    f"YOLO layer {index} does not expose layer index 'i'."
                )

    def _run_yolo(
        self,
        img: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the YOLO feature graph and capture selected feature maps.

        This follows Ultralytics' layer routing semantics:
        - module.f selects previous saved activations.
        - module.save determines which outputs remain available.
        - the Detect head is not executed.
        """

        captured = {}
        saved_outputs = []

        for layer_index, module in enumerate(self.layers):
            if layer_index > max(self.feature_indices):
                break

            if module.f != -1:
                if isinstance(module.f, int):
                    img = saved_outputs[module.f]
                else:
                    img = [
                        img if source == -1 else saved_outputs[source]
                        for source in module.f
                    ]

            img = module(img)

            # Preserve Ultralytics graph routing behavior.
            saved_outputs.append(
                img if module.i in self.yolo_model.save else None
            )

            if layer_index in self.feature_indices:
                captured[layer_index] = img

        if len(captured) != 3:
            raise RuntimeError(
                f"Expected 3 feature maps, received {len(captured)}."
            )

        return tuple(
            captured[index] for index in self.feature_indices
        )

    def _infer_feature_channels(self):
        """
        Infer feature channels from a dummy forward pass.

        Also checks the expected P3/P4/P5 output strides using
        a 256 x 256 input.
        """

        was_training = self.yolo_model.training
        self.yolo_model.eval()

        try:
            try:
                parameter = next(self.yolo_model.parameters())
            except StopIteration as exc:
                raise RuntimeError(
                    "Loaded YOLO model has no parameters."
                ) from exc

            dummy = torch.zeros(
                1,
                3,
                256,
                256,
                device=parameter.device,
                dtype=parameter.dtype,
            )

            with torch.no_grad():
                features = self._run_yolo(dummy)

            channels = []
            expected_sizes = ((32, 32), (16, 16), (8, 8))

            for index, feature, expected_size in zip(
                self.feature_indices,
                features,
                expected_sizes,
            ):
                if feature.ndim != 4:
                    raise RuntimeError(
                        f"Feature layer {index} returned invalid shape "
                        f"{tuple(feature.shape)}."
                    )

                if feature.shape[0] != 1:
                    raise RuntimeError(
                        f"Feature layer {index} returned an invalid batch."
                    )

                actual_size = tuple(feature.shape[-2:])

                if actual_size != expected_size:
                    raise RuntimeError(
                        f"Feature layer {index} produced spatial size "
                        f"{actual_size}; expected {expected_size}. "
                        "Check feature_indices. P3, P4 and P5 must have "
                        "strides 8, 16 and 32 respectively."
                    )

                channels.append(int(feature.shape[1]))

            return channels

        finally:
            self.yolo_model.train(was_training)

    def _freeze_yolo(self) -> None:
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

        if not isinstance(img, torch.Tensor):
            raise TypeError("Input image must be a torch.Tensor.")

        if img.ndim != 4:
            raise ValueError(
                f"Expected [B,3,H,W], got {tuple(img.shape)}."
            )

        if img.shape[0] <= 0:
            raise ValueError("Image batch cannot be empty.")

        if img.shape[1] != 3:
            raise ValueError("Input must have three RGB channels.")

        if not img.is_floating_point():
            raise TypeError("Input image tensor must be floating point.")

        if not torch.isfinite(img).all().item():
            raise ValueError("Input image contains NaN or infinite values.")

        if (
            img.shape[-2] <= 0
            or img.shape[-1] <= 0
            or img.shape[-2] % 32 != 0
            or img.shape[-1] % 32 != 0
        ):
            raise ValueError(
                "Input height and width must be positive and divisible by 32."
            )

        backbone_parameter = next(self.yolo_model.parameters())

        if img.device != backbone_parameter.device:
            raise ValueError(
                f"Input is on {img.device}, but YOLO backbone is on "
                f"{backbone_parameter.device}."
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
