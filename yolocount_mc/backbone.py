
"""
YOLOv8 backbone + multi-scale feature fusion for YOLO-Count.

Extracts P3, P4 and P5 feature maps from the pretrained YOLOv8
architecture and fuses them into one shared visual representation.

Output:
    [B, out_ch, H/8, W/8]

The backbone runs once per image, irrespective of the number
of text queries.
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
    Pretrained YOLOv8 feature extractor with multi-scale fusion.

    Args:
        in_ch:
            Number of input image channels.

        out_ch:
            Number of output feature channels.

        pretrained:
            Load pretrained YOLOv8 weights when True.

        weights:
            YOLOv8 checkpoint or model configuration.

        feature_indices:
            YOLOv8 layer indices corresponding to P3, P4 and P5.

        freeze_backbone:
            Freeze YOLOv8 feature extractor parameters.

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
            raise ValueError(
                "The pretrained YOLOv8 implementation expects RGB input "
                "(in_ch=3)."
            )

        if out_ch <= 0:
            raise ValueError("out_ch must be positive.")

        if len(feature_indices) != 3:
            raise ValueError(
                "Exactly three feature indices are required: P3, P4, P5."
            )

        if tuple(sorted(feature_indices)) != feature_indices:
            raise ValueError("feature_indices must be in ascending order.")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for the YOLOv8 backbone. "
                "Install it in the Jupyter environment using "
                "`pip install ultralytics`."
            ) from exc

        # -----------------------------------------------------
        # LOAD YOLOV8
        # -----------------------------------------------------
        model_source = weights if pretrained else (
            weights.replace(".pt", ".yaml")
            if weights.endswith(".pt")
            else weights
        )

        self.detector = YOLO(model_source)
        self.yolo_model = self.detector.model

        self.layers = self.yolo_model.model
        self.feature_indices = tuple(feature_indices)
        self.max_feature_index = max(feature_indices)

        if self.max_feature_index >= len(self.layers):
            raise ValueError(
                f"Feature index {self.max_feature_index} is invalid. "
                f"YOLOv8 model has {len(self.layers)} layers."
            )

        # Ultralytics uses this list to retain intermediate
        # activations needed by later layers.
        self.save_indices = set(self.yolo_model.save)

        # Also retain our three requested feature maps.
        self.save_indices.update(self.feature_indices)

        # -----------------------------------------------------
        # DETERMINE FEATURE CHANNELS
        # -----------------------------------------------------
        feature_channels = []

        for index in self.feature_indices:
            layer = self.layers[index]

            conv_layers = [
                module
                for module in layer.modules()
                if isinstance(module, nn.Conv2d)
            ]

            if not conv_layers:
                raise RuntimeError(
                    f"Could not determine output channels "
                    f"for YOLO layer {index}."
                )

            feature_channels.append(conv_layers[-1].out_channels)

        self.feature_channels = tuple(feature_channels)

        # -----------------------------------------------------
        # MULTI-SCALE PROJECTIONS
        # -----------------------------------------------------
        self.lateral_p3 = ConvBNAct(
            self.feature_channels[0], out_ch, kernel_size=1, padding=0
        )

        self.lateral_p4 = ConvBNAct(
            self.feature_channels[1], out_ch, kernel_size=1, padding=0
        )

        self.lateral_p5 = ConvBNAct(
            self.feature_channels[2], out_ch, kernel_size=1, padding=0
        )

        # -----------------------------------------------------
        # FEATURE FUSION
        # -----------------------------------------------------
        self.fusion = nn.Sequential(
            ConvBNAct(
                out_ch * 3,
                out_ch,
                kernel_size=3,
                padding=1,
            ),
            ConvBNAct(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
            ),
        )

        self.out_channels = out_ch
        self.output_stride = 8

        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for parameter in self.yolo_model.parameters():
                parameter.requires_grad = False

    def extract_multiscale_features(
        self,
        img: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute YOLOv8 layers and extract P3, P4 and P5.

        This follows the routing mechanism used by Ultralytics'
        BaseModel._predict_once implementation.
        """

        x = img
        saved_outputs = []
        extracted = {}

        for index, layer in enumerate(self.layers):

            if index > self.max_feature_index:
                break

            # Resolve YOLO layer connections.
            if layer.f != -1:
                if isinstance(layer.f, int):
                    x = saved_outputs[layer.f]
                else:
                    x = [
                        x if source == -1 else saved_outputs[source]
                        for source in layer.f
                    ]

            x = layer(x)

            # Save only outputs required by YOLO or our extractor.
            saved_outputs.append(
                x if index in self.save_indices else None
            )

            if index in self.feature_indices:
                extracted[index] = x

        if len(extracted) != 3:
            raise RuntimeError(
                "Failed to extract all three YOLO feature maps."
            )

        return tuple(
            extracted[index] for index in self.feature_indices
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract and fuse multi-scale YOLOv8 features.

        Input:
            img: [B, 3, H, W]

        Output:
            [B, out_channels, H/8, W/8]
        """

        if img.ndim != 4:
            raise ValueError(
                f"Expected a 4D image tensor [B,C,H,W], "
                f"received shape {tuple(img.shape)}."
            )

        if img.shape[1] != 3:
            raise ValueError(
                f"Expected 3 RGB channels, received {img.shape[1]}."
            )

        p3, p4, p5 = self.extract_multiscale_features(img)

        # Project all scales to a shared channel dimension.
        p3 = self.lateral_p3(p3)
        p4 = self.lateral_p4(p4)
        p5 = self.lateral_p5(p5)

        target_size = p3.shape[-2:]

        # Resize P4 and P5 to the P3 spatial resolution.
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

        # Fuse fine, medium and coarse features.
        fused = torch.cat([p3, p4, p5], dim=1)

        return self.fusion(fused)
