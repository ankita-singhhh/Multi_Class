
"""YOLO-Count Multi-Class + QVI package.

Three models:
    Model A: Original single-query YOLO-Count.
    Model B: Multi-class counting without QVI.
    Model C: Model B enhanced with QVI.
"""

from .config import Config
from .text_encoder import TextEncoder
from .backbone import YoloBackbonePAFPN
from .head import MultiClassHead
from .qvi import QVI
from .models import ModelA, ModelB, ModelC

__all__ = [
    "Config",
    "TextEncoder",
    "YoloBackbonePAFPN",
    "MultiClassHead",
    "QVI",
    "ModelA",
    "ModelB",
    "ModelC",
]
