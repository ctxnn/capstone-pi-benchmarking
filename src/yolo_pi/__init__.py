"""Laptop-first perception toolkit for the YOLO Pi haptic cane."""

from .detector import Detector, UltralyticsDetector
from .types import BoundingBox, Detection, FrameDetections, Sector

__all__ = [
    "BoundingBox",
    "Detection",
    "Detector",
    "FrameDetections",
    "Sector",
    "UltralyticsDetector",
]

__version__ = "0.1.0"
