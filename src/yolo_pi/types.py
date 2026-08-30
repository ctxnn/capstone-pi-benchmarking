"""Stable, backend-neutral data contracts used by perception and fusion."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Sector(str, Enum):
    """Horizontal warning sector relative to the camera image."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned bounding box in original-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Bounding-box maximum coordinates must not be below minima")

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def as_dict(self) -> Dict[str, float]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass(frozen=True)
class Detection:
    """One normalized object detection."""

    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox
    sector: Sector = Sector.UNKNOWN

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be in [0, 1]")

    def as_dict(self) -> Dict[str, object]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "box": self.box.as_dict(),
            "sector": self.sector.value,
        }


@dataclass(frozen=True)
class FrameDetections:
    """All normalized detections and timings for one captured frame."""

    frame_id: int
    captured_ns: int
    model_started_ns: int
    model_finished_ns: int
    width: int
    height: int
    detections: List[Detection] = field(default_factory=list)
    backend: str = "unknown"
    model_id: str = "unknown"
    stage_ms: Dict[str, float] = field(default_factory=dict)
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Frame width and height must be positive")
        if self.model_finished_ns < self.model_started_ns:
            raise ValueError("Model finish time must not precede start time")

    @property
    def model_call_ms(self) -> float:
        return (self.model_finished_ns - self.model_started_ns) / 1_000_000.0

    @property
    def frame_age_at_model_finish_ms(self) -> float:
        return (self.model_finished_ns - self.captured_ns) / 1_000_000.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "captured_ns": self.captured_ns,
            "model_started_ns": self.model_started_ns,
            "model_finished_ns": self.model_finished_ns,
            "width": self.width,
            "height": self.height,
            "backend": self.backend,
            "model_id": self.model_id,
            "source": self.source,
            "model_call_ms": self.model_call_ms,
            "frame_age_at_model_finish_ms": self.frame_age_at_model_finish_ms,
            "stage_ms": dict(self.stage_ms),
            "detections": [detection.as_dict() for detection in self.detections],
        }
