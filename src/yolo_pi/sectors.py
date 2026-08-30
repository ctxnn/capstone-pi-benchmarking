"""Mapping from image-space detections to haptic warning sectors."""

from dataclasses import replace
from typing import Iterable, List

from .types import Detection, Sector


def horizontal_sector(center_x: float, frame_width: int, center_fraction: float = 1 / 3) -> Sector:
    """Return left/center/right using a configurable centered image band.

    ``center_fraction=1/3`` creates equal-width thirds. A wider center band can
    later be selected for the final camera field of view without changing the
    detector contract.
    """

    if frame_width <= 0:
        raise ValueError("frame_width must be positive")
    if not 0.0 < center_fraction < 1.0:
        raise ValueError("center_fraction must be strictly between 0 and 1")

    normalized_x = center_x / frame_width
    center_left = (1.0 - center_fraction) / 2.0
    center_right = 1.0 - center_left
    # A small tolerance keeps mathematically equal third boundaries stable when
    # values such as 1/3 are represented as binary floating point.
    epsilon = 1e-12
    if normalized_x < center_left - epsilon:
        return Sector.LEFT
    if normalized_x > center_right + epsilon:
        return Sector.RIGHT
    return Sector.CENTER


def assign_sectors(
    detections: Iterable[Detection],
    frame_width: int,
    center_fraction: float = 1 / 3,
) -> List[Detection]:
    """Return immutable detections with image sectors assigned."""

    return [
        replace(
            detection,
            sector=horizontal_sector(detection.box.center_x, frame_width, center_fraction),
        )
        for detection in detections
    ]
