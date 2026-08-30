"""Validated camera-mount profiles for repeatable cane experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


class MountProfileError(ValueError):
    """Raised when a camera-mount profile is incomplete or unsafe."""


@dataclass(frozen=True)
class CameraMountProfile:
    """The viewpoint contract shared by data collection and Pi experiments.

    ``distance_below_handle_mm`` is measured down the cane shaft from the lower
    edge of the user's normal grip. Positive pitch means that the optical axis
    points below a horizontal line perpendicular to the local cane shaft.
    """

    profile_id: str
    mount_zone: str
    distance_below_handle_mm: float
    max_distance_below_handle_mm: float
    optical_axis: str
    nominal_pitch_down_deg: float
    pitch_tolerance_deg: float
    nominal_roll_deg: float
    roll_tolerance_deg: float
    hand_occlusion_check_required: bool
    intrinsic_calibration_required: bool
    extrinsic_calibration_required: bool
    hardware_verified: bool

    def validate(self) -> None:
        if not self.profile_id.strip():
            raise MountProfileError("profile_id must be non-empty")
        if self.mount_zone != "near_top_handle":
            raise MountProfileError("mount_zone must be 'near_top_handle'")
        if self.optical_axis != "forward":
            raise MountProfileError("optical_axis must be 'forward'")
        if not 0 <= self.distance_below_handle_mm <= self.max_distance_below_handle_mm:
            raise MountProfileError(
                "distance_below_handle_mm must be within the declared top-handle zone"
            )
        if not 0 < self.max_distance_below_handle_mm <= 80:
            raise MountProfileError(
                "max_distance_below_handle_mm must keep the camera within 80 mm of the grip"
            )
        if not 0 <= self.nominal_pitch_down_deg <= 30:
            raise MountProfileError("nominal_pitch_down_deg must be between 0 and 30")
        if not 0 < self.pitch_tolerance_deg <= 10:
            raise MountProfileError("pitch_tolerance_deg must be in (0, 10]")
        if abs(self.nominal_roll_deg) > 10:
            raise MountProfileError("nominal_roll_deg magnitude must not exceed 10")
        if not 0 < self.roll_tolerance_deg <= 10:
            raise MountProfileError("roll_tolerance_deg must be in (0, 10]")
        if not self.hand_occlusion_check_required:
            raise MountProfileError("top-handle mounting requires a hand-occlusion check")
        if not self.intrinsic_calibration_required:
            raise MountProfileError("camera intrinsics must be calibrated")
        if not self.extrinsic_calibration_required:
            raise MountProfileError("mount extrinsics must be calibrated")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_mount_profile(path: Path) -> CameraMountProfile:
    """Load and validate a mount profile from JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        profile = CameraMountProfile(**payload)
    except TypeError as exc:
        raise MountProfileError(f"invalid mount profile fields: {exc}") from exc
    profile.validate()
    return profile
