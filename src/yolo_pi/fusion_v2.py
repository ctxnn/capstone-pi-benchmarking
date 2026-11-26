"""Five-stage directional fusion engine for the SmartCane (v2).

The engine turns a heterogeneous sensor set (two ToF sensors and three
ultrasonic rangers, a 6-axis IMU, and a 15-class YOLO detector) into three
independent hazard states, one per haptic sector, plus a directional haptic
frame a motor driver can play back directly.

Stages, in the order data flows through them:

1. Per-sensor three-sample median pre-filter with a fast-attack bypass, so a
   single ghost ping cannot become a hazard while a real, twice-confirmed
   step closer does not wait for the median to roll over.
2. Sensor liveness: a healthy VL53L0X reporting open space counts as a fresh
   sensor, while an invalid or stale report counts as a missing one.
3. IMU conditioning: shaft pitch projection onto the walking plane, floor-echo
   suppression with a hard 0.35 m safety floor, and vision yaw-lag
   compensation.
4. Co-aligned inverse-variance (Gauss-Markov) fusion with consistency gating,
   including the persistent-closer-obstacle rule for narrow poles and glass.
5. Per-sector alpha-beta tracking for closing velocity and time-to-contact,
   distance-scheduled range/vision blending, and asymmetric hysteresis with a
   minimum dwell on de-escalation.

Everything here is standard library only and every public method is guarded by
a single re-entrant lock, because sensor threads and the haptic loop call in
concurrently.
"""

import math
import threading
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union

from .fusion import HazardDecision, HazardLevel, SensorKind, SensorReading
from .imu import IMUConfig, IMUProcessor, IMUSample, IMUState
from .sectors import horizontal_sector
from .types import Detection, FrameDetections, Sector

_SECTORS = (Sector.LEFT, Sector.CENTER, Sector.RIGHT)
_TRACKER_ALPHA = 0.55
_TRACKER_BETA = 0.30
_TRACKER_REINIT_M = 0.50
_TRACKER_MAX_SPEED_MPS = 3.0
_TRACKER_MAX_GAP_S = 0.6
_FILTER_WINDOW = 3
_FILTER_ATTACK_MARGIN_M = 0.08
_FILTER_ATTACK_REL_MARGIN = 0.04
_FILTER_RESET_GAP_NS = 1_000_000_000


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


class ExtendedHazardLevel(IntEnum):
    """Hazard ladder with a soft advisory rung and an explicit fault state."""

    CLEAR = 0
    ADVISORY = 1
    WARNING = 2
    CRITICAL = 3
    SENSOR_FAULT = 4


class SectorHealth(str, Enum):
    """Liveness of the sensors expected to cover one sector."""

    OK = "ok"
    DEGRADED = "degraded"
    FAULT = "fault"


@dataclass(frozen=True)
class RangeReading:
    """One range sensor report understood by the v2 engine.

    ``no_target`` is the important distinction: a healthy VL53L0X that sees
    nothing inside its 2 m range reports ``valid=True, no_target=True`` with no
    distance. That means "sensor alive, path clear", not "sensor broken".
    """

    sensor_id: str
    kind: SensorKind
    timestamp_ns: int
    sector: Sector
    distance_m: Optional[float] = None
    valid: bool = True
    no_target: bool = False
    max_range_m: float = 2.0

    @classmethod
    def from_sensor_reading(
        cls,
        reading: SensorReading,
        max_range_m: float = 2.0,
    ) -> "RangeReading":
        no_target = reading.valid and reading.distance_m is None
        return cls(
            sensor_id=reading.sensor_id,
            kind=reading.kind,
            timestamp_ns=reading.timestamp_ns,
            sector=reading.sector,
            distance_m=reading.distance_m,
            valid=reading.valid,
            no_target=no_target,
            max_range_m=max_range_m,
        )

    def has_measurement(self) -> bool:
        return (
            self.valid
            and not self.no_target
            and self.distance_m is not None
            and math.isfinite(self.distance_m)
            and self.distance_m > 0.0
        )


@dataclass(frozen=True)
class ClassVisionPrior:
    """Per-class severity, gating, and physical size for one YOLO class."""

    severity: float
    min_confidence: float
    canonical_height_m: float
    can_trigger_warning: bool = False


DEFAULT_CLASS_PRIORS: Dict[str, ClassVisionPrior] = {
    "stairs": ClassVisionPrior(1.00, 0.32, 1.20, True),
    "vehicle": ClassVisionPrior(0.95, 0.35, 1.55, True),
    "bicycle_motorcycle": ClassVisionPrior(0.90, 0.35, 1.35, True),
    "overhead_obstacle": ClassVisionPrior(0.90, 0.35, 0.80, True),
    "curb": ClassVisionPrior(0.85, 0.32, 0.20, True),
    "barrier": ClassVisionPrior(0.78, 0.35, 1.00, False),
    "person": ClassVisionPrior(0.75, 0.35, 1.68, False),
    "pole": ClassVisionPrior(0.72, 0.35, 2.20, False),
    "dog": ClassVisionPrior(0.70, 0.35, 0.55, False),
    "slope": ClassVisionPrior(0.70, 0.35, 1.00, False),
    "generic_obstacle": ClassVisionPrior(0.65, 0.38, 0.90, False),
    "tree": ClassVisionPrior(0.62, 0.38, 2.50, False),
    "cone": ClassVisionPrior(0.60, 0.38, 0.65, False),
    "signboard": ClassVisionPrior(0.55, 0.40, 1.50, False),
    "door": ClassVisionPrior(0.50, 0.40, 2.00, False),
}


def _default_expected_sensors() -> Dict[str, Sector]:
    return {
        "tof-left": Sector.LEFT,
        "tof-center": Sector.CENTER,
        "tof-right": Sector.RIGHT,
        "us-center-1": Sector.CENTER,
        "us-center-2": Sector.CENTER,
    }


def _default_class_priors() -> Dict[str, ClassVisionPrior]:
    return dict(DEFAULT_CLASS_PRIORS)


@dataclass(frozen=True)
class SmartCaneFusionConfig:
    """Tuning for the whole v2 pipeline."""

    critical_distance_m: float = 0.50
    warning_distance_m: float = 1.00
    advisory_distance_m: float = 1.60

    critical_release_m: float = 0.58
    critical_min_dwell_ms: float = 150.0
    warning_release_m: float = 1.12
    warning_min_dwell_ms: float = 200.0
    advisory_risk_on: float = 0.28
    advisory_risk_off: float = 0.18
    advisory_min_dwell_ms: float = 300.0

    sensor_stale_after_ms: float = 250.0
    vision_stale_after_ms: float = 500.0

    tof_base_sigma_m: float = 0.010
    tof_scale_sigma: float = 0.030
    us_base_sigma_m: float = 0.008
    us_scale_sigma: float = 0.025
    walk_speed_sigma_mps: float = 1.2

    consistency_min_gap_m: float = 0.15
    consistency_sigma_mult: float = 3.0
    closer_outlier_persist_count: int = 2
    closer_outlier_window_ms: float = 180.0

    ttc_warning_s: float = 0.85
    ttc_critical_s: float = 0.40
    ttc_min_closing_mps: float = 0.30
    ttc_max_distance_m: float = 1.60

    camera_height_m: float = 0.95
    camera_pitch_down_deg: float = 12.0
    camera_vfov_deg: float = 48.8
    center_fraction: float = 1.0 / 3.0
    vision_max_range_m: float = 3.8
    vision_decay_length_m: float = 1.6
    vision_persist_frames: int = 2
    vision_persist_window_ms: float = 450.0
    vision_warning_max_distance_m: float = 2.0
    vision_warning_risk_threshold: float = 0.52
    cross_modal_agreement_boost: float = 0.15

    vision_covers_blind_sectors: bool = True
    blind_sector_vision_max_distance_m: float = 2.0
    blind_sector_vision_min_confidence: float = 0.45
    blind_sector_vision_critical: bool = True
    blind_sector_vision_critical_max_m: float = 0.50
    blind_sector_vision_critical_min_confidence: float = 0.60
    blind_sector_vision_critical_frames: int = 3

    expected_sensors: Dict[str, Sector] = field(default_factory=_default_expected_sensors)
    sensor_pitch_offsets_deg: Dict[str, float] = field(default_factory=dict)
    imu_config: IMUConfig = field(default_factory=IMUConfig)
    class_priors: Dict[str, ClassVisionPrior] = field(default_factory=_default_class_priors)
    stationary_noncritical_scale: float = 0.65
    degraded_sector_pulse: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.critical_distance_m < self.warning_distance_m:
            raise ValueError("distances must satisfy 0 < critical < warning")
        if not self.warning_distance_m < self.advisory_distance_m:
            raise ValueError("advisory_distance_m must exceed warning_distance_m")
        if self.critical_release_m < self.critical_distance_m:
            raise ValueError("critical_release_m must not be below critical_distance_m")
        if self.warning_release_m < self.warning_distance_m:
            raise ValueError("warning_release_m must not be below warning_distance_m")
        if not 0.0 <= self.advisory_risk_off < self.advisory_risk_on <= 1.0:
            raise ValueError("advisory risk thresholds must satisfy 0 <= off < on <= 1")
        if self.sensor_stale_after_ms <= 0.0 or self.vision_stale_after_ms <= 0.0:
            raise ValueError("staleness limits must be positive")
        for name in ("tof_base_sigma_m", "us_base_sigma_m", "walk_speed_sigma_mps"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("tof_scale_sigma", "us_scale_sigma"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must not be negative")
        if self.consistency_min_gap_m <= 0.0 or self.consistency_sigma_mult <= 0.0:
            raise ValueError("consistency gates must be positive")
        if self.closer_outlier_persist_count < 1:
            raise ValueError("closer_outlier_persist_count must be at least 1")
        if self.closer_outlier_window_ms <= 0.0:
            raise ValueError("closer_outlier_window_ms must be positive")
        if not 0.0 < self.ttc_critical_s < self.ttc_warning_s:
            raise ValueError("TTC thresholds must satisfy 0 < critical < warning")
        if self.ttc_min_closing_mps <= 0.0 or self.ttc_max_distance_m <= 0.0:
            raise ValueError("TTC gating constants must be positive")
        if self.camera_height_m <= 0.0:
            raise ValueError("camera_height_m must be positive")
        if not 0.0 < self.camera_vfov_deg < 180.0:
            raise ValueError("camera_vfov_deg must be in (0, 180)")
        if not 0.0 < self.center_fraction < 1.0:
            raise ValueError("center_fraction must be strictly between 0 and 1")
        if self.vision_max_range_m <= self.vision_decay_length_m:
            raise ValueError("vision_max_range_m must exceed vision_decay_length_m")
        if self.vision_persist_frames < 1 or self.vision_persist_window_ms <= 0.0:
            raise ValueError("vision persistence settings must be positive")
        if not 0.0 <= self.stationary_noncritical_scale <= 1.0:
            raise ValueError("stationary_noncritical_scale must be in [0, 1]")
        if self.vision_covers_blind_sectors:
            if self.blind_sector_vision_max_distance_m <= 0.0:
                raise ValueError("blind_sector_vision_max_distance_m must be positive")
            if not 0.0 <= self.blind_sector_vision_min_confidence <= 1.0:
                raise ValueError("blind_sector_vision_min_confidence must be in [0, 1]")
            if not 0.0 <= self.blind_sector_vision_critical_min_confidence <= 1.0:
                raise ValueError(
                    "blind_sector_vision_critical_min_confidence must be in [0, 1]"
                )
            if self.blind_sector_vision_critical_max_m <= 0.0:
                raise ValueError("blind_sector_vision_critical_max_m must be positive")
            if self.blind_sector_vision_critical_frames < 1:
                raise ValueError("blind_sector_vision_critical_frames must be at least 1")


def distance_weight_schedule(distance_m: Optional[float]) -> Tuple[float, float]:
    """Return ``(w_range, w_vision)`` summing to one for the fused distance.

    Range owns everything inside one metre, hands over smoothly between one and
    two metres, and leaves vision on its own once the rangers are effectively
    blind.
    """

    if distance_m is None:
        return (0.0, 1.0)
    distance = float(distance_m)
    if not math.isfinite(distance):
        return (0.0, 1.0)
    if distance <= 1.0:
        progress = _clamp((1.0 - distance) / 0.65, 0.0, 1.0)
        w_range = 0.90 + 0.08 * progress
    elif distance <= 2.0:
        progress = distance - 1.0
        smooth = 0.5 * (1.0 - math.cos(math.pi * progress))
        w_range = 0.90 + (0.20 - 0.90) * smooth
    else:
        progress = _clamp((distance - 2.0) / 0.6, 0.0, 1.0)
        w_range = 0.20 * (1.0 - progress)
    w_range = _clamp(w_range, 0.0, 1.0)
    return (w_range, 1.0 - w_range)


def _focal_length_px(frame_height: int, config: SmartCaneFusionConfig) -> float:
    return (frame_height / 2.0) / math.tan(math.radians(config.camera_vfov_deg / 2.0))


def estimate_monocular_distance_m(
    det: Detection,
    frame_height: int,
    config: SmartCaneFusionConfig,
    imu_state: Optional[IMUState] = None,
) -> float:
    """Estimate range from one detection by blending two geometric cues.

    The ground-contact cue projects the box's lower edge onto the floor using
    the known camera height and depression angle; the canonical-height cue uses
    the box's apparent height against the class's real-world size. The first is
    the better cue while the cane is carried normally, the second takes over
    when the cane is lifted or tilted and the floor geometry breaks down.
    """

    if frame_height <= 0:
        raise ValueError("frame_height must be positive")
    focal_px = _focal_length_px(frame_height, config)
    center_y = frame_height / 2.0
    prior = config.class_priors.get(det.class_name)
    canonical_height = prior.canonical_height_m if prior is not None else 1.0
    depression = config.camera_pitch_down_deg
    ground_weight = 0.70
    if imu_state is not None and imu_state.fresh:
        depression += imu_state.relative_pitch_deg
        if imu_state.is_lifted_or_tilted:
            ground_weight = 0.20
    depression += math.degrees(math.atan2(det.box.y2 - center_y, focal_px))

    ground_distance: Optional[float] = None
    if depression > 2.0:
        ground_distance = config.camera_height_m / math.tan(math.radians(depression))
        if not math.isfinite(ground_distance) or ground_distance <= 0.0:
            ground_distance = None

    apparent_px = max(det.box.y2 - det.box.y1, 1.0)
    height_distance = focal_px * canonical_height / apparent_px

    if ground_distance is None:
        blended = height_distance
    else:
        blended = ground_weight * ground_distance + (1.0 - ground_weight) * height_distance
    return _clamp(blended, 0.35, 6.0)


def _confidence_factor(confidence: float, min_confidence: float) -> float:
    span = max(1.0 - min_confidence, 1e-6)
    progress = _clamp((confidence - min_confidence) / span, 0.0, 1.0)
    return 0.55 + 0.45 * progress


def _vision_range_factor(distance_m: float, config: SmartCaneFusionConfig) -> float:
    return _clamp(
        1.0 - (distance_m - 0.35) / (config.vision_max_range_m - 0.35),
        0.0,
        1.0,
    )


def _range_risk(distance_m: float, config: SmartCaneFusionConfig) -> float:
    critical = config.critical_distance_m
    warning = config.warning_distance_m
    advisory = config.advisory_distance_m
    if distance_m <= critical:
        return 1.0
    if distance_m < warning:
        span = warning - critical
        return 1.0 + (0.62 - 1.0) * (distance_m - critical) / span
    if distance_m < advisory:
        span = advisory - warning
        return 0.62 + (0.30 - 0.62) * (distance_m - warning) / span
    if distance_m < 2.0:
        span = 2.0 - advisory
        return 0.30 + (0.10 - 0.30) * (distance_m - advisory) / span
    return 0.10 * _clamp(1.0 - (distance_m - 2.0) / 1.5, 0.0, 1.0)


@dataclass(frozen=True)
class SectorFusionState:
    """Everything the engine believes about one haptic sector."""

    sector: Sector
    level: ExtendedHazardLevel
    health: SectorHealth
    fused_distance_m: Optional[float]
    distance_sigma_m: Optional[float]
    closing_velocity_mps: float
    ttc_s: Optional[float]
    range_risk: float
    vision_risk: float
    combined_risk: float
    range_weight: float
    vision_weight: float
    range_triggered: bool
    vision_triggered: bool
    ttc_triggered: bool
    fresh_sensor_ids: Tuple[str, ...]
    missing_sensor_ids: Tuple[str, ...]
    semantic_labels: Tuple[str, ...]
    vision_nearest_m: Optional[float]
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "sector": self.sector.value,
            "level": self.level.name.lower(),
            "health": self.health.value,
            "fused_distance_m": self.fused_distance_m,
            "distance_sigma_m": self.distance_sigma_m,
            "closing_velocity_mps": self.closing_velocity_mps,
            "ttc_s": self.ttc_s,
            "range_risk": self.range_risk,
            "vision_risk": self.vision_risk,
            "combined_risk": self.combined_risk,
            "range_weight": self.range_weight,
            "vision_weight": self.vision_weight,
            "range_triggered": self.range_triggered,
            "vision_triggered": self.vision_triggered,
            "ttc_triggered": self.ttc_triggered,
            "fresh_sensor_ids": list(self.fresh_sensor_ids),
            "missing_sensor_ids": list(self.missing_sensor_ids),
            "semantic_labels": list(self.semantic_labels),
            "vision_nearest_m": self.vision_nearest_m,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DirectionalHazardDecision:
    """One evaluation of all three sectors."""

    timestamp_ns: int
    overall_level: ExtendedHazardLevel
    sectors: Dict[Sector, SectorFusionState]
    nearest_distance_m: Optional[float]
    imu_state: IMUState
    fresh_range_sensor_count: int
    expected_range_sensor_count: int
    degraded_sensors: Tuple[str, ...]
    reason: str

    def to_legacy_decision(self) -> HazardDecision:
        """Map onto the v1 decision contract for old logs and tooling.

        ADVISORY deliberately maps to CLEAR: it is a soft cue, and a v1
        consumer that only understands four levels would otherwise buzz a
        warning at the user for advice-level confidence.
        """

        if self.overall_level == ExtendedHazardLevel.SENSOR_FAULT:
            legacy_level = HazardLevel.SENSOR_FAULT
            sectors: Tuple[Sector, ...] = (Sector.UNKNOWN,)
        elif self.overall_level == ExtendedHazardLevel.CRITICAL:
            legacy_level = HazardLevel.CRITICAL
            sectors = self._legacy_sectors({ExtendedHazardLevel.CRITICAL})
        elif self.overall_level == ExtendedHazardLevel.WARNING:
            legacy_level = HazardLevel.WARNING
            sectors = self._legacy_sectors(
                {ExtendedHazardLevel.WARNING, ExtendedHazardLevel.CRITICAL}
            )
        else:
            legacy_level = HazardLevel.CLEAR
            sectors = tuple()
        labels: List[str] = []
        for sector in _SECTORS:
            for label in self.sectors[sector].semantic_labels:
                if label not in labels:
                    labels.append(label)
        return HazardDecision(
            timestamp_ns=self.timestamp_ns,
            level=legacy_level,
            sectors=sectors,
            nearest_distance_m=self.nearest_distance_m,
            semantic_labels=tuple(sorted(labels)),
            range_triggered=any(state.range_triggered for state in self.sectors.values()),
            fresh_range_sensor_count=self.fresh_range_sensor_count,
            reason=self.reason,
        )

    def _legacy_sectors(self, levels: Iterable[ExtendedHazardLevel]) -> Tuple[Sector, ...]:
        wanted = set(levels)
        return tuple(
            sector for sector in _SECTORS if self.sectors[sector].level in wanted
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "overall_level": self.overall_level.name.lower(),
            "nearest_distance_m": self.nearest_distance_m,
            "fresh_range_sensor_count": self.fresh_range_sensor_count,
            "expected_range_sensor_count": self.expected_range_sensor_count,
            "degraded_sensors": list(self.degraded_sensors),
            "reason": self.reason,
            "imu": self.imu_state.as_dict(),
            "sectors": {
                sector.value: self.sectors[sector].as_dict() for sector in _SECTORS
            },
        }


@dataclass(frozen=True)
class SectorHapticCommand:
    """How one motor should be driven for one sector."""

    sector: Sector
    level: ExtendedHazardLevel
    intensity: float
    pulse_period_ms: Optional[int]
    duty_cycle: float
    waveform_id: int
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "sector": self.sector.value,
            "level": self.level.name.lower(),
            "intensity": self.intensity,
            "pulse_period_ms": self.pulse_period_ms,
            "duty_cycle": self.duty_cycle,
            "waveform_id": self.waveform_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DirectionalHapticFrame:
    """One frame of independent left/center/right motor commands."""

    timestamp_ns: int
    overall_level: ExtendedHazardLevel
    motors: Dict[Sector, SectorHapticCommand]
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "overall_level": self.overall_level.name.lower(),
            "reason": self.reason,
            "motors": {
                sector.value: self.motors[sector].as_dict() for sector in _SECTORS
            },
        }


class DirectionalHapticSink(Protocol):
    """Destination for directional haptic frames."""

    def emit_frame(self, frame: DirectionalHapticFrame) -> None:
        """Play one frame on the real or recording driver."""

    def stop(self) -> None:
        """Shut the driver down and park every motor."""


@dataclass
class RecordingDirectionalHapticSink:
    """Test sink that keeps every emitted frame in order."""

    frames: List[DirectionalHapticFrame] = field(default_factory=list)
    stopped: bool = False

    def emit_frame(self, frame: DirectionalHapticFrame) -> None:
        self.frames.append(frame)

    def stop(self) -> None:
        self.stopped = True


def decision_to_directional_haptics(
    decision: DirectionalHazardDecision,
    config: Optional[SmartCaneFusionConfig] = None,
) -> DirectionalHapticFrame:
    """Map each sector onto its own motor command.

    CRITICAL always plays at full strength. ADVISORY and WARNING are softened
    by ``stationary_noncritical_scale`` while the cane is standing still, which
    keeps a long wait at a kerb from becoming a constant buzz.
    """

    settings = config or SmartCaneFusionConfig()
    total_loss = (
        decision.overall_level == ExtendedHazardLevel.SENSOR_FAULT
        and decision.fresh_range_sensor_count == 0
        and decision.expected_range_sensor_count > 0
    )
    motors: Dict[Sector, SectorHapticCommand] = {}
    for sector in _SECTORS:
        state = decision.sectors[sector]
        if total_loss:
            motors[sector] = _haptic_command(
                sector,
                ExtendedHazardLevel.SENSOR_FAULT,
                0.35,
                1000,
                0.20,
                10,
                "total range sensor loss",
            )
            continue
        level = state.level
        if level == ExtendedHazardLevel.CRITICAL:
            motors[sector] = _haptic_command(
                sector, level, 1.0, 100, 0.70, 47, state.reason
            )
            continue
        if level == ExtendedHazardLevel.WARNING:
            closeness = _warning_closeness(state, settings)
            intensity = 0.60 + 0.28 * closeness
            period = 250 - 90 * closeness
            if decision.imu_state.is_stationary:
                intensity *= settings.stationary_noncritical_scale
            motors[sector] = _haptic_command(
                sector, level, intensity, int(round(period)), 0.45, 14, state.reason
            )
            continue
        if level == ExtendedHazardLevel.ADVISORY:
            basis = state.vision_risk if state.vision_risk > 0.0 else state.combined_risk
            intensity = 0.28 + 0.20 * _clamp(basis, 0.0, 1.0)
            if decision.imu_state.is_stationary:
                intensity *= settings.stationary_noncritical_scale
            motors[sector] = _haptic_command(
                sector, level, intensity, 520, 0.25, 1, state.reason
            )
            continue
        if level == ExtendedHazardLevel.SENSOR_FAULT:
            if settings.degraded_sector_pulse:
                motors[sector] = _haptic_command(
                    sector, level, 0.35, 1000, 0.20, 10, state.reason
                )
            else:
                motors[sector] = _haptic_command(
                    sector, ExtendedHazardLevel.CLEAR, 0.0, None, 0.0, 0, state.reason
                )
            continue
        motors[sector] = _haptic_command(
            sector, ExtendedHazardLevel.CLEAR, 0.0, None, 0.0, 0, state.reason
        )
    return DirectionalHapticFrame(
        timestamp_ns=decision.timestamp_ns,
        overall_level=decision.overall_level,
        motors=motors,
        reason=decision.reason,
    )


def _warning_closeness(
    state: SectorFusionState,
    config: SmartCaneFusionConfig,
) -> float:
    proximity = 0.0
    if state.fused_distance_m is not None:
        span = max(config.warning_distance_m - config.critical_distance_m, 1e-6)
        proximity = _clamp(
            (config.warning_distance_m - state.fused_distance_m) / span, 0.0, 1.0
        )
    return _clamp(max(proximity, state.combined_risk), 0.0, 1.0)


def _haptic_command(
    sector: Sector,
    level: ExtendedHazardLevel,
    intensity: float,
    period_ms: Optional[int],
    duty: float,
    waveform_id: int,
    reason: str,
) -> SectorHapticCommand:
    return SectorHapticCommand(
        sector=sector,
        level=level,
        intensity=float(_clamp(intensity, 0.0, 1.0)),
        pulse_period_ms=period_ms,
        duty_cycle=float(_clamp(duty, 0.0, 1.0)),
        waveform_id=int(waveform_id),
        reason=reason,
    )


class _MedianFilter:
    """Three-sample median with a two-sample fast-attack bypass.

    For an even number of retained samples the lower of the two middle values
    wins, which is the conservative choice for an obstacle detector.
    """

    __slots__ = ("_values", "_window", "_margin")

    def __init__(self, window: int = _FILTER_WINDOW, margin: float = _FILTER_ATTACK_MARGIN_M) -> None:
        self._values: List[float] = []
        self._window = max(int(window), 1)
        self._margin = float(margin)

    def clear(self) -> None:
        self._values.clear()

    def value(self) -> Optional[float]:
        if not self._values:
            return None
        ordered = sorted(self._values)
        return ordered[(len(ordered) - 1) // 2]

    def push(self, distance_m: float) -> Tuple[Optional[float], bool]:
        previous = self.value()
        self._values.append(float(distance_m))
        if len(self._values) > self._window:
            del self._values[0]
        if previous is not None and len(self._values) >= 2:
            margin = max(self._margin, _FILTER_ATTACK_REL_MARGIN * previous)
            if (
                self._values[-1] < previous - margin
                and self._values[-2] < previous - margin
            ):
                return self._values[-1], True
        return self.value(), False


@dataclass(frozen=True)
class _Measurement:
    sensor_id: str
    distance_m: float
    sigma_m: float
    persistent: bool = False
    timestamp_ns: int = 0


@dataclass(frozen=True)
class _Blend:
    distance_m: float
    sigma_m: float
    used_ids: Tuple[str, ...]
    rejected_ids: Tuple[str, ...]
    note: str
    latest_sensor_ns: int = 0


@dataclass(frozen=True)
class _VisionRecord:
    frame_id: int
    timestamp_ns: int
    labels: Tuple[str, ...]
    nearest_m: float
    severity: float
    confidence: float
    risk: float
    can_warn: bool


class _AlphaBetaTracker:
    """One-dimensional alpha-beta filter over a sector's fused distance."""

    __slots__ = ("distance_m", "velocity_mps", "last_ns", "has_value")

    def __init__(self) -> None:
        self.distance_m = 0.0
        self.velocity_mps = 0.0
        self.last_ns: Optional[int] = None
        self.has_value = False

    def reset(self) -> None:
        self.distance_m = 0.0
        self.velocity_mps = 0.0
        self.last_ns = None
        self.has_value = False

    @property
    def closing_velocity_mps(self) -> float:
        if not self.has_value:
            return 0.0
        return -self.velocity_mps

    def update(self, measurement_m: float, sensor_ns: int) -> None:
        if not self.has_value or self.last_ns is None:
            self.distance_m = measurement_m
            self.velocity_mps = 0.0
            self.last_ns = sensor_ns
            self.has_value = True
            return
        if sensor_ns <= self.last_ns:
            return
        dt_s = (sensor_ns - self.last_ns) / 1e9
        if dt_s > _TRACKER_MAX_GAP_S:
            self.distance_m = measurement_m
            self.velocity_mps = 0.0
            self.last_ns = sensor_ns
            return
        dt_s = max(dt_s, 0.01)
        predicted = self.distance_m + self.velocity_mps * dt_s
        residual = measurement_m - predicted
        reinit_threshold = max(_TRACKER_REINIT_M, 0.5 * abs(predicted))
        if abs(residual) > reinit_threshold:
            self.distance_m = measurement_m
            self.velocity_mps = 0.0
            self.last_ns = sensor_ns
            return
        self.distance_m = max(predicted + _TRACKER_ALPHA * residual, 0.0)
        self.velocity_mps = _clamp(
            self.velocity_mps + (_TRACKER_BETA / dt_s) * residual,
            -_TRACKER_MAX_SPEED_MPS,
            _TRACKER_MAX_SPEED_MPS,
        )
        self.last_ns = sensor_ns


@dataclass
class _LevelState:
    level: ExtendedHazardLevel = ExtendedHazardLevel.CLEAR
    entry_ns: int = 0
    closer_sensor_id: Optional[str] = None
    closer_distance_m: float = 0.0
    closer_count: int = 0
    closer_first_ns: int = 0
    closer_last_ns: int = 0
    closer_last_reading_ns: int = -1

    def reset(self) -> None:
        self.level = ExtendedHazardLevel.CLEAR
        self.entry_ns = 0
        self.closer_sensor_id = None
        self.closer_distance_m = 0.0
        self.closer_count = 0
        self.closer_first_ns = 0
        self.closer_last_ns = 0
        self.closer_last_reading_ns = -1


class SmartCaneFusionEngine:
    """Thread-safe five-stage fusion across LEFT, CENTER and RIGHT."""

    def __init__(self, config: Optional[SmartCaneFusionConfig] = None) -> None:
        self.config = config or SmartCaneFusionConfig()
        self._lock = threading.RLock()
        self._imu = IMUProcessor(self.config.imu_config)
        self._readings: Dict[str, RangeReading] = {}
        self._filters: Dict[str, _MedianFilter] = {}
        self._filtered: Dict[str, Tuple[float, bool]] = {}
        self._last_measure_ns: Dict[str, int] = {}
        self._trackers: Dict[Sector, _AlphaBetaTracker] = {
            sector: _AlphaBetaTracker() for sector in _SECTORS
        }
        self._levels: Dict[Sector, _LevelState] = {
            sector: _LevelState() for sector in _SECTORS
        }
        self._vision: Optional[FrameDetections] = None
        self._vision_history: Dict[Sector, List[_VisionRecord]] = {
            sector: [] for sector in _SECTORS
        }
        self._blind_critical_streak: Dict[Sector, int] = {
            sector: 0 for sector in _SECTORS
        }
        self._blind_critical_frame: Dict[Sector, int] = {}

    @property
    def imu(self) -> IMUProcessor:
        return self._imu

    def update_sensor(self, reading: Union[SensorReading, RangeReading]) -> None:
        with self._lock:
            normalized = (
                reading
                if isinstance(reading, RangeReading)
                else RangeReading.from_sensor_reading(reading)
            )
            previous = self._readings.get(normalized.sensor_id)
            if previous is not None and normalized.timestamp_ns < previous.timestamp_ns:
                return
            self._readings[normalized.sensor_id] = normalized
            if not normalized.has_measurement():
                return
            filter_state = self._filters.get(normalized.sensor_id)
            if filter_state is None:
                filter_state = _MedianFilter()
                self._filters[normalized.sensor_id] = filter_state
            last_measure = self._last_measure_ns.get(normalized.sensor_id)
            if last_measure is not None and normalized.timestamp_ns - last_measure > _FILTER_RESET_GAP_NS:
                filter_state.clear()
            filtered, attacked = filter_state.push(float(normalized.distance_m))
            if filtered is not None:
                self._filtered[normalized.sensor_id] = (filtered, attacked)
            self._last_measure_ns[normalized.sensor_id] = normalized.timestamp_ns

    def update_sensors(self, readings: Iterable[Union[SensorReading, RangeReading]]) -> None:
        with self._lock:
            for reading in readings:
                self.update_sensor(reading)

    def update_imu(self, sample: IMUSample) -> IMUState:
        with self._lock:
            return self._imu.update(sample)

    def calibrate_imu(self, samples: Sequence[IMUSample]) -> Tuple[float, float, float]:
        with self._lock:
            return self._imu.calibrate(samples)

    def update_vision(self, detections: FrameDetections) -> None:
        with self._lock:
            if (
                self._vision is None
                or detections.model_finished_ns >= self._vision.model_finished_ns
            ):
                self._vision = detections

    def reset(self) -> None:
        """Drop all sensor, tracker, level and IMU history."""

        with self._lock:
            self._readings.clear()
            self._filters.clear()
            self._filtered.clear()
            self._last_measure_ns.clear()
            for tracker in self._trackers.values():
                tracker.reset()
            for level_state in self._levels.values():
                level_state.reset()
            for history in self._vision_history.values():
                history.clear()
            for sector in _SECTORS:
                self._blind_critical_streak[sector] = 0
            self._blind_critical_frame.clear()
            self._vision = None
            self._imu.reset()

    def evaluate(self, now_ns: int) -> DirectionalHazardDecision:
        with self._lock:
            now = int(now_ns)
            imu_state = self._imu.state(now)
            liveness = self._liveness(now)
            vision_records, vision_frames = self._vision_evidence(now, imu_state)
            sectors = {
                sector: self._evaluate_sector(
                    sector, now, imu_state, liveness, vision_records, vision_frames
                )
                for sector in _SECTORS
            }
            fresh_count = sum(1 for status in liveness.values() if status != "missing")
            expected_count = len(self.config.expected_sensors)
            expected_missing = {
                sensor_id
                for sensor_id in self.config.expected_sensors
                if liveness.get(sensor_id, "missing") == "missing"
            }
            degraded = tuple(
                sorted(
                    expected_missing
                    | {
                        sensor_id
                        for sensor_id, status in liveness.items()
                        if status == "missing"
                    }
                )
            )
            total_loss = expected_count > 0 and fresh_count == 0
            sector_level = ExtendedHazardLevel(
                max(int(self._levels[sector].level) for sector in _SECTORS)
            )
            if total_loss and int(sector_level) < int(ExtendedHazardLevel.WARNING):
                # Nothing can range and nothing covers the gap: heartbeat the fault.
                overall = ExtendedHazardLevel.SENSOR_FAULT
            else:
                overall = sector_level
            distances = [
                state.fused_distance_m
                for state in sectors.values()
                if state.fused_distance_m is not None
            ]
            if distances:
                nearest: Optional[float] = min(distances)
            else:
                vision_distances = [
                    state.vision_nearest_m
                    for state in sectors.values()
                    if state.vision_nearest_m is not None
                ]
                nearest = min(vision_distances) if vision_distances else None
            return DirectionalHazardDecision(
                timestamp_ns=now,
                overall_level=overall,
                sectors=sectors,
                nearest_distance_m=nearest,
                imu_state=imu_state,
                fresh_range_sensor_count=fresh_count,
                expected_range_sensor_count=expected_count,
                degraded_sensors=degraded,
                reason=self._decision_reason(sectors, overall, degraded, nearest),
            )

    def _liveness(self, now_ns: int) -> Dict[str, str]:
        stale_ns = int(self.config.sensor_stale_after_ms * 1e6)
        status: Dict[str, str] = {}
        for sensor_id, reading in self._readings.items():
            if not reading.valid:
                status[sensor_id] = "missing"
                continue
            age_ns = now_ns - reading.timestamp_ns
            if age_ns < 0:
                age_ns = 0
            if age_ns > stale_ns:
                status[sensor_id] = "missing"
                continue
            if reading.no_target or reading.distance_m is None:
                status[sensor_id] = "clear"
                continue
            distance = reading.distance_m
            if not math.isfinite(distance) or distance <= 0.0:
                status[sensor_id] = "missing"
                continue
            status[sensor_id] = "measure"
        return status

    def _sigma_for(self, kind: SensorKind, distance_m: float) -> float:
        if kind == SensorKind.ULTRASONIC:
            return self.config.us_base_sigma_m + self.config.us_scale_sigma * distance_m
        return self.config.tof_base_sigma_m + self.config.tof_scale_sigma * distance_m

    def _evaluate_sector(
        self,
        sector: Sector,
        now_ns: int,
        imu_state: IMUState,
        liveness: Dict[str, str],
        vision_records: Dict[Sector, _VisionRecord],
        vision_frames: Dict[Sector, int],
    ) -> SectorFusionState:
        config = self.config
        expected = [
            sensor_id
            for sensor_id, sector_of_sensor in config.expected_sensors.items()
            if sector_of_sensor == sector
        ]
        expected_alive = [
            sensor_id
            for sensor_id in expected
            if liveness.get(sensor_id) in {"measure", "clear"}
        ]
        missing = [sensor_id for sensor_id in expected if sensor_id not in expected_alive]
        if not expected:
            health = SectorHealth.OK
        elif not expected_alive:
            health = SectorHealth.FAULT
        elif missing:
            health = SectorHealth.DEGRADED
        else:
            health = SectorHealth.OK

        fresh_sensor_ids: List[str] = []
        measurements: List[_Measurement] = []
        suppressed: List[str] = []
        for sensor_id in sorted(self._readings):
            reading = self._readings[sensor_id]
            if reading.sector != sector:
                continue
            status = liveness.get(sensor_id, "missing")
            if status == "missing":
                continue
            fresh_sensor_ids.append(sensor_id)
            if status != "measure":
                continue
            filtered = self._filtered.get(sensor_id)
            distance = filtered[0] if filtered is not None else reading.distance_m
            if distance is None or not math.isfinite(distance) or distance <= 0.0:
                continue
            pitch_offset = config.sensor_pitch_offsets_deg.get(sensor_id, 0.0)
            if self._imu.is_likely_ground_echo(
                distance, imu_state, sensor_pitch_offset_deg=pitch_offset
            ):
                suppressed.append(sensor_id)
                continue
            projected = self._imu.project_horizontal_distance(
                distance, imu_state, sensor_pitch_offset_deg=pitch_offset
            )
            if projected <= 0.0:
                continue
            age_s = max(0.0, (now_ns - reading.timestamp_ns) / 1e9)
            sigma = (
                self._sigma_for(reading.kind, projected)
                + config.walk_speed_sigma_mps * age_s
            )
            persistent = bool(filtered[1]) if filtered is not None else False
            measurements.append(
                _Measurement(sensor_id, projected, sigma, persistent, reading.timestamp_ns)
            )

        blend = self._fuse_sector_measurements(sector, measurements, now_ns)
        tracker = self._trackers[sector]
        if blend is None:
            tracker.reset()
            fused_distance = None
            distance_sigma = None
            closing_velocity = 0.0
            ttc = None
        else:
            tracker.update(blend.distance_m, blend.latest_sensor_ns or now_ns)
            fused_distance = tracker.distance_m
            distance_sigma = blend.sigma_m
            closing_velocity = tracker.closing_velocity_mps
            ttc = None
            tilting_toward_floor = imu_state.fresh and (
                imu_state.is_pitched_down or imu_state.relative_pitch_deg > 10.0
            )
            if (
                not tilting_toward_floor
                and closing_velocity >= config.ttc_min_closing_mps
                and fused_distance <= config.ttc_max_distance_m
            ):
                ttc = fused_distance / closing_velocity

        record = vision_records.get(sector)
        persistent_vision = (
            record is not None
            and vision_frames.get(sector, 0) >= config.vision_persist_frames
        )
        vision_risk = record.risk if record is not None else 0.0
        if fused_distance is not None:
            range_risk = _range_risk(fused_distance, config)
            range_weight, vision_weight = distance_weight_schedule(fused_distance)
        else:
            range_risk = 0.0
            range_weight, vision_weight = 0.0, 1.0
        combined_risk = min(1.0, range_risk + vision_weight * vision_risk)
        if (
            fused_distance is not None
            and record is not None
            and abs(fused_distance - record.nearest_m)
            <= max(0.30, 0.25 * fused_distance)
        ):
            combined_risk = min(
                1.0, combined_risk + config.cross_modal_agreement_boost
            )

        candidate = self._candidate_level(
            fused_distance, ttc, combined_risk, record, persistent_vision
        )
        blind_level = ExtendedHazardLevel.CLEAR
        frame = self._vision
        if frame is not None and frame.frame_id != self._blind_critical_frame.get(sector):
            self._blind_critical_frame[sector] = frame.frame_id
            qualifies = (
                health is not SectorHealth.OK
                and fused_distance is None
                and record is not None
                and record.nearest_m <= config.blind_sector_vision_critical_max_m
                and record.confidence >= config.blind_sector_vision_critical_min_confidence
            )
            self._blind_critical_streak[sector] = (
                self._blind_critical_streak[sector] + 1 if qualifies else 0
            )
        if (
            config.vision_covers_blind_sectors
            and health is not SectorHealth.OK
            and fused_distance is None
            and record is not None
        ):
            blind_warning = (
                vision_frames.get(sector, 0) >= config.vision_persist_frames
                and record.nearest_m <= config.blind_sector_vision_max_distance_m
                and record.confidence >= config.blind_sector_vision_min_confidence
            )
            if blind_warning:
                blind_level = ExtendedHazardLevel.WARNING
                if (
                    config.blind_sector_vision_critical
                    and self._blind_critical_streak.get(sector, 0)
                    >= config.blind_sector_vision_critical_frames
                ):
                    blind_level = ExtendedHazardLevel.CRITICAL
        if int(blind_level) > int(candidate):
            candidate = blind_level
        level = self._apply_hysteresis(
            sector, candidate, fused_distance, combined_risk, ttc, now_ns
        )
        published = (
            ExtendedHazardLevel.SENSOR_FAULT
            if health is SectorHealth.FAULT
            and int(level) < int(ExtendedHazardLevel.WARNING)
            else level
        )

        ttc_triggered = ttc is not None and ttc <= config.ttc_warning_s
        range_triggered = (
            fused_distance is not None
            and fused_distance <= config.advisory_distance_m
        ) or ttc_triggered
        vision_warning = (
            persistent_vision
            and record is not None
            and record.can_warn
            and record.risk >= config.vision_warning_risk_threshold
            and record.nearest_m <= config.vision_warning_max_distance_m
        )
        vision_triggered = (
            not range_triggered
            and (
                vision_warning
                or int(blind_level) >= int(ExtendedHazardLevel.WARNING)
                or (
                    combined_risk >= config.advisory_risk_on
                    and (persistent_vision or fused_distance is not None)
                )
            )
        )

        return SectorFusionState(
            sector=sector,
            level=published,
            health=health,
            fused_distance_m=fused_distance,
            distance_sigma_m=distance_sigma,
            closing_velocity_mps=closing_velocity,
            ttc_s=ttc,
            range_risk=range_risk,
            vision_risk=vision_risk,
            combined_risk=combined_risk,
            range_weight=range_weight,
            vision_weight=vision_weight,
            range_triggered=range_triggered,
            vision_triggered=vision_triggered,
            ttc_triggered=ttc_triggered,
            fresh_sensor_ids=tuple(sorted(fresh_sensor_ids)),
            missing_sensor_ids=tuple(sorted(missing)),
            semantic_labels=record.labels if record is not None else tuple(),
            vision_nearest_m=record.nearest_m if record is not None else None,
            reason=self._sector_reason(
                published,
                health,
                blend,
                measurements,
                suppressed,
                record,
                persistent_vision,
                ttc,
                missing,
            ),
        )

    def _fuse_sector_measurements(
        self,
        sector: Sector,
        measurements: Sequence[_Measurement],
        now_ns: int,
    ) -> Optional[_Blend]:
        level_state = self._levels[sector]
        if not measurements:
            level_state.closer_sensor_id = None
            level_state.closer_count = 0
            return None
        ordered = sorted(measurements, key=lambda measurement: measurement.distance_m)
        if len(ordered) == 1:
            level_state.closer_sensor_id = None
            level_state.closer_count = 0
            level_state.closer_last_reading_ns = -1
            return _Blend(
                ordered[0].distance_m,
                ordered[0].sigma_m,
                (ordered[0].sensor_id,),
                tuple(),
                "single sensor",
                ordered[0].timestamp_ns,
            )
        indices = range(len(ordered))
        all_agree = all(
            self._measurements_agree(ordered[i], ordered[j])
            for i in indices
            for j in indices
            if i < j
        )
        if all_agree:
            level_state.closer_sensor_id = None
            level_state.closer_count = 0
            level_state.closer_last_reading_ns = -1
            return self._blue(ordered, tuple(), "inverse-variance mean of agreeing sensors")
        core_indices = self._best_agreeing_pair(ordered)
        if core_indices is None:
            level_state.closer_sensor_id = None
            level_state.closer_count = 0
            level_state.closer_last_reading_ns = -1
            nearest = ordered[0]
            return _Blend(
                nearest.distance_m,
                nearest.sigma_m,
                (nearest.sensor_id,),
                tuple(m.sensor_id for m in ordered[1:]),
                "no sensor agreement, nearest reading kept",
                nearest.timestamp_ns,
            )
        core = self._blue(
            [ordered[index] for index in core_indices],
            tuple(
                measurement.sensor_id
                for index, measurement in enumerate(ordered)
                if index not in core_indices
            ),
            "agreeing pair",
        )
        leftovers = [
            measurement
            for index, measurement in enumerate(ordered)
            if index not in core_indices
        ]
        closer = [
            measurement
            for measurement in leftovers
            if measurement.distance_m < core.distance_m
        ]
        if not closer:
            level_state.closer_sensor_id = None
            level_state.closer_count = 0
            level_state.closer_last_reading_ns = -1
            return core
        candidate = closer[0]
        rejected = tuple(
            measurement.sensor_id
            for measurement in leftovers
            if measurement.sensor_id != candidate.sensor_id
        )
        if self._track_closer_outlier(sector, candidate, now_ns):
            return _Blend(
                candidate.distance_m,
                candidate.sigma_m,
                (candidate.sensor_id,),
                rejected,
                "persistent closer return trusted",
                candidate.timestamp_ns,
            )
        return _Blend(
            core.distance_m,
            core.sigma_m,
            core.used_ids,
            tuple(measurement.sensor_id for measurement in leftovers),
            "closer outlier rejected pending confirmation",
            core.latest_sensor_ns,
        )

    def _measurements_agree(self, first: _Measurement, second: _Measurement) -> bool:
        gap = abs(first.distance_m - second.distance_m)
        allowed = max(
            self.config.consistency_min_gap_m,
            self.config.consistency_sigma_mult
            * math.hypot(first.sigma_m, second.sigma_m),
        )
        return gap <= allowed

    def _best_agreeing_pair(
        self,
        ordered: Sequence[_Measurement],
    ) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int]] = None
        best_cost = float("inf")
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                if not self._measurements_agree(ordered[i], ordered[j]):
                    continue
                cost = ordered[i].sigma_m + ordered[j].sigma_m
                if cost < best_cost:
                    best_cost = cost
                    best = (i, j)
        return best

    def _blue(
        self,
        measurements: Sequence[_Measurement],
        rejected_ids: Tuple[str, ...],
        note: str,
    ) -> _Blend:
        total_weight = 0.0
        weighted_sum = 0.0
        for measurement in measurements:
            weight = 1.0 / (measurement.sigma_m * measurement.sigma_m)
            total_weight += weight
            weighted_sum += weight * measurement.distance_m
        distance = weighted_sum / total_weight
        sigma = math.sqrt(1.0 / total_weight)
        return _Blend(
            distance,
            sigma,
            tuple(measurement.sensor_id for measurement in measurements),
            rejected_ids,
            note,
            max((m.timestamp_ns for m in measurements), default=0),
        )

    def _track_closer_outlier(
        self,
        sector: Sector,
        candidate: _Measurement,
        now_ns: int,
    ) -> bool:
        config = self.config
        state = self._levels[sector]
        window_ns = int(config.closer_outlier_window_ms * 1e6)
        continuing = (
            state.closer_sensor_id == candidate.sensor_id
            and state.closer_count > 0
            and 0 <= now_ns - state.closer_last_ns <= window_ns
            and abs(state.closer_distance_m - candidate.distance_m) <= 0.30
        )
        if continuing:
            if candidate.timestamp_ns > state.closer_last_reading_ns:
                state.closer_count += 1
                state.closer_last_reading_ns = candidate.timestamp_ns
                state.closer_last_ns = now_ns
            state.closer_distance_m = candidate.distance_m
        else:
            state.closer_sensor_id = candidate.sensor_id
            state.closer_distance_m = candidate.distance_m
            state.closer_count = 1
            state.closer_first_ns = now_ns
            state.closer_last_ns = now_ns
            state.closer_last_reading_ns = candidate.timestamp_ns
        if state.closer_count >= config.closer_outlier_persist_count:
            return True
        return bool(candidate.persistent) and state.closer_count >= 1

    def _candidate_level(
        self,
        fused_distance: Optional[float],
        ttc: Optional[float],
        combined_risk: float,
        record: Optional[_VisionRecord],
        persistent_vision: bool,
    ) -> ExtendedHazardLevel:
        config = self.config
        if fused_distance is not None and fused_distance <= config.critical_distance_m:
            return ExtendedHazardLevel.CRITICAL
        if ttc is not None and ttc <= config.ttc_critical_s:
            return ExtendedHazardLevel.CRITICAL
        if fused_distance is not None and fused_distance <= config.warning_distance_m:
            return ExtendedHazardLevel.WARNING
        if ttc is not None and ttc <= config.ttc_warning_s:
            return ExtendedHazardLevel.WARNING
        if (
            persistent_vision
            and record is not None
            and record.can_warn
            and record.risk >= config.vision_warning_risk_threshold
            and record.nearest_m <= config.vision_warning_max_distance_m
        ):
            return ExtendedHazardLevel.WARNING
        if combined_risk >= config.advisory_risk_on and (
            persistent_vision or fused_distance is not None
        ):
            return ExtendedHazardLevel.ADVISORY
        return ExtendedHazardLevel.CLEAR

    def _apply_hysteresis(
        self,
        sector: Sector,
        candidate: ExtendedHazardLevel,
        fused_distance: Optional[float],
        combined_risk: float,
        ttc: Optional[float],
        now_ns: int,
    ) -> ExtendedHazardLevel:
        state = self._levels[sector]
        previous = state.level
        if candidate >= previous:
            if candidate != previous:
                state.level = candidate
                state.entry_ns = now_ns
            return state.level
        level = previous
        while level > candidate:
            if not self._release_ok(level, fused_distance, combined_risk, ttc):
                break
            if now_ns - state.entry_ns < int(self._dwell_ms(level) * 1e6):
                break
            level = ExtendedHazardLevel(max(0, int(level) - 1))
        if level != previous:
            state.level = level
            state.entry_ns = now_ns
        return state.level

    def _release_ok(
        self,
        level: ExtendedHazardLevel,
        fused_distance: Optional[float],
        combined_risk: float,
        ttc: Optional[float],
    ) -> bool:
        config = self.config
        if level == ExtendedHazardLevel.CRITICAL:
            if ttc is not None and ttc <= config.ttc_critical_s:
                return False
            if fused_distance is None:
                return True
            return fused_distance > config.critical_release_m
        if level == ExtendedHazardLevel.WARNING:
            if ttc is not None and ttc <= config.ttc_warning_s:
                return False
            if fused_distance is None:
                return True
            return fused_distance > config.warning_release_m
        if level == ExtendedHazardLevel.ADVISORY:
            return combined_risk <= config.advisory_risk_off
        return True

    def _dwell_ms(self, level: ExtendedHazardLevel) -> float:
        config = self.config
        if level == ExtendedHazardLevel.CRITICAL:
            return config.critical_min_dwell_ms
        if level == ExtendedHazardLevel.WARNING:
            return config.warning_min_dwell_ms
        if level == ExtendedHazardLevel.ADVISORY:
            return config.advisory_min_dwell_ms
        return 0.0

    def _vision_evidence(
        self,
        now_ns: int,
        imu_state: IMUState,
    ) -> Tuple[Dict[Sector, _VisionRecord], Dict[Sector, int]]:
        config = self.config
        window_ns = int(config.vision_persist_window_ms * 1e6)
        for sector in _SECTORS:
            self._vision_history[sector] = [
                record
                for record in self._vision_history[sector]
                if 0 <= now_ns - record.timestamp_ns <= window_ns
            ]
        records: Dict[Sector, _VisionRecord] = {}
        frame = self._vision
        if frame is not None:
            age_ns = now_ns - frame.model_finished_ns
            if 0 <= age_ns <= int(config.vision_stale_after_ms * 1e6):
                records = self._records_for_frame(frame, now_ns, imu_state)
                for sector, record in records.items():
                    history = self._vision_history[sector]
                    if not any(item.frame_id == record.frame_id for item in history):
                        history.append(record)
        counts = {
            sector: len({item.frame_id for item in self._vision_history[sector]})
            for sector in _SECTORS
        }
        return records, counts

    def _records_for_frame(
        self,
        frame: FrameDetections,
        now_ns: int,
        imu_state: IMUState,
    ) -> Dict[Sector, _VisionRecord]:
        config = self.config
        capture_age_s = max(0.0, (now_ns - frame.captured_ns) / 1e9)
        labels: Dict[Sector, set] = {}
        nearest: Dict[Sector, float] = {}
        best: Dict[Sector, _VisionRecord] = {}
        for detection in frame.detections:
            prior = config.class_priors.get(detection.class_name)
            if prior is None or detection.confidence < prior.min_confidence:
                continue
            distance = estimate_monocular_distance_m(
                detection, frame.height, config, imu_state
            )
            if distance > config.vision_max_range_m:
                continue
            if not imu_state.fresh and detection.sector in _SECTORS:
                sector = detection.sector
            else:
                compensated = self._imu.compensate_vision_center_x(
                    detection.box.center_x,
                    frame.width,
                    capture_age_s,
                    imu_state,
                )
                sector = horizontal_sector(
                    compensated, frame.width, config.center_fraction
                )
            risk = (
                prior.severity
                * _confidence_factor(detection.confidence, prior.min_confidence)
                * _vision_range_factor(distance, config)
            )
            labels.setdefault(sector, set()).add(detection.class_name)
            current_nearest = nearest.get(sector)
            nearest[sector] = (
                distance if current_nearest is None else min(current_nearest, distance)
            )
            existing = best.get(sector)
            if existing is None or risk > existing.risk:
                best[sector] = _VisionRecord(
                    frame_id=frame.frame_id,
                    timestamp_ns=frame.model_finished_ns,
                    labels=(detection.class_name,),
                    nearest_m=distance,
                    severity=prior.severity,
                    confidence=detection.confidence,
                    risk=risk,
                    can_warn=prior.can_trigger_warning
                    and distance <= config.vision_warning_max_distance_m,
                )
        return {
            sector: _VisionRecord(
                frame_id=record.frame_id,
                timestamp_ns=record.timestamp_ns,
                labels=tuple(sorted(labels[sector])),
                nearest_m=nearest[sector],
                severity=record.severity,
                confidence=record.confidence,
                risk=record.risk,
                can_warn=record.can_warn,
            )
            for sector, record in best.items()
        }

    def _sector_reason(
        self,
        level: ExtendedHazardLevel,
        health: SectorHealth,
        blend: Optional[_Blend],
        measurements: Sequence[_Measurement],
        suppressed: Sequence[str],
        record: Optional[_VisionRecord],
        persistent_vision: bool,
        ttc: Optional[float],
        missing: Sequence[str],
    ) -> str:
        parts: List[str] = []
        if blend is not None:
            parts.append(
                "range {:.3f} m from {} ({})".format(
                    blend.distance_m, ", ".join(blend.used_ids), blend.note
                )
            )
            if blend.rejected_ids:
                parts.append("ignored " + ", ".join(blend.rejected_ids))
        elif measurements:
            parts.append("range readings unusable")
        else:
            parts.append("no range measurement")
        if ttc is not None:
            parts.append("ttc {:.2f} s".format(ttc))
        if record is not None:
            suffix = "" if persistent_vision else " (unconfirmed)"
            parts.append(
                "vision {} at {:.2f} m{}".format(
                    "/".join(record.labels), record.nearest_m, suffix
                )
            )
        if suppressed:
            parts.append("floor echo from " + ", ".join(sorted(suppressed)))
        if missing:
            parts.append("missing " + ", ".join(sorted(missing)))
        parts.append("health {}".format(health.value))
        parts.append(level.name.lower())
        return "; ".join(parts)

    def _decision_reason(
        self,
        sectors: Dict[Sector, SectorFusionState],
        overall: ExtendedHazardLevel,
        degraded: Sequence[str],
        nearest: Optional[float],
    ) -> str:
        if overall == ExtendedHazardLevel.SENSOR_FAULT:
            return "total range sensor loss: none of the {} expected sensors is alive".format(
                len(self.config.expected_sensors)
            )
        active = [
            "{}={}".format(sector.value, sectors[sector].level.name.lower())
            for sector in _SECTORS
            if sectors[sector].level != ExtendedHazardLevel.CLEAR
        ]
        if not active:
            detail = "all sectors clear"
        else:
            detail = ", ".join(active)
        if nearest is not None:
            detail += "; nearest {:.3f} m".format(nearest)
        if degraded:
            detail += "; degraded " + ", ".join(degraded)
        return detail
