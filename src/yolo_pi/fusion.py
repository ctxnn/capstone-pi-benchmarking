"""Fail-safe range-first fusion policy for deterministic laptop replay."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

from .types import FrameDetections, Sector


class SensorKind(str, Enum):
    TOF = "tof"
    ULTRASONIC = "ultrasonic"
    IMU = "imu"


class HazardLevel(int, Enum):
    CLEAR = 0
    WARNING = 1
    CRITICAL = 2
    SENSOR_FAULT = 3


@dataclass(frozen=True)
class SensorReading:
    sensor_id: str
    kind: SensorKind
    timestamp_ns: int
    sector: Sector
    distance_m: Optional[float] = None
    valid: bool = True

    def usable_range(self) -> bool:
        return (
            self.valid
            and self.kind in {SensorKind.TOF, SensorKind.ULTRASONIC}
            and self.distance_m is not None
            and math.isfinite(self.distance_m)
            and self.distance_m > 0.0
        )


@dataclass(frozen=True)
class FusionConfig:
    warning_distance_m: float = 1.0
    critical_distance_m: float = 0.5
    sensor_stale_after_ms: float = 250.0
    vision_stale_after_ms: float = 500.0
    vision_confidence: float = 0.35

    def __post_init__(self) -> None:
        if not 0 < self.critical_distance_m <= self.warning_distance_m:
            raise ValueError("Distances must satisfy 0 < critical <= warning")
        if self.sensor_stale_after_ms <= 0 or self.vision_stale_after_ms <= 0:
            raise ValueError("Staleness limits must be positive")
        if not 0.0 <= self.vision_confidence <= 1.0:
            raise ValueError("vision_confidence must be in [0, 1]")


@dataclass(frozen=True)
class HazardDecision:
    timestamp_ns: int
    level: HazardLevel
    sectors: Tuple[Sector, ...]
    nearest_distance_m: Optional[float]
    semantic_labels: Tuple[str, ...]
    range_triggered: bool
    fresh_range_sensor_count: int
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "level": self.level.name.lower(),
            "sectors": [sector.value for sector in self.sectors],
            "nearest_distance_m": self.nearest_distance_m,
            "semantic_labels": list(self.semantic_labels),
            "range_triggered": self.range_triggered,
            "fresh_range_sensor_count": self.fresh_range_sensor_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HapticCommand:
    timestamp_ns: int
    level: HazardLevel
    sectors: Tuple[Sector, ...]
    intensity: float
    pulse_period_ms: Optional[int]
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "level": self.level.name.lower(),
            "sectors": [sector.value for sector in self.sectors],
            "intensity": self.intensity,
            "pulse_period_ms": self.pulse_period_ms,
            "reason": self.reason,
        }


class HapticSink(Protocol):
    def emit(self, command: HapticCommand) -> None:
        """Deliver a haptic command to a real or recording implementation."""


@dataclass
class RecordingHapticSink:
    commands: List[HapticCommand] = field(default_factory=list)

    def emit(self, command: HapticCommand) -> None:
        self.commands.append(command)


def decision_to_haptic(decision: HazardDecision) -> HapticCommand:
    if decision.level == HazardLevel.CRITICAL:
        return HapticCommand(
            decision.timestamp_ns,
            decision.level,
            decision.sectors,
            intensity=1.0,
            pulse_period_ms=100,
            reason=decision.reason,
        )
    if decision.level == HazardLevel.WARNING:
        return HapticCommand(
            decision.timestamp_ns,
            decision.level,
            decision.sectors,
            intensity=0.65,
            pulse_period_ms=250,
            reason=decision.reason,
        )
    if decision.level == HazardLevel.SENSOR_FAULT:
        return HapticCommand(
            decision.timestamp_ns,
            decision.level,
            (Sector.UNKNOWN,),
            intensity=0.35,
            pulse_period_ms=1000,
            reason=decision.reason,
        )
    return HapticCommand(
        decision.timestamp_ns,
        decision.level,
        tuple(),
        intensity=0.0,
        pulse_period_ms=None,
        reason=decision.reason,
    )


class FusionEngine:
    """Retains the newest sensor values and evaluates a range-first policy."""

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()
        self._latest_readings: Dict[str, SensorReading] = {}
        self._latest_vision: Optional[FrameDetections] = None

    def update_sensor(self, reading: SensorReading) -> None:
        previous = self._latest_readings.get(reading.sensor_id)
        if previous is None or reading.timestamp_ns >= previous.timestamp_ns:
            self._latest_readings[reading.sensor_id] = reading

    def update_sensors(self, readings: Iterable[SensorReading]) -> None:
        for reading in readings:
            self.update_sensor(reading)

    def update_vision(self, detections: FrameDetections) -> None:
        if (
            self._latest_vision is None
            or detections.model_finished_ns >= self._latest_vision.model_finished_ns
        ):
            self._latest_vision = detections

    def _fresh_ranges(self, now_ns: int) -> List[SensorReading]:
        stale_ns = int(self.config.sensor_stale_after_ms * 1_000_000)
        return [
            reading
            for reading in self._latest_readings.values()
            if reading.usable_range()
            and 0 <= now_ns - reading.timestamp_ns <= stale_ns
        ]

    def _semantic_labels(self, now_ns: int, sectors: Tuple[Sector, ...]) -> Tuple[str, ...]:
        vision = self._latest_vision
        if vision is None:
            return tuple()
        stale_ns = int(self.config.vision_stale_after_ms * 1_000_000)
        if not 0 <= now_ns - vision.model_finished_ns <= stale_ns:
            return tuple()
        sector_set = set(sectors)
        labels = {
            detection.class_name
            for detection in vision.detections
            if detection.confidence >= self.config.vision_confidence
            and (not sector_set or detection.sector in sector_set)
        }
        return tuple(sorted(labels))

    def evaluate(self, now_ns: int) -> HazardDecision:
        fresh_ranges = self._fresh_ranges(now_ns)
        if not fresh_ranges:
            return HazardDecision(
                timestamp_ns=now_ns,
                level=HazardLevel.SENSOR_FAULT,
                sectors=(Sector.UNKNOWN,),
                nearest_distance_m=None,
                semantic_labels=tuple(),
                range_triggered=False,
                fresh_range_sensor_count=0,
                reason="no fresh valid range sensor reading",
            )

        nearest = min(float(reading.distance_m) for reading in fresh_ranges if reading.distance_m)
        if nearest <= self.config.critical_distance_m:
            level = HazardLevel.CRITICAL
        elif nearest <= self.config.warning_distance_m:
            level = HazardLevel.WARNING
        else:
            level = HazardLevel.CLEAR

        triggering = [
            reading
            for reading in fresh_ranges
            if reading.distance_m is not None
            and (
                (level == HazardLevel.CRITICAL and reading.distance_m <= self.config.critical_distance_m)
                or (level == HazardLevel.WARNING and reading.distance_m <= self.config.warning_distance_m)
            )
        ]
        sectors = tuple(
            sorted({reading.sector for reading in triggering}, key=lambda item: item.value)
        )
        labels = self._semantic_labels(now_ns, sectors)
        if level == HazardLevel.CLEAR:
            reason = "all fresh range readings exceed warning distance"
        else:
            semantic = f"; vision labels: {', '.join(labels)}" if labels else ""
            reason = f"nearest range {nearest:.3f} m{semantic}"
        return HazardDecision(
            timestamp_ns=now_ns,
            level=level,
            sectors=sectors,
            nearest_distance_m=nearest,
            semantic_labels=labels,
            range_triggered=level in {HazardLevel.WARNING, HazardLevel.CRITICAL},
            fresh_range_sensor_count=len(fresh_ranges),
            reason=reason,
        )
