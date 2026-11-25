"""Deterministic physics-based simulator for the SmartCane v2 pipeline.

The module replays fourteen scripted real-world situations through the real
fusion engine (:class:`~yolo_pi.fusion_v2.SmartCaneFusionEngine`) and the real
haptic mapper, so a scenario failure means the production code -- not a mock --
misbehaved. Nothing here touches hardware.

Sensor models follow the numbers in the Capstone report:

``VL53L0X`` (left, centre, right)
    30 Hz, 3 % Gaussian scale noise with an 8 mm floor, a 2.0 m maximum range
    that reports ``no_target`` rather than a bogus distance, and switchable
    specular misses (dark glass) or hard dropouts.
``HC-SR04`` (#1 and #2, both centre)
    Time-division multiplexed alternate pings, ~14 Hz combined, 5.6 mm Gaussian
    standard deviation plus a 1 % scale error, with optional single-ping
    multipath spikes and wide-beam background returns.
``MPU-6050``
    100 Hz, startup gyro-bias calibration, walking gait oscillation, a
    programmable shaft pitch, yaw sweep, stationary dwell and impact shock.
``IMX219 + YOLO``
    ~15 Hz frames with a 75 ms inference latency, 640x480 boxes projected from
    simulated 3D obstacles (class, true distance, true sector, canonical
    height), confidence noise, single-frame false positives and a genuinely
    yaw-shifted apparent pixel centre while the cane sweeps.

The decision loop runs at 30 Hz -- the fastest range sensor's rate -- while the
simulated IMU is sampled at 100 Hz, which is how the haptic thread is expected
to run on the Pi.

Every scenario is scored on the decision stream it produces: latency from the
physical stimulus, hysteresis chatter, false alarms against ground truth, and
per-tick evaluation cost.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .fusion import SensorKind
from .fusion_v2 import (
    DirectionalHapticFrame,
    DirectionalHazardDecision,
    ExtendedHazardLevel,
    RangeReading,
    SectorHealth,
    SmartCaneFusionConfig,
    SmartCaneFusionEngine,
    decision_to_directional_haptics,
)
from .imu import GRAVITY_MPS2, IMUSample, IMUState
from .sectors import horizontal_sector
from .types import BoundingBox, Detection, FrameDetections, Sector

MS = 1_000_000
TICK_NS = 33_333_333
IMU_PERIOD_NS = 10 * MS
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TOF_PERIOD_NS = 33_333_333
US_PING_PERIOD_NS = 70 * MS
US_PING_OFFSET_NS = 35 * MS
VISION_PERIOD_NS = 66_666_667
VISION_LATENCY_NS = 75 * MS
TOF_MAX_RANGE_M = 2.0
US_MAX_RANGE_M = 4.0
IMU_BASELINE_PITCH_DEG = 12.0
GAIT_HZ = 1.9
TOF_GYRO_NOISE_RPS = 0.006
ACCEL_NOISE_MPS2 = 0.05

SECTORS: Tuple[Sector, ...] = (Sector.LEFT, Sector.CENTER, Sector.RIGHT)

_TOF_IDS: Tuple[str, ...] = ("tof-left", "tof-center", "tof-right")
_US_IDS: Tuple[str, ...] = ("us-center-1", "us-center-2")
_SENSOR_SECTORS: Dict[str, Sector] = {
    "tof-left": Sector.LEFT,
    "tof-center": Sector.CENTER,
    "tof-right": Sector.RIGHT,
    "us-center-1": Sector.CENTER,
    "us-center-2": Sector.CENTER,
}
_ALL_SENSORS = _TOF_IDS + _US_IDS


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


@dataclass(frozen=True)
class VisionTarget:
    """One obstacle as it exists in the simulated world."""

    class_name: str
    distance_m: float
    bearing_deg: float = 0.0
    confidence: float = 0.90
    class_id: int = 0


@dataclass
class Truth:
    """The physical world at one instant of simulated time.

    ``ranges`` maps a sensor id to the true distance it would measure; a
    missing key or ``None`` means genuinely open space. ``misses`` models a
    sensor that returns nothing although something is there (dark glass on a
    ToF, absorbent clothing on an ultrasonic), ``silent`` models a sensor that
    stops reporting at all, and ``dropouts`` models one that reports itself as
    invalid.
    """

    ranges: Dict[str, Optional[float]] = field(default_factory=dict)
    misses: Set[str] = field(default_factory=set)
    silent: Set[str] = field(default_factory=set)
    dropouts: Set[str] = field(default_factory=set)
    vision: List[VisionTarget] = field(default_factory=list)
    pitch_deg: float = IMU_BASELINE_PITCH_DEG
    yaw_rate_dps: float = 0.0
    gait_mps2: float = 0.0
    shock_mps2: float = 0.0
    stationary: bool = False
    hazards: Dict[Sector, Optional[float]] = field(
        default_factory=lambda: {sector: None for sector in SECTORS}
    )


class _TickContext:
    """Mutable bookkeeping handed to a scenario's truth function."""

    def __init__(self) -> None:
        self.pings: Dict[str, int] = {sensor_id: 0 for sensor_id in _ALL_SENSORS}
        self.frame_index = -1


@dataclass(frozen=True)
class _Sample:
    t_ns: int
    hazards: Dict[Sector, Optional[float]]
    decision: DirectionalHazardDecision
    haptic: DirectionalHapticFrame

    @property
    def t_ms(self) -> float:
        return self.t_ns / 1e6


@dataclass(frozen=True)
class SimulationResult:
    """Scorecard for one scenario."""

    name: str
    passed: bool
    checks: Dict[str, bool]
    ticks: int
    chatter_transitions: int
    false_critical_count: int
    false_warning_count: int
    reaction_latency_ms: Optional[float]
    max_tick_eval_ms: float
    p95_tick_eval_ms: float
    summary: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "checks": dict(self.checks),
            "ticks": self.ticks,
            "chatter_transitions": self.chatter_transitions,
            "false_critical_count": self.false_critical_count,
            "false_warning_count": self.false_warning_count,
            "reaction_latency_ms": self.reaction_latency_ms,
            "max_tick_eval_ms": round(self.max_tick_eval_ms, 4),
            "p95_tick_eval_ms": round(self.p95_tick_eval_ms, 4),
            "summary": self.summary,
        }


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(0.95 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def _level(sample: _Sample, sector: Sector) -> ExtendedHazardLevel:
    return sample.decision.sectors[sector].level


def _first_sample(
    samples: Sequence[_Sample],
    predicate: Callable[[_Sample], bool],
) -> Optional[_Sample]:
    for sample in samples:
        if predicate(sample):
            return sample
    return None


def _latency_ms(
    samples: Sequence[_Sample],
    stimulus: Callable[[_Sample], bool],
    alert: Callable[[_Sample], bool],
) -> Optional[float]:
    start = _first_sample(samples, stimulus)
    if start is None:
        return None
    index = samples.index(start)
    for sample in samples[index:]:
        if alert(sample):
            return (sample.t_ns - start.t_ns) / 1e6
    return None


class _Rig:
    """The bench: world physics in, sensor readings and decisions out."""

    def __init__(self, seed: int, config: Optional[SmartCaneFusionConfig] = None) -> None:
        self.config = config or SmartCaneFusionConfig()
        self.rng = random.Random(seed)
        self.engine = SmartCaneFusionEngine(self.config)
        self.gyro_bias_rps = (0.012, -0.009, 0.017)
        self.frame_records: List[Dict[str, object]] = []
        self._calibrate()

    def _calibrate(self) -> None:
        samples = [
            self._imu_sample(
                t_ns=-(40 - index) * TICK_NS,
                pitch_deg=IMU_BASELINE_PITCH_DEG,
                pitch_rate_dps=0.0,
                yaw_rate_dps=0.0,
                gait_mps2=0.0,
                shock_mps2=0.0,
            )
            for index in range(40)
        ]
        self.engine.calibrate_imu(samples)

    def _imu_sample(
        self,
        t_ns: int,
        pitch_deg: float,
        pitch_rate_dps: float,
        yaw_rate_dps: float,
        gait_mps2: float,
        shock_mps2: float,
    ) -> IMUSample:
        pitch = math.radians(pitch_deg)
        gait = gait_mps2 * math.sin(2.0 * math.pi * GAIT_HZ * t_ns / 1e9)
        roll_noise = self.rng.gauss(0.0, 0.03)
        return IMUSample(
            timestamp_ns=int(t_ns),
            ax_mps2=-GRAVITY_MPS2 * math.sin(pitch) + 0.25 * gait + self.rng.gauss(0.0, ACCEL_NOISE_MPS2),
            ay_mps2=roll_noise,
            az_mps2=GRAVITY_MPS2 * math.cos(pitch) + gait + shock_mps2 + self.rng.gauss(0.0, ACCEL_NOISE_MPS2),
            gx_rps=self.rng.gauss(0.0, TOF_GYRO_NOISE_RPS),
            gy_rps=math.radians(pitch_rate_dps) + self.rng.gauss(0.0, TOF_GYRO_NOISE_RPS),
            gz_rps=math.radians(yaw_rate_dps)
            + self.gyro_bias_rps[2]
            + self.rng.gauss(0.0, TOF_GYRO_NOISE_RPS),
            temp_c=31.0,
        )

    def _range_reading(self, sensor_id: str, truth: Truth, now_ns: int) -> Optional[RangeReading]:
        if sensor_id in truth.silent:
            return None
        kind = SensorKind.TOF if sensor_id.startswith("tof") else SensorKind.ULTRASONIC
        sector = _SENSOR_SECTORS[sensor_id]
        invalid = sensor_id in truth.dropouts
        if invalid or sensor_id in truth.misses:
            return RangeReading(
                sensor_id=sensor_id,
                kind=kind,
                timestamp_ns=int(now_ns),
                sector=sector,
                distance_m=None,
                valid=not invalid,
                no_target=not invalid,
                max_range_m=TOF_MAX_RANGE_M if kind is SensorKind.TOF else US_MAX_RANGE_M,
            )
        distance = truth.ranges.get(sensor_id)
        if distance is not None:
            max_range = TOF_MAX_RANGE_M if kind is SensorKind.TOF else US_MAX_RANGE_M
            if distance > max_range:
                distance = None
        if distance is None:
            return RangeReading(
                sensor_id=sensor_id,
                kind=kind,
                timestamp_ns=int(now_ns),
                sector=sector,
                distance_m=None,
                valid=True,
                no_target=True,
                max_range_m=TOF_MAX_RANGE_M if kind is SensorKind.TOF else US_MAX_RANGE_M,
            )
        if kind is SensorKind.TOF:
            sigma = max(0.008, 0.03 * distance)
            measured = distance + self.rng.gauss(0.0, sigma)
        else:
            measured = distance * (1.0 + self.rng.gauss(0.0, 0.01)) + self.rng.gauss(0.0, 0.0056)
        return RangeReading(
            sensor_id=sensor_id,
            kind=kind,
            timestamp_ns=int(now_ns),
            sector=sector,
            distance_m=max(measured, 0.02),
            valid=True,
            no_target=False,
            max_range_m=TOF_MAX_RANGE_M if kind is SensorKind.TOF else US_MAX_RANGE_M,
        )

    def _build_frame(self, truth: Truth, captured_ns: int, frame_id: int) -> FrameDetections:
        config = self.config
        focal_px = (FRAME_HEIGHT / 2.0) / math.tan(math.radians(config.camera_vfov_deg / 2.0))
        pixels_per_degree = FRAME_WIDTH / config.imu_config.camera_hfov_deg
        pitch = config.camera_pitch_down_deg
        detections: List[Detection] = []
        for target in truth.vision:
            prior = config.class_priors.get(target.class_name)
            height_m = prior.canonical_height_m if prior is not None else 1.0
            ground_deg = math.degrees(math.atan2(config.camera_height_m, target.distance_m))
            y2 = FRAME_HEIGHT / 2.0 + focal_px * math.tan(math.radians(ground_deg - pitch))
            elevation_deg = math.degrees(
                math.atan2(height_m - config.camera_height_m, target.distance_m)
            )
            y1 = FRAME_HEIGHT / 2.0 - focal_px * math.tan(math.radians(elevation_deg + pitch))
            center_x = FRAME_WIDTH / 2.0 + target.bearing_deg * pixels_per_degree
            half_width = max(14.0, focal_px * 0.30 / target.distance_m)
            confidence = _clamp(
                target.confidence + self.rng.gauss(0.0, 0.02), 0.05, 0.99
            )
            detections.append(
                Detection(
                    class_id=target.class_id,
                    class_name=target.class_name,
                    confidence=confidence,
                    box=BoundingBox(center_x - half_width, y1, center_x + half_width, y2),
                    sector=Sector.UNKNOWN,
                )
            )
            self.frame_records.append(
                {
                    "frame_id": frame_id,
                    "captured_ns": int(captured_ns),
                    "finished_ns": int(captured_ns) + VISION_LATENCY_NS,
                    "class_name": target.class_name,
                    "distance_m": target.distance_m,
                    "bearing_deg": target.bearing_deg,
                    "center_x": center_x,
                    "half_width": half_width,
                    "raw_sector": horizontal_sector(
                        center_x, FRAME_WIDTH, config.center_fraction
                    ),
                }
            )
        return FrameDetections(
            frame_id=frame_id,
            captured_ns=int(captured_ns),
            model_started_ns=int(captured_ns) + 5 * MS,
            model_finished_ns=int(captured_ns) + VISION_LATENCY_NS,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            detections=detections,
            backend="simulator",
            model_id="sim-yolo-v8n-640",
            source="simulator",
        )

    def run(
        self,
        duration_s: float,
        truth_fn: Callable[[float, _TickContext], Truth],
    ) -> Tuple[List[_Sample], List[float]]:
        """Drive the pipeline for ``duration_s`` and return every decision."""

        tof_next = {sensor_id: 0 for sensor_id in _TOF_IDS}
        us_next = {
            "us-center-1": 0,
            "us-center-2": US_PING_OFFSET_NS,
        }
        next_frame = 0
        next_imu = 0
        pending: List[Tuple[int, FrameDetections]] = []
        context = _TickContext()
        samples: List[_Sample] = []
        timings: List[float] = []
        previous_pitch: Optional[float] = None
        previous_imu_ns: Optional[int] = None
        tick_count = int(round(duration_s * 1e9 / TICK_NS))

        for tick in range(tick_count + 1):
            now_ns = tick * TICK_NS
            pings: List[str] = []
            for sensor_id, next_ns in tof_next.items():
                if next_ns <= now_ns:
                    pings.append(sensor_id)
                    tof_next[sensor_id] = next_ns + TOF_PERIOD_NS
            for sensor_id, next_ns in us_next.items():
                if next_ns <= now_ns:
                    pings.append(sensor_id)
                    us_next[sensor_id] = next_ns + US_PING_PERIOD_NS
            for sensor_id in pings:
                context.pings[sensor_id] += 1

            truth = truth_fn(now_ns / 1e9, context)
            for sensor_id in pings:
                reading = self._range_reading(sensor_id, truth, now_ns)
                if reading is not None:
                    self.engine.update_sensor(reading)

            while next_frame <= now_ns:
                captured_ns = next_frame
                context.frame_index += 1
                frame_truth = truth_fn(captured_ns / 1e9, context)
                frame = self._build_frame(frame_truth, captured_ns, context.frame_index)
                pending.append((int(captured_ns) + VISION_LATENCY_NS, frame))
                next_frame += VISION_PERIOD_NS

            while pending and pending[0][0] <= now_ns:
                _, ready = pending.pop(0)
                self.engine.update_vision(ready)

            while next_imu <= now_ns:
                imu_truth = truth if next_imu == now_ns else truth_fn(next_imu / 1e9, context)
                pitch_rate_dps = 0.0
                if previous_pitch is not None and previous_imu_ns is not None:
                    pitch_rate_dps = (imu_truth.pitch_deg - previous_pitch) / (
                        (next_imu - previous_imu_ns) / 1e9
                    )
                self.engine.update_imu(
                    self._imu_sample(
                        t_ns=next_imu,
                        pitch_deg=imu_truth.pitch_deg,
                        pitch_rate_dps=pitch_rate_dps,
                        yaw_rate_dps=imu_truth.yaw_rate_dps,
                        gait_mps2=imu_truth.gait_mps2,
                        shock_mps2=imu_truth.shock_mps2,
                    )
                )
                previous_pitch = imu_truth.pitch_deg
                previous_imu_ns = next_imu
                next_imu += IMU_PERIOD_NS

            started = time.perf_counter()
            decision = self.engine.evaluate(now_ns)
            haptic = decision_to_directional_haptics(decision, self.config)
            timings.append((time.perf_counter() - started) * 1000.0)
            samples.append(_Sample(now_ns, dict(truth.hazards), decision, haptic))

        return samples, timings

    def imu_state_at(self, samples: Sequence[_Sample], t_ns: int) -> IMUState:
        for sample in samples:
            if sample.t_ns >= t_ns:
                return sample.decision.imu_state
        return samples[-1].decision.imu_state


@dataclass(frozen=True)
class _Metrics:
    chatter_transitions: int
    false_critical_count: int
    false_warning_count: int
    max_tick_eval_ms: float
    p95_tick_eval_ms: float


def _measure(
    rig: _Rig,
    samples: Sequence[_Sample],
    timings: Sequence[float],
    warning_limit_m: Optional[float] = None,
    critical_limit_m: Optional[float] = None,
) -> _Metrics:
    chatter = 0
    previous: Dict[Sector, int] = {}
    false_critical = 0
    false_warning = 0
    last_critical = {sector: False for sector in SECTORS}
    last_warning = {sector: False for sector in SECTORS}
    critical_limit = (
        rig.config.critical_distance_m + 0.08 if critical_limit_m is None else critical_limit_m
    )
    warning_limit = (
        rig.config.warning_distance_m + 0.12 if warning_limit_m is None else warning_limit_m
    )
    for sample in samples:
        faulted = sample.decision.overall_level == ExtendedHazardLevel.SENSOR_FAULT
        for sector in SECTORS:
            state = sample.decision.sectors[sector]
            if not faulted and state.health is not SectorHealth.FAULT:
                level = int(state.level)
                if sector in previous and level != previous[sector]:
                    chatter += 1
                previous[sector] = level
            hazard = sample.hazards.get(sector)
            is_critical = not faulted and state.level == ExtendedHazardLevel.CRITICAL
            is_warning = not faulted and state.level == ExtendedHazardLevel.WARNING
            if is_critical and not last_critical[sector]:
                if hazard is None or hazard > critical_limit:
                    false_critical += 1
            if is_warning and not last_warning[sector]:
                if hazard is None or hazard > warning_limit:
                    false_warning += 1
            last_critical[sector] = is_critical
            last_warning[sector] = is_warning
    return _Metrics(
        chatter_transitions=chatter,
        false_critical_count=false_critical,
        false_warning_count=false_warning,
        max_tick_eval_ms=max(timings) if timings else 0.0,
        p95_tick_eval_ms=_p95(timings),
    )


def _finish(
    name: str,
    rig: _Rig,
    samples: Sequence[_Sample],
    timings: Sequence[float],
    checks: Dict[str, bool],
    latency_ms: Optional[float],
    summary: str,
    metric_checks: Optional[Dict[str, Callable[[_Metrics], bool]]] = None,
    warning_limit_m: Optional[float] = None,
) -> SimulationResult:
    metrics = _measure(rig, samples, timings, warning_limit_m=warning_limit_m)
    summary = summary.replace("{chatter}", str(metrics.chatter_transitions))
    for key, predicate in (metric_checks or {}).items():
        checks[key] = bool(predicate(metrics))
    return SimulationResult(
        name=name,
        passed=all(checks.values()),
        checks=dict(checks),
        ticks=len(samples),
        chatter_transitions=metrics.chatter_transitions,
        false_critical_count=metrics.false_critical_count,
        false_warning_count=metrics.false_warning_count,
        reaction_latency_ms=None if latency_ms is None else round(latency_ms, 1),
        max_tick_eval_ms=metrics.max_tick_eval_ms,
        p95_tick_eval_ms=metrics.p95_tick_eval_ms,
        summary=summary,
    )


# --------------------------------------------------------------------------- #
# Scenario helpers
# --------------------------------------------------------------------------- #


def _center(distance_m: Optional[float]) -> Dict[str, Optional[float]]:
    return {
        "tof-center": distance_m,
        "us-center-1": distance_m,
        "us-center-2": distance_m,
    }


def _median_fused(samples: Sequence[_Sample], sector: Sector) -> Optional[float]:
    values = [
        sample.decision.sectors[sector].fused_distance_m
        for sample in samples
        if sample.decision.sectors[sector].fused_distance_m is not None
    ]
    if not values:
        return None
    return statistics.median(values)


def _fraction(samples: Sequence[_Sample], predicate: Callable[[_Sample], bool]) -> float:
    if not samples:
        return 0.0
    return sum(1 for sample in samples if predicate(sample)) / float(len(samples))


def _between(samples: Sequence[_Sample], start_s: float, end_s: float) -> List[_Sample]:
    return [sample for sample in samples if start_s <= sample.t_ns / 1e9 <= end_s]


# --------------------------------------------------------------------------- #
# 1. A wall approaching head-on down the centre corridor
# --------------------------------------------------------------------------- #


def _approach_wall_head_on(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)

    def truth(t: float, _ctx: _TickContext) -> Truth:
        distance = max(0.30, 3.2 - 0.42 * t)
        return Truth(
            ranges=_center(distance),
            vision=[VisionTarget("barrier", distance, 0.0, 0.90)],
            gait_mps2=2.4,
            hazards={Sector.LEFT: None, Sector.CENTER: distance, Sector.RIGHT: None},
        )

    samples, timings = rig.run(7.0, truth)
    advisory = _first_sample(
        samples, lambda s: _level(s, Sector.CENTER) >= ExtendedHazardLevel.ADVISORY
    )
    advisory_distance = None if advisory is None else advisory.hazards[Sector.CENTER]
    warning_latency = _latency_ms(
        samples,
        lambda s: (s.hazards[Sector.CENTER] or 99.0) <= 1.0,
        lambda s: _level(s, Sector.CENTER) >= ExtendedHazardLevel.WARNING,
    )
    critical_latency = _latency_ms(
        samples,
        lambda s: (s.hazards[Sector.CENTER] or 99.0) <= 0.5,
        lambda s: _level(s, Sector.CENTER) >= ExtendedHazardLevel.CRITICAL,
    )
    checks = {
        "vision_advisory_before_1.80m": advisory_distance is not None and advisory_distance > 1.80,
        "warning_within_120ms_of_1.00m": warning_latency is not None and warning_latency <= 120.0,
        "critical_within_120ms_of_0.50m": critical_latency is not None and critical_latency <= 120.0,
    }
    return _finish(
        "approach_wall_head_on",
        rig,
        samples,
        timings,
        checks,
        critical_latency,
        "wall 3.20 -> 0.30 m: advisory {:.2f} m, warning {:.0f} ms after 1.00 m, "
        "critical {:.0f} ms after 0.50 m".format(
            advisory_distance if advisory_distance is not None else float("nan"),
            warning_latency if warning_latency is not None else float("nan"),
            critical_latency if critical_latency is not None else float("nan"),
        ),
        {
            "no_false_critical": lambda m: m.false_critical_count == 0,
            "no_false_warning": lambda m: m.false_warning_count == 0,
            "low_chatter": lambda m: m.chatter_transitions <= 6,
        },
    )


# --------------------------------------------------------------------------- #
# 2. A narrow pole on the centreline, invisible to the wide ultrasonic beam
# --------------------------------------------------------------------------- #


def _narrow_pole_center(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)

    def truth(_t: float, _ctx: _TickContext) -> Truth:
        return Truth(
            ranges={"tof-center": 0.65, "us-center-1": 1.60, "us-center-2": 1.60},
            vision=[VisionTarget("pole", 0.65, 0.0, 0.88)],
            gait_mps2=2.0,
            hazards={Sector.LEFT: None, Sector.CENTER: 0.65, Sector.RIGHT: None},
        )

    samples, timings = rig.run(3.0, truth)
    steady = _between(samples, 0.5, 3.0)
    warning_fraction = _fraction(
        steady, lambda s: _level(s, Sector.CENTER) == ExtendedHazardLevel.WARNING
    )
    gate_used = any(
        "persistent closer return trusted" in s.decision.sectors[Sector.CENTER].reason
        for s in steady
    )
    rejected = any(
        "us-center" in s.decision.sectors[Sector.CENTER].reason
        and "ignored" in s.decision.sectors[Sector.CENTER].reason
        for s in samples
    )
    fused = _median_fused(steady, Sector.CENTER)
    checks = {
        "center_warning_sustained": warning_fraction >= 0.90,
        "never_critical": not any(
            _level(s, Sector.CENTER) == ExtendedHazardLevel.CRITICAL for s in samples
        ),
        "closer_return_trusted": gate_used,
        "ultrasonic_background_rejected": rejected,
        "fused_tracks_pole": fused is not None and 0.55 <= fused <= 0.80,
    }
    return _finish(
        "narrow_pole_center",
        rig,
        samples,
        timings,
        checks,
        None,
        "pole 0.65 m vs US background 1.60 m: warning {:.0%} of the steady window, "
        "fused {:.2f} m".format(warning_fraction, fused or float("nan")),
        {
            "no_false_critical": lambda m: m.false_critical_count == 0,
            "no_false_warning": lambda m: m.false_warning_count == 0,
        },
    )


# --------------------------------------------------------------------------- #
# 3. A glass door the ToF cannot see but the ultrasonic pair can
# --------------------------------------------------------------------------- #


def _glass_door_center(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)

    def truth(_t: float, _ctx: _TickContext) -> Truth:
        return Truth(
            ranges={"tof-center": None, "us-center-1": 0.58, "us-center-2": 0.58},
            misses={"tof-center"},
            gait_mps2=1.8,
            hazards={Sector.LEFT: None, Sector.CENTER: 0.58, Sector.RIGHT: None},
        )

    samples, timings = rig.run(3.0, truth)
    steady = _between(samples, 0.4, 3.0)
    warning_fraction = _fraction(
        steady, lambda s: _level(s, Sector.CENTER) == ExtendedHazardLevel.WARNING
    )
    tof_clear = all("tof-center" in s.decision.sectors[Sector.CENTER].fresh_sensor_ids for s in steady)
    fused = _median_fused(steady, Sector.CENTER)
    health_ok = all(s.decision.sectors[Sector.CENTER].health is SectorHealth.OK for s in steady)
    checks = {
        "tof_alive_but_clear": tof_clear,
        "center_warning_sustained": warning_fraction >= 0.90,
        "ultrasonic_consensus_distance": fused is not None and 0.50 <= fused <= 0.68,
        "center_health_ok": health_ok,
        "never_critical": not any(
            _level(s, Sector.CENTER) == ExtendedHazardLevel.CRITICAL for s in samples
        ),
    }
    return _finish(
        "glass_door_center",
        rig,
        samples,
        timings,
        checks,
        None,
        "glass 0.58 m: ToF reports clear, US pair fuses to {:.2f} m, warning {:.0%} of ticks".format(
            fused or float("nan"), warning_fraction
        ),
        {
            "no_false_critical": lambda m: m.false_critical_count == 0,
            "no_false_warning": lambda m: m.false_warning_count == 0,
        },
    )


# --------------------------------------------------------------------------- #
# 4. A single-ping ultrasonic multipath spike at a genuine 1.50 m
# --------------------------------------------------------------------------- #


def _ultrasonic_multipath_spike(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)
    spikes = [0]
    counted = [0]

    def truth(_t: float, ctx: _TickContext) -> Truth:
        us1 = 1.50
        ping = ctx.pings["us-center-1"]
        if ping % 6 == 3:
            us1 = 0.32
            if ping != counted[0]:
                counted[0] = ping
                spikes[0] += 1
        return Truth(
            ranges={"tof-center": 1.50, "us-center-1": us1, "us-center-2": 1.50},
            gait_mps2=2.0,
            hazards={Sector.LEFT: None, Sector.CENTER: 1.50, Sector.RIGHT: None},
        )

    samples, timings = rig.run(6.0, truth)
    stable = [
        s
        for s in samples
        if s.decision.sectors[Sector.CENTER].fused_distance_m is not None
        and 1.40 <= s.decision.sectors[Sector.CENTER].fused_distance_m <= 1.62
    ]
    stable_fraction = len(stable) / float(len(samples))
    checks = {
        "spikes_were_injected": spikes[0] >= 6,
        "distance_stable": stable_fraction >= 0.95,
        "never_warning": not any(
            _level(s, Sector.CENTER) >= ExtendedHazardLevel.WARNING for s in samples
        ),
    }
    return _finish(
        "ultrasonic_multipath_spike",
        rig,
        samples,
        timings,
        checks,
        None,
        "{} single-ping 0.32 m spikes at 1.50 m: {:.0%} of ticks stay inside "
        "1.40-1.62 m".format(spikes[0], stable_fraction),
        {
            "zero_false_critical": lambda m: m.false_critical_count == 0,
            "zero_false_warning": lambda m: m.false_warning_count == 0,
        },
    )


# --------------------------------------------------------------------------- #
# 5. Three sectors with three different hazards at the same instant
# --------------------------------------------------------------------------- #


def _multi_sector_simultaneous(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)

    def truth(_t: float, _ctx: _TickContext) -> Truth:
        return Truth(
            ranges={
                "tof-left": 0.35,
                "tof-center": None,
                "tof-right": 0.80,
                "us-center-1": 2.50,
                "us-center-2": 2.50,
            },
            gait_mps2=2.2,
            hazards={Sector.LEFT: 0.35, Sector.CENTER: 2.50, Sector.RIGHT: 0.80},
        )

    samples, timings = rig.run(2.5, truth)
    steady = _between(samples, 0.6, 2.5)
    left_ok = all(
        s.decision.sectors[Sector.LEFT].level == ExtendedHazardLevel.CRITICAL
        and s.haptic.motors[Sector.LEFT].intensity == 1.0
        and s.haptic.motors[Sector.LEFT].pulse_period_ms == 100
        for s in steady
    )
    right_ok = all(
        s.decision.sectors[Sector.RIGHT].level == ExtendedHazardLevel.WARNING
        and 0.60 <= s.haptic.motors[Sector.RIGHT].intensity <= 0.88
        for s in steady
    )
    center_ok = all(
        s.decision.sectors[Sector.CENTER].level == ExtendedHazardLevel.CLEAR
        and s.haptic.motors[Sector.CENTER].intensity == 0.0
        for s in steady
    )
    checks = {
        "left_motor_critical_1.0_100ms": left_ok,
        "right_motor_warning_0.6_0.88": right_ok,
        "center_motor_clear_0.0": center_ok,
        "overall_critical": all(
            s.decision.overall_level == ExtendedHazardLevel.CRITICAL for s in steady
        ),
    }
    right_intensity = steady[-1].haptic.motors[Sector.RIGHT].intensity
    return _finish(
        "multi_sector_simultaneous",
        rig,
        samples,
        timings,
        checks,
        None,
        "left 0.35 m critical, right 0.80 m warning (intensity {:.2f}), centre clear".format(
            right_intensity
        ),
        {
            "no_false_critical": lambda m: m.false_critical_count == 0,
            "no_false_warning": lambda m: m.false_warning_count == 0,
        },
    )


# --------------------------------------------------------------------------- #
# 6. Stairs ahead inside the range sensors' blind cone
# --------------------------------------------------------------------------- #


def _range_blind_stairs_ahead(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)
    start_s = 0.2
    approach_s = 2.5

    def truth(t: float, _ctx: _TickContext) -> Truth:
        progress = _clamp((t - start_s) / approach_s, 0.0, 1.0)
        distance = 2.2 - 0.9 * progress
        return Truth(
            ranges=_center(None),
            vision=[VisionTarget("stairs", distance, 0.0, 0.88)],
            gait_mps2=2.4,
            hazards={Sector.LEFT: None, Sector.CENTER: distance, Sector.RIGHT: None},
        )

    samples, timings = rig.run(4.0, truth)
    advisory = _first_sample(
        samples, lambda s: _level(s, Sector.CENTER) >= ExtendedHazardLevel.ADVISORY
    )
    warning = _first_sample(
        samples, lambda s: _level(s, Sector.CENTER) >= ExtendedHazardLevel.WARNING
    )
    labels = any(
        "stairs" in s.decision.sectors[Sector.CENTER].semantic_labels for s in samples
    )
    range_blind = all(
        s.decision.sectors[Sector.CENTER].fused_distance_m is None for s in samples
    )
    checks = {
        "vision_advisory_seen": advisory is not None,
        "vision_warning_seen": warning is not None,
        "never_vision_critical": not any(
            s.decision.sectors[Sector.CENTER].level >= ExtendedHazardLevel.CRITICAL
            for s in samples
        ),
        "range_sensors_blind": range_blind,
        "stairs_label_present": labels,
    }
    return _finish(
        "range_blind_stairs_ahead",
        rig,
        samples,
        timings,
        checks,
        None,
        "stairs 2.20 -> 1.30 m, both rangers blind: advisory at {:.2f} m, warning at {:.2f} m, "
        "never critical".format(
            advisory.hazards[Sector.CENTER] if advisory else float("nan"),
            warning.hazards[Sector.CENTER] if warning else float("nan"),
        ),
        {
            "no_false_critical": lambda m: m.false_critical_count == 0,
            "no_false_warning": lambda m: m.false_warning_count == 0,
        },
        warning_limit_m=2.05,
    )


# --------------------------------------------------------------------------- #
# 7. A distant vehicle on the left and a one-frame person flicker on the right
# --------------------------------------------------------------------------- #


def _long_range_vehicle_and_flicker(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)
    flicker_frame = 25

    def truth(_t: float, ctx: _TickContext) -> Truth:
        targets = [VisionTarget("vehicle", 2.60, -25.0, 0.93)]
        if ctx.frame_index == flicker_frame:
            targets.append(VisionTarget("person", 2.40, 28.0, 0.91))
        return Truth(
            ranges={"tof-left": None, "tof-center": None, "tof-right": None},
            vision=targets,
            gait_mps2=2.2,
            hazards={Sector.LEFT: 2.60, Sector.CENTER: None, Sector.RIGHT: None},
        )

    samples, timings = rig.run(3.0, truth)
    left_advisory = _first_sample(
        samples, lambda s: _level(s, Sector.LEFT) == ExtendedHazardLevel.ADVISORY
    )
    left_advisory_ms = None if left_advisory is None else left_advisory.t_ms
    right_clear = all(
        _level(s, Sector.RIGHT) == ExtendedHazardLevel.CLEAR for s in samples
    )
    flicker_delivered = any(
        record["class_name"] == "person" and record["frame_id"] == flicker_frame
        for record in rig.frame_records
    )
    checks = {
        "left_advisory_within_400ms": left_advisory_ms is not None and left_advisory_ms <= 400.0,
        "left_never_warning_or_critical": all(
            _level(s, Sector.LEFT) <= ExtendedHazardLevel.ADVISORY for s in samples
        ),
        "right_flicker_ignored": right_clear,
        "flicker_frame_was_simulated": flicker_delivered,
    }
    return _finish(
        "long_range_vehicle_and_flicker",
        rig,
        samples,
        timings,
        checks,
        left_advisory_ms,
        "vehicle 2.60 m left: advisory at {:.0f} ms; single-frame right flicker: 0 alerts".format(
            left_advisory_ms if left_advisory_ms is not None else float("nan")
        ),
        {
            "no_false_critical": lambda m: m.false_critical_count == 0,
            "no_false_warning": lambda m: m.false_warning_count == 0,
        },
    )


# --------------------------------------------------------------------------- #
# 8. A noisy target hovering on the critical and warning boundaries
# --------------------------------------------------------------------------- #


def _boundary_hover_chatter(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)

    def truth(t: float, _ctx: _TickContext) -> Truth:
        if t < 2.0:
            distance = 0.50 + 0.03 * math.sin(2.0 * math.pi * 2.5 * t)
        else:
            distance = 1.00 + 0.04 * math.sin(2.0 * math.pi * 2.5 * t)
        return Truth(
            ranges=_center(distance),
            gait_mps2=2.0,
            hazards={Sector.LEFT: None, Sector.CENTER: distance, Sector.RIGHT: None},
        )

    samples, timings = rig.run(4.2, truth)
    phase_a = _between(samples, 0.4, 1.95)
    phase_b = _between(samples, 2.6, 4.2)
    critical_held = all(
        s.decision.sectors[Sector.CENTER].level == ExtendedHazardLevel.CRITICAL for s in phase_a
    )
    warning_held = all(
        s.decision.sectors[Sector.CENTER].level == ExtendedHazardLevel.WARNING for s in phase_b
    )
    checks = {
        "hysteresis_minimal_transitions": True,
        "critical_held_while_hovering_0.50": critical_held,
        "warning_held_while_hovering_1.00": warning_held,
    }
    return _finish(
        "boundary_hover_chatter",
        rig,
        samples,
        timings,
        checks,
        None,
        "0.50 +/- 0.03 m then 1.00 +/- 0.04 m for 2 s each: {chatter} transitions",
        {"hysteresis_minimal_transitions": lambda m: m.chatter_transitions <= 3},
    )


# --------------------------------------------------------------------------- #
# 9. A rightward cane sweep with a person dead ahead
# --------------------------------------------------------------------------- #


def _cane_sweep_imu_yaw_comp(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)
    sweep_start_s = 0.05
    sweep_s = 0.40
    yaw_rate_dps = -60.0
    person_m = 1.20

    def truth(t: float, _ctx: _TickContext) -> Truth:
        elapsed = _clamp(t - sweep_start_s, 0.0, sweep_s)
        bearing = 17.5 + yaw_rate_dps * elapsed
        return Truth(
            ranges=_center(None),
            vision=[VisionTarget("person", person_m, bearing, 0.92)],
            yaw_rate_dps=yaw_rate_dps if 0.0 <= t - sweep_start_s <= sweep_s else 0.0,
            gait_mps2=2.0,
            hazards={Sector.LEFT: None, Sector.CENTER: person_m, Sector.RIGHT: None},
        )

    samples, timings = rig.run(1.5, truth)
    person_frames = [
        record for record in rig.frame_records if record["class_name"] == "person"
    ]
    compensated_center = 0
    for record in person_frames:
        state = rig.imu_state_at(samples, record["finished_ns"])
        age_s = max(0.0, (state.timestamp_ns - record["captured_ns"]) / 1e9)
        compensated_x = rig.engine.imu.compensate_vision_center_x(
            record["center_x"], FRAME_WIDTH, age_s, state
        )
        record["compensated_x"] = compensated_x
        if horizontal_sector(compensated_x, FRAME_WIDTH, rig.config.center_fraction) is Sector.CENTER:
            compensated_center += 1
    raw_outside = [
        record for record in person_frames if record["raw_sector"] is not Sector.CENTER
    ]
    sweep_window = _between(samples, sweep_start_s + 0.1, sweep_start_s + sweep_s)
    center_label_fraction = _fraction(
        sweep_window,
        lambda s: "person" in s.decision.sectors[Sector.CENTER].semantic_labels,
    )
    side_alerts = [
        s
        for s in samples
        if _level(s, Sector.LEFT) != ExtendedHazardLevel.CLEAR
        or _level(s, Sector.RIGHT) != ExtendedHazardLevel.CLEAR
    ]
    checks = {
        "yaw_lag_pushed_raw_box_sideways": len(raw_outside) >= 1,
        "compensation_keeps_every_frame_center": compensated_center == len(person_frames),
        "center_carries_person_label": center_label_fraction >= 0.85,
        "no_side_alerts": not side_alerts,
        "center_advisory_seen": any(
            _level(s, Sector.CENTER) >= ExtendedHazardLevel.ADVISORY for s in samples
        ),
    }
    max_offset = max(
        (abs(record["center_x"] - FRAME_WIDTH / 2.0) for record in person_frames),
        default=0.0,
    )
    return _finish(
        "cane_sweep_imu_yaw_comp",
        rig,
        samples,
        timings,
        checks,
        None,
        "60 deg/s sweep: {} of {} frames landed off-centre before compensation, all "
        "compensated into centre (max raw offset {:.0f} px)".format(
            len(raw_outside), len(person_frames), max_offset
        ),
        {
            "no_false_critical": lambda m: m.false_critical_count == 0,
            "no_false_warning": lambda m: m.false_warning_count == 0,
        },
    )


# --------------------------------------------------------------------------- #
# 10. A 30 degree downward tilt full of floor echoes
# --------------------------------------------------------------------------- #


def _steep_downward_tilt_ground_echo(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)
    ramp_start_s = 1.5
    ramp_end_s = 2.0
    obstacle_s = 3.5
    sensor_height_m = rig.config.imu_config.sensor_height_m

    def truth(t: float, _ctx: _TickContext) -> Truth:
        progress = _clamp((t - ramp_start_s) / (ramp_end_s - ramp_start_s), 0.0, 1.0)
        pitch = IMU_BASELINE_PITCH_DEG + 30.0 * progress
        if t >= obstacle_s:
            return Truth(
                ranges=_center(0.30),
                pitch_deg=pitch,
                gait_mps2=0.0,
                hazards={Sector.LEFT: None, Sector.CENTER: 0.30, Sector.RIGHT: None},
            )
        if progress <= 0.0:
            return Truth(
                ranges=_center(None),
                pitch_deg=pitch,
                gait_mps2=0.0,
                hazards={Sector.LEFT: None, Sector.CENTER: None, Sector.RIGHT: None},
            )
        floor = sensor_height_m / math.sin(math.radians(pitch))
        return Truth(
            ranges=_center(floor),
            pitch_deg=pitch,
            gait_mps2=0.0,
            hazards={Sector.LEFT: None, Sector.CENTER: None, Sector.RIGHT: None},
        )

    samples, timings = rig.run(4.0, truth)
    before_obstacle = [s for s in samples if s.t_ns / 1e9 < obstacle_s]
    echo_window = _between(samples, 2.6, 3.4)
    suppressed = all(
        s.decision.sectors[Sector.CENTER].fused_distance_m is None
        and "floor echo" in s.decision.sectors[Sector.CENTER].reason
        for s in echo_window
    )
    critical_latency = _latency_ms(
        samples,
        lambda s: s.t_ns / 1e9 >= obstacle_s,
        lambda s: _level(s, Sector.CENTER) == ExtendedHazardLevel.CRITICAL,
    )
    pitched = rig.imu_state_at(samples, int(3.0e9))
    first_critical = _first_sample(
        samples,
        lambda s: s.t_ns / 1e9 >= obstacle_s
        and _level(s, Sector.CENTER) == ExtendedHazardLevel.CRITICAL,
    )
    safety_floor = first_critical is not None and (
        first_critical.hazards[Sector.CENTER] or 99.0
    ) < 0.35
    checks = {
        "no_warning_from_floor_echo": all(
            _level(s, Sector.CENTER) < ExtendedHazardLevel.WARNING for s in before_obstacle
        ),
        "floor_return_suppressed": suppressed,
        "cane_pitch_tracked": pitched.relative_pitch_deg > 25.0,
        "real_obstacle_critical_within_150ms": critical_latency is not None
        and critical_latency <= 150.0,
        "obstacle_below_0.35m_floor_still_critical": safety_floor,
    }
    return _finish(
        "steep_downward_tilt_ground_echo",
        rig,
        samples,
        timings,
        checks,
        critical_latency,
        "42 deg shaft (30 deg relative), floor echo {:.2f} m suppressed, 0.30 m obstacle "
        "critical in {:.0f} ms".format(
            sensor_height_m / math.sin(math.radians(42.0)),
            critical_latency if critical_latency is not None else float("nan"),
        ),
        {
            "no_false_critical": lambda m: m.false_critical_count == 0,
            "no_false_warning": lambda m: m.false_warning_count == 0,
        },
    )


# --------------------------------------------------------------------------- #
# 11. The left ToF dies mid-walk after a knock
# --------------------------------------------------------------------------- #


def _partial_sensor_dropout(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)

    def truth(t: float, _ctx: _TickContext) -> Truth:
        ranges: Dict[str, Optional[float]] = {"tof-center": 0.90, "tof-right": 2.60}
        silent: Set[str] = set()
        if t < 1.5:
            ranges["tof-left"] = 1.20
        else:
            silent.add("tof-left")
        if t >= 2.0:
            ranges["tof-right"] = 0.60
        shock = 45.0 if 1.5 <= t < 1.53 else 0.0
        return Truth(
            ranges=ranges,
            silent=silent,
            gait_mps2=2.2,
            shock_mps2=shock,
            hazards={
                Sector.LEFT: 1.20 if t < 1.5 else None,
                Sector.CENTER: 0.90,
                Sector.RIGHT: 0.60 if t >= 2.0 else 2.60,
            },
        )

    samples, timings = rig.run(4.0, truth)
    after_loss = _between(samples, 1.9, 4.0)
    center_window = _between(samples, 0.5, 4.0)
    right_window = _between(samples, 2.4, 4.0)
    left_fault = all(
        s.decision.sectors[Sector.LEFT].health is SectorHealth.FAULT
        and s.decision.sectors[Sector.LEFT].level == ExtendedHazardLevel.SENSOR_FAULT
        for s in after_loss
    )
    degraded_reported = all("tof-left" in s.decision.degraded_sensors for s in after_loss)
    center_ok = all(
        s.decision.sectors[Sector.CENTER].health is SectorHealth.OK
        and s.decision.sectors[Sector.CENTER].level == ExtendedHazardLevel.WARNING
        for s in center_window
    )
    right_ok = all(
        s.decision.sectors[Sector.RIGHT].level == ExtendedHazardLevel.WARNING for s in right_window
    )
    pitch_peak = max(
        abs(s.decision.imu_state.relative_pitch_deg) for s in _between(samples, 1.4, 1.9)
    )
    checks = {
        "left_flagged_fault": left_fault,
        "left_reported_degraded": degraded_reported,
        "center_still_warning": center_ok,
        "right_still_detects": right_ok,
        "overall_not_sensor_fault": all(
            s.decision.overall_level != ExtendedHazardLevel.SENSOR_FAULT for s in samples
        ),
        "knock_does_not_corrupt_pitch": pitch_peak < 6.0,
    }
    return _finish(
        "partial_sensor_dropout",
        rig,
        samples,
        timings,
        checks,
        None,
        "tof-left silent at 1.50 s: left fault, centre warning 0.90 m, right warning 0.60 m, "
        "pitch drift {:.1f} deg".format(pitch_peak),
        {
            "no_false_critical": lambda m: m.false_critical_count == 0,
            "no_false_warning": lambda m: m.false_warning_count == 0,
        },
    )


# --------------------------------------------------------------------------- #
# 12. Every range sensor stops answering, then recovers
# --------------------------------------------------------------------------- #


def _total_range_blackout_and_recovery(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)
    loss_s = 1.0
    recovery_s = 2.5

    def truth(t: float, _ctx: _TickContext) -> Truth:
        if loss_s <= t < recovery_s:
            return Truth(silent=set(_ALL_SENSORS), gait_mps2=2.0)
        distance = 1.20 if t < loss_s else 1.40
        return Truth(
            ranges=_center(distance),
            gait_mps2=2.0,
            hazards={Sector.LEFT: None, Sector.CENTER: distance, Sector.RIGHT: None},
        )

    samples, timings = rig.run(4.0, truth)
    fault_samples = [
        s for s in samples if s.decision.overall_level == ExtendedHazardLevel.SENSOR_FAULT
    ]
    first_fault = fault_samples[0] if fault_samples else None
    fault_latency = (
        None if first_fault is None else (first_fault.t_ns / 1e9 - loss_s) * 1000.0
    )
    motors_fault = bool(fault_samples) and all(
        fault_samples[0].haptic.motors[sector].level == ExtendedHazardLevel.SENSOR_FAULT
        and fault_samples[0].haptic.motors[sector].intensity == 0.35
        and fault_samples[0].haptic.motors[sector].pulse_period_ms == 1000
        for sector in SECTORS
    )
    before_loss = [s for s in samples if s.t_ns / 1e9 < loss_s]
    after_recovery = [s for s in samples if s.t_ns / 1e9 >= recovery_s]
    clear_latency = _latency_ms(
        samples,
        lambda s: s.t_ns / 1e9 >= recovery_s,
        lambda s: s.decision.overall_level != ExtendedHazardLevel.SENSOR_FAULT,
    )
    checks = {
        "fault_within_300ms": fault_latency is not None and fault_latency <= 300.0,
        "all_three_motors_fault_pattern": motors_fault,
        "no_fault_before_loss": all(
            s.decision.overall_level != ExtendedHazardLevel.SENSOR_FAULT for s in before_loss
        ),
        "fault_clears_on_recovery": clear_latency is not None and clear_latency <= 120.0,
        "healthy_again_after_recovery": all(
            s.decision.fresh_range_sensor_count > 0 for s in after_recovery
        ),
    }
    return _finish(
        "total_range_blackout_and_recovery",
        rig,
        samples,
        timings,
        checks,
        fault_latency,
        "blackout at 1.00 s: SENSOR_FAULT after {:.0f} ms on all 3 motors, cleared in "
        "{:.0f} ms".format(
            fault_latency if fault_latency is not None else float("nan"),
            clear_latency if clear_latency is not None else float("nan"),
        ),
        {
            "no_false_critical_before_loss": lambda m: m.false_critical_count == 0,
        },
    )


# --------------------------------------------------------------------------- #
# 13. A fast-approaching obstacle with a short time to contact
# --------------------------------------------------------------------------- #


def _fast_approaching_ttc(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)
    start_s = 0.5
    speed_mps = 1.5

    def truth(t: float, _ctx: _TickContext) -> Truth:
        elapsed = max(0.0, t - start_s)
        distance = max(1.08, 1.55 - speed_mps * elapsed)
        return Truth(
            ranges=_center(distance),
            gait_mps2=2.0,
            hazards={Sector.LEFT: None, Sector.CENTER: distance, Sector.RIGHT: None},
        )

    samples, timings = rig.run(2.5, truth)
    warning = _first_sample(
        samples, lambda s: _level(s, Sector.CENTER) >= ExtendedHazardLevel.WARNING
    )
    warning_distance = None if warning is None else warning.hazards[Sector.CENTER]
    warning_ttc = None if warning is None else warning.decision.sectors[Sector.CENTER].ttc_s
    warning_velocity = (
        None if warning is None else warning.decision.sectors[Sector.CENTER].closing_velocity_mps
    )
    latency = _latency_ms(
        samples,
        lambda s: s.t_ns / 1e9 >= start_s and (s.hazards[Sector.CENTER] or 99.0) <= 1.275,
        lambda s: _level(s, Sector.CENTER) >= ExtendedHazardLevel.WARNING,
    )
    checks = {
        "warning_before_1.0m": warning_distance is not None and warning_distance > 1.0,
        "ttc_below_0.85s_at_escalation": warning_ttc is not None and warning_ttc < 0.85,
        "closing_velocity_measured": warning_velocity is not None and warning_velocity > 1.0,
        "escalation_within_250ms": latency is not None and latency <= 250.0,
        "never_critical": not any(
            _level(s, Sector.CENTER) == ExtendedHazardLevel.CRITICAL for s in samples
        ),
    }
    return _finish(
        "fast_approaching_ttc",
        rig,
        samples,
        timings,
        checks,
        latency,
        "1.50 m/s from 1.55 m: warning at {:.2f} m with TTC {:.2f} s closing {:.2f} m/s".format(
            warning_distance if warning_distance is not None else float("nan"),
            warning_ttc if warning_ttc is not None else float("nan"),
            warning_velocity if warning_velocity is not None else float("nan"),
        ),
        {
            "no_false_critical": lambda m: m.false_critical_count == 0,
        },
    )


# --------------------------------------------------------------------------- #
# 14. Standing still against a wall: anti-fatigue softening
# --------------------------------------------------------------------------- #


def _stationary_anti_fatigue(seed: int, config: Optional[SmartCaneFusionConfig]) -> SimulationResult:
    rig = _Rig(seed, config)
    move_s = 2.5

    def truth(t: float, _ctx: _TickContext) -> Truth:
        distance = 0.80 if t < move_s else 0.40
        return Truth(
            ranges=_center(distance),
            gait_mps2=0.0,
            stationary=True,
            hazards={Sector.LEFT: None, Sector.CENTER: distance, Sector.RIGHT: None},
        )

    samples, timings = rig.run(4.2, truth)
    standing = _between(samples, 2.0, 2.45)
    stationary_flag = all(s.decision.imu_state.is_stationary for s in standing)
    warning_held = all(
        s.decision.sectors[Sector.CENTER].level == ExtendedHazardLevel.WARNING for s in standing
    )
    intensities = [s.haptic.motors[Sector.CENTER].intensity for s in standing]
    softened = all(0.45 <= value <= 0.58 for value in intensities)
    critical_latency = _latency_ms(
        samples,
        lambda s: s.t_ns / 1e9 >= move_s,
        lambda s: _level(s, Sector.CENTER) == ExtendedHazardLevel.CRITICAL,
    )
    critical_motor = [
        s.haptic.motors[Sector.CENTER] for s in samples if s.t_ns / 1e9 >= move_s + 0.15
    ][:1]
    checks = {
        "stationary_detected": stationary_flag,
        "warning_held_at_0.80m": warning_held,
        "intensity_softened": softened,
        "critical_after_move_within_150ms": critical_latency is not None
        and critical_latency <= 150.0,
        "critical_motor_never_softened": bool(critical_motor)
        and critical_motor[0].intensity == 1.0
        and critical_motor[0].pulse_period_ms == 100,
    }
    return _finish(
        "stationary_anti_fatigue",
        rig,
        samples,
        timings,
        checks,
        critical_latency,
        "standing 3 s at 0.80 m: warning intensity {:.2f} (softened), 0.40 m move -> "
        "critical in {:.0f} ms".format(
            statistics.mean(intensities) if intensities else float("nan"),
            critical_latency if critical_latency is not None else float("nan"),
        ),
        {
            "no_false_critical": lambda m: m.false_critical_count == 0,
        },
    )


_SCENARIO_BUILDERS: Tuple[Callable[[int, Optional[SmartCaneFusionConfig]], SimulationResult], ...] = (
    _approach_wall_head_on,
    _narrow_pole_center,
    _glass_door_center,
    _ultrasonic_multipath_spike,
    _multi_sector_simultaneous,
    _range_blind_stairs_ahead,
    _long_range_vehicle_and_flicker,
    _boundary_hover_chatter,
    _cane_sweep_imu_yaw_comp,
    _steep_downward_tilt_ground_echo,
    _partial_sensor_dropout,
    _total_range_blackout_and_recovery,
    _fast_approaching_ttc,
    _stationary_anti_fatigue,
)


def run_all_scenarios(
    seed: int = 42,
    config: Optional[SmartCaneFusionConfig] = None,
) -> List[SimulationResult]:
    """Run every scenario once and return the fourteen scorecards in order."""

    return [
        builder(seed * 1000 + index, config)
        for index, builder in enumerate(_SCENARIO_BUILDERS)
    ]
