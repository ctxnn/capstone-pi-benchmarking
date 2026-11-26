"""Who decides at which distance.

* Close (up to ``fusion.warning_distance_m``): the ToF sensors and the forward
  ultrasonic decide, exactly as the range-first fusion engine always did. A
  camera detection cannot override a range sensor in this band.
* Far (beyond that, out to ``priority.far_max_m``): the camera decides. The
  range sensors say nothing out here, so a confirmed detection pulses the motor
  on the side where the object is, gently.
* Overhead: the second ultrasonic (straight along the shaft, so it looks up on
  a leaning cane) warns about head-height obstacles on the centre motor. It
  never mixes with the ground-level ranges.
* The IMU takes no part here. It has its own one-shot pulse (see ``imu.py``).

``fusion.py`` is untouched: the forward sensors still go through
``FusionEngine``; this module wraps it and adds the camera and overhead rules.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..fusion import (
    FusionConfig,
    FusionEngine,
    HapticCommand,
    HazardDecision,
    HazardLevel,
    SensorReading,
    decision_to_haptic,
)
from ..imu import DEG_TO_RAD, GRAVITY_MPS2, IMUProcessor, IMUSample, IMUState
from ..types import Detection, FrameDetections, Sector
from .config import HapticsConfig, PriorityConfig
from .patterns import urgency_for

SOURCE_RANGE = "range"
SOURCE_OVERHEAD = "overhead"
SOURCE_VISION = "vision"
SOURCE_FAULT = "fault"

_REAL_SECTORS = (Sector.LEFT, Sector.CENTER, Sector.RIGHT)


@dataclass(frozen=True)
class CaneDecision(HazardDecision):
    """A fusion decision that also says which sense produced it."""

    source: str = SOURCE_RANGE

    def as_dict(self) -> Dict[str, object]:
        payload = super().as_dict()
        payload["source"] = self.source
        return payload


def _tag(decision: HazardDecision, source: str) -> CaneDecision:
    return CaneDecision(
        timestamp_ns=decision.timestamp_ns,
        level=decision.level,
        sectors=decision.sectors,
        nearest_distance_m=decision.nearest_distance_m,
        semantic_labels=decision.semantic_labels,
        range_triggered=decision.range_triggered,
        fresh_range_sensor_count=decision.fresh_range_sensor_count,
        reason=decision.reason,
        source=source,
    )


def estimate_distance_m(
    detection: Detection, frame_width: int, frame_height: int, config: PriorityConfig
) -> Optional[float]:
    """Distance from the box height and the class's typical real height (pinhole model).

    This is an estimate: it assumes the object is a typical size and the box
    spans all of it. A box clipped by the top or bottom of the frame is taller
    in reality than it looks, which would put the estimate too far away, so such
    boxes are pulled in to ``far_max_m`` and still raise an alert.
    """

    box_height = detection.box.y2 - detection.box.y1
    if frame_height <= 0 or box_height < 2.0:
        return None
    focal_px = (frame_height / 2.0) / math.tan(math.radians(config.vertical_fov_deg) / 2.0)
    distance = focal_px * config.height_for(detection.class_name) / box_height
    clipped = (
        detection.box.y1 <= config.edge_margin_px
        or detection.box.y2 >= frame_height - config.edge_margin_px
    )
    if clipped:
        distance = min(distance, config.far_max_m)
    return distance


class PriorityFusion:
    """Not thread-safe; wrap it in a lock (``LockedFusion``) when sensors write from threads."""

    def __init__(
        self,
        fusion: Optional[FusionConfig] = None,
        priority: Optional[PriorityConfig] = None,
        overhead_ids: Iterable[str] = (),
        *,
        ground_ids: Iterable[str] = (),
        sensor_pitch_offsets_deg: Optional[Mapping[str, float]] = None,
        expected_by_sector: Optional[Mapping[Sector, Sequence[str]]] = None,
    ) -> None:
        self.fusion = fusion or FusionConfig()
        self.priority = priority or PriorityConfig()
        self._engine = FusionEngine(self.fusion)
        self._overhead_ids = frozenset(overhead_ids)
        self._ground_ids = frozenset(ground_ids)
        offsets: Dict[str, float] = {sid: 8.0 for sid in self._ground_ids}
        if sensor_pitch_offsets_deg:
            for sid, val in sensor_pitch_offsets_deg.items():
                offsets[str(sid)] = float(val)
        self._pitch_offsets = offsets
        self._expected: Dict[Sector, Tuple[str, ...]] = {
            sector: tuple(ids)
            for sector, ids in (expected_by_sector or {}).items()
            if ids
        }
        self._last_seen_ns: Dict[str, int] = {}
        self._imu = IMUProcessor()
        self._imu.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=self._imu.config.nominal_pitch_deg)
        self._overhead: Dict[str, SensorReading] = {}
        self._history: Dict[str, Deque[Tuple[int, float]]] = {}
        self._sector_track: Dict[Sector, Deque[Tuple[int, float]]] = {
            sector: deque(maxlen=6) for sector in _REAL_SECTORS
        }
        self._vision: Optional[FrameDetections] = None
        self._estimates: List[Tuple[Detection, float]] = []
        self._streak: Dict[Sector, int] = {sector: 0 for sector in _REAL_SECTORS}

    # -- inputs ------------------------------------------------------------------

    def update_imu(self, sample: Any, now_ns: Optional[int] = None) -> IMUState:
        """Accept either ``yolo_pi.imu.IMUSample`` (SI) or ``yolo_pi.cane.imu.ImuSample`` (g, dps)."""

        if isinstance(sample, IMUSample):
            return self._imu.update(sample)
        ts = int(now_ns if now_ns is not None else getattr(sample, "timestamp_ns", 0))
        accel_g = getattr(sample, "accel_g", (0.0, 0.0, -1.0))
        gyro_dps = getattr(sample, "gyro_dps", (0.0, 0.0, 0.0))
        temp_c = getattr(sample, "temperature_c", None)
        si_sample = IMUSample(
            timestamp_ns=ts,
            ax_mps2=float(accel_g[0]) * GRAVITY_MPS2,
            ay_mps2=float(accel_g[1]) * GRAVITY_MPS2,
            az_mps2=float(accel_g[2]) * GRAVITY_MPS2,
            gx_rps=float(gyro_dps[0]) * DEG_TO_RAD,
            gy_rps=float(gyro_dps[1]) * DEG_TO_RAD,
            gz_rps=float(gyro_dps[2]) * DEG_TO_RAD,
            temp_c=None if temp_c is None else float(temp_c),
        )
        return self._imu.update(si_sample)

    def _filter_forward_reading(self, reading: SensorReading) -> SensorReading:
        if not reading.usable_range() or reading.distance_m is None:
            return reading
        raw_d = float(reading.distance_m)
        ts = reading.timestamp_ns
        imu_state = self._imu.state(ts)
        offset = self._pitch_offsets.get(reading.sensor_id, 0.0)
        if self._imu.is_likely_ground_echo(raw_d, imu_state, sensor_pitch_offset_deg=offset):
            clear_d = max(raw_d, self.fusion.warning_distance_m + 0.6)
            return replace(reading, distance_m=clear_d)

        hist = self._history.get(reading.sensor_id)
        if hist is None:
            hist = deque(maxlen=3)
            self._history[reading.sensor_id] = hist
        if hist and ts <= hist[-1][0]:
            hist[-1] = (ts, raw_d)
        else:
            hist.append((ts, raw_d))

        filtered_d = raw_d
        if len(hist) >= 3:
            d0, d1, d2 = hist[0][1], hist[1][1], hist[2][1]
            median_d = sorted((d0, d1, d2))[1]
            margin = max(0.08, 0.04 * median_d)
            two_closer = (d2 < median_d - margin) and (d1 < median_d - margin)
            if not two_closer and abs(d2 - median_d) > 0.20:
                filtered_d = median_d

        if reading.sector in _REAL_SECTORS:
            track = self._sector_track[reading.sector]
            if track and ts == track[-1][0]:
                track[-1] = (ts, min(track[-1][1], filtered_d))
            elif not track or ts > track[-1][0]:
                track.append((ts, filtered_d))

        if filtered_d != raw_d:
            return replace(reading, distance_m=filtered_d)
        return reading

    def update_sensor(self, reading: SensorReading) -> None:
        if reading.valid:
            previous_ns = self._last_seen_ns.get(reading.sensor_id, -1)
            if reading.timestamp_ns >= previous_ns:
                self._last_seen_ns[reading.sensor_id] = reading.timestamp_ns
        if reading.sensor_id in self._overhead_ids:
            previous = self._overhead.get(reading.sensor_id)
            if previous is None or reading.timestamp_ns >= previous.timestamp_ns:
                self._overhead[reading.sensor_id] = reading
        else:
            self._engine.update_sensor(self._filter_forward_reading(reading))

    def blind_sectors(self, now_ns: int) -> Tuple[Sector, ...]:
        """Sectors whose every expected range sensor is stale or invalid."""

        stale_ns = int(self.fusion.sensor_stale_after_ms * 1_000_000)
        blind: List[Sector] = []
        for sector, sensor_ids in self._expected.items():
            alive = any(
                0 <= now_ns - self._last_seen_ns.get(sensor_id, -10**18) <= stale_ns
                for sensor_id in sensor_ids
            )
            if not alive:
                blind.append(sector)
        return tuple(sorted(blind, key=lambda item: item.value))

    def update_sensors(self, readings: Iterable[SensorReading]) -> None:
        for reading in readings:
            self.update_sensor(reading)

    def update_vision(self, detections: FrameDetections) -> None:
        if self._vision is not None and detections.model_finished_ns < self._vision.model_finished_ns:
            return
        self._engine.update_vision(detections)  # labels for near range alerts
        self._vision = detections
        self._estimates = []
        for detection in detections.detections:
            if detection.sector not in _REAL_SECTORS:
                continue
            if detection.confidence < self.priority.vision_min_confidence:
                continue
            distance = estimate_distance_m(detection, detections.width, detections.height, self.priority)
            if distance is not None and distance <= self.priority.far_max_m:
                self._estimates.append((detection, distance))
        seen = {detection.sector for detection, _ in self._estimates}
        for sector in _REAL_SECTORS:
            self._streak[sector] = self._streak[sector] + 1 if sector in seen else 0

    # -- decision ----------------------------------------------------------------

    def _ttc_alert(self, now_ns: int, forward: HazardDecision) -> Optional[CaneDecision]:
        if forward.level != HazardLevel.CLEAR:
            return None
        imu_state = self._imu.state(now_ns)
        if imu_state.fresh and (imu_state.is_pitched_down or imu_state.relative_pitch_deg > 10.0):
            return None
        stale_ns = int(self.fusion.sensor_stale_after_ms * 1_000_000)
        triggered_sectors: List[Sector] = []
        best_d: Optional[float] = None
        best_ttc: Optional[float] = None
        for sector in _REAL_SECTORS:
            track = self._sector_track[sector]
            if len(track) < 3:
                continue
            t_last, d_last = track[-1]
            if not (0 <= now_ns - t_last <= stale_ns):
                continue
            if not (self.fusion.warning_distance_m < d_last <= 1.55):
                continue
            t_first, d_first = track[-3]
            dt_s = (t_last - t_first) / 1e9
            if not (0.03 <= dt_s <= 0.60):
                continue
            if not (track[-3][1] > track[-2][1] > track[-1][1]):
                continue
            v_close = (d_first - d_last) / dt_s
            if v_close < 0.45:
                continue
            ttc_s = d_last / v_close
            if ttc_s <= 0.85:
                triggered_sectors.append(sector)
                if best_d is None or d_last < best_d:
                    best_d = d_last
                    best_ttc = ttc_s
        if not triggered_sectors or best_d is None or best_ttc is None:
            return None
        return CaneDecision(
            timestamp_ns=now_ns,
            level=HazardLevel.WARNING,
            sectors=tuple(sorted(triggered_sectors, key=lambda item: item.value)),
            nearest_distance_m=best_d,
            semantic_labels=forward.semantic_labels,
            range_triggered=True,
            fresh_range_sensor_count=forward.fresh_range_sensor_count,
            reason=f"fast-approaching range {best_d:.3f} m (TTC {best_ttc:.2f} s)",
            source=SOURCE_RANGE,
        )

    def evaluate(self, now_ns: int) -> CaneDecision:
        forward = self._engine.evaluate(now_ns)
        ttc_warn = self._ttc_alert(now_ns, forward)
        forward_alert = forward.level in (HazardLevel.WARNING, HazardLevel.CRITICAL)
        overhead = self._overhead_alert(now_ns)

        near: List[CaneDecision] = []
        if forward_alert:
            near.append(_tag(forward, SOURCE_RANGE))
        elif ttc_warn is not None:
            near.append(ttc_warn)
        if overhead is not None:
            near.append(overhead)
        if near:
            return self._merge_near(forward, near)

        ranges_ok = forward.level != HazardLevel.SENSOR_FAULT
        far = self._vision_alert(now_ns, ranges_ok, forward, self.blind_sectors(now_ns))
        if far is not None:
            return far
        return _tag(forward, SOURCE_FAULT if not ranges_ok else SOURCE_RANGE)

    def _overhead_alert(self, now_ns: int) -> Optional[CaneDecision]:
        stale_ns = int(self.fusion.sensor_stale_after_ms * 1_000_000)
        fresh = [
            reading
            for reading in self._overhead.values()
            if reading.usable_range() and 0 <= now_ns - reading.timestamp_ns <= stale_ns
        ]
        if not fresh:
            return None
        nearest = min(float(reading.distance_m) for reading in fresh if reading.distance_m)
        if nearest <= self.priority.overhead_critical_m:
            level = HazardLevel.CRITICAL
        elif nearest <= self.priority.overhead_warning_m:
            level = HazardLevel.WARNING
        else:
            return None
        sectors = tuple(sorted({reading.sector for reading in fresh}, key=lambda item: item.value))
        return CaneDecision(
            timestamp_ns=now_ns,
            level=level,
            sectors=sectors,
            nearest_distance_m=nearest,
            semantic_labels=tuple(),
            range_triggered=True,
            fresh_range_sensor_count=len(fresh),
            reason=f"overhead range {nearest:.3f} m",
            source=SOURCE_OVERHEAD,
        )

    @staticmethod
    def _merge_near(forward: HazardDecision, near: Sequence[CaneDecision]) -> CaneDecision:
        top = max(decision.level.value for decision in near)
        chosen = [decision for decision in near if decision.level.value == top]
        if len(chosen) == 1:
            return chosen[0]
        sectors = tuple(sorted({s for d in chosen for s in d.sectors}, key=lambda item: item.value))
        nearest = min(d.nearest_distance_m for d in chosen if d.nearest_distance_m is not None)
        first = chosen[0]
        return CaneDecision(
            timestamp_ns=first.timestamp_ns,
            level=HazardLevel(top),
            sectors=sectors,
            nearest_distance_m=nearest,
            semantic_labels=first.semantic_labels,
            range_triggered=True,
            fresh_range_sensor_count=forward.fresh_range_sensor_count,
            reason="; ".join(d.reason for d in chosen),
            source=SOURCE_RANGE,
        )

    def _vision_alert(
        self,
        now_ns: int,
        ranges_ok: bool,
        forward: HazardDecision,
        blind: Sequence[Sector] = (),
    ) -> Optional[CaneDecision]:
        vision = self._vision
        if vision is None:
            return None
        stale_ns = int(self.fusion.vision_stale_after_ms * 1_000_000)
        if not 0 <= now_ns - vision.model_finished_ns <= stale_ns:
            return None
        blind_set = set(blind)
        candidates: List[Tuple[float, Detection]] = []
        for detection, distance in self._estimates:
            sector = detection.sector
            streak = self._streak[sector]
            if streak < self.priority.confirm_frames:
                continue
            if sector in blind_set:
                if distance <= self.priority.blind_sector_max_distance_m:
                    candidates.append((distance, detection))
                continue
            # With working range sensors the camera only speaks beyond their band.
            lower = self.fusion.warning_distance_m if ranges_ok else 0.0
            if distance > lower:
                candidates.append((distance, detection))
        if not candidates:
            return None
        distance, best = min(candidates, key=lambda item: item[0])
        level = HazardLevel.WARNING
        if (
            best.sector in blind_set
            and distance <= self.fusion.critical_distance_m
            and best.confidence >= self.priority.blind_sector_critical_confidence
            and self._streak[best.sector] >= self.priority.blind_sector_critical_frames
        ):
            level = HazardLevel.CRITICAL
        reason = (
            f"vision covers blind {best.sector.value} range: {best.class_name} ~{distance:.1f} m"
            if best.sector in blind_set
            else f"vision: {best.class_name} ~{distance:.1f} m {best.sector.value}"
        )
        return CaneDecision(
            timestamp_ns=now_ns,
            level=level,
            sectors=(best.sector,),
            nearest_distance_m=distance,
            semantic_labels=(best.class_name,),
            range_triggered=False,
            fresh_range_sensor_count=forward.fresh_range_sensor_count,
            reason=reason,
            source=SOURCE_VISION,
        )


@dataclass(frozen=True)
class CaneCommand(HapticCommand):
    """A haptic command that also says which sense raised it and how close the hazard is."""

    source: str = SOURCE_RANGE
    urgency: float = 0.0  # 0 at the warning distance, 1 when very near (drives the tempo)

    def as_dict(self) -> Dict[str, object]:
        payload = super().as_dict()
        payload.update({"source": self.source, "urgency": self.urgency})
        return payload


def command_for(
    decision: HazardDecision,
    priority: PriorityConfig,
    fusion: FusionConfig,
    haptics: Optional[HapticsConfig] = None,
) -> CaneCommand:
    """Turn a decision into a command for the single haptic channel.

    Closer means a faster rhythm and a stronger buzz. A far camera alert is
    deliberately gentle (soft, slow tempo) so the hand can tell "something ahead,
    still far" from "close".
    """

    haptics = haptics or HapticsConfig()
    source = getattr(decision, "source", SOURCE_RANGE)
    base = decision_to_haptic(decision)

    def make(intensity: float, urgency: float, label: str) -> CaneCommand:
        return CaneCommand(
            decision.timestamp_ns, decision.level, base.sectors, intensity,
            base.pulse_period_ms, decision.reason, label, urgency,
        )

    if decision.level == HazardLevel.CLEAR:
        return make(0.0, 0.0, source)
    if decision.level == HazardLevel.SENSOR_FAULT:
        return make(base.intensity, 0.0, SOURCE_FAULT)
    distance = decision.nearest_distance_m
    if source == SOURCE_VISION and distance is not None and distance > fusion.warning_distance_m:
        return make(priority.far_intensity, 0.0, SOURCE_VISION)
    warning = priority.overhead_warning_m if source == SOURCE_OVERHEAD else fusion.warning_distance_m
    urgency = urgency_for(distance if distance is not None else warning, warning, haptics.urgent_distance_m)
    return make(0.65 + 0.35 * urgency, urgency, source)
