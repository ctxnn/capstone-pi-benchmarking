"""Fault tolerance: one dead sensor must never take the cane's other senses away.

Each test injects a specific hardware or software failure and asserts that the
remaining sensors keep detecting, the hazard thresholds keep their accuracy, and
a blind sector is covered by vision instead of silently going quiet.
"""

import math
import time

import pytest

from yolo_pi.cane.config import PriorityConfig
from yolo_pi.cane.priority import PriorityFusion
from yolo_pi.fusion import FusionConfig, HazardLevel, SensorKind, SensorReading
from yolo_pi.fusion_v2 import (
    DEFAULT_CLASS_PRIORS,
    ExtendedHazardLevel,
    RangeReading,
    SectorHealth,
    SmartCaneFusionConfig,
    SmartCaneFusionEngine,
)
from yolo_pi.runtime import build_simulated_runtime
from yolo_pi.types import BoundingBox, Detection, FrameDetections, Sector

MS = 1_000_000
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SECTOR_OF = {
    "tof-left": Sector.LEFT,
    "tof-center": Sector.CENTER,
    "tof-right": Sector.RIGHT,
    "us-center-1": Sector.CENTER,
    "us-center-2": Sector.CENTER,
}


def tof(sensor_id, distance, ts, valid=True, no_target=False):
    return RangeReading(
        sensor_id=sensor_id,
        kind=SensorKind.TOF,
        timestamp_ns=ts,
        sector=SECTOR_OF[sensor_id],
        distance_m=distance,
        valid=valid,
        no_target=no_target,
    )


def us(sensor_id, distance, ts, valid=True, no_target=False):
    return RangeReading(
        sensor_id=sensor_id,
        kind=SensorKind.ULTRASONIC,
        timestamp_ns=ts,
        sector=SECTOR_OF[sensor_id],
        distance_m=distance,
        valid=valid,
        no_target=no_target,
    )


def _box_for(sector, distance_m, class_name, focal_px, config):
    """A box whose pinhole geometry implies ``distance_m`` (the engine's own model)."""

    canonical = DEFAULT_CLASS_PRIORS[class_name].canonical_height_m
    depression = math.degrees(math.atan2(config.camera_height_m, distance_m))
    y2 = FRAME_HEIGHT / 2.0 + focal_px * math.tan(
        math.radians(depression - config.camera_pitch_down_deg)
    )
    elevation = math.degrees(
        math.atan2(canonical - config.camera_height_m, distance_m)
    )
    y1 = FRAME_HEIGHT / 2.0 - focal_px * math.tan(
        math.radians(elevation + config.camera_pitch_down_deg)
    )
    center = {Sector.LEFT: 100.0, Sector.CENTER: 320.0, Sector.RIGHT: 540.0}[sector]
    return BoundingBox(center - 30, y1, center + 30, y2)


def _cane_box_for(sector, distance_m, class_name, priority):
    """The cane's simpler estimator: distance = focal * class height / box height."""

    focal = (FRAME_HEIGHT / 2.0) / math.tan(math.radians(priority.vertical_fov_deg) / 2.0)
    height = focal * priority.height_for(class_name) / distance_m
    center = {Sector.LEFT: 100.0, Sector.CENTER: 320.0, Sector.RIGHT: 540.0}[sector]
    return BoundingBox(center - 30, 240 - height / 2, center + 30, 240 + height / 2)


def vision_frame(
    sector,
    distance_m,
    ts,
    frame_id=1,
    name="person",
    confidence=0.9,
    config=None,
    box_builder=None,
):
    config = config or SmartCaneFusionConfig()
    if box_builder is not None:
        box = box_builder(sector, distance_m)
    else:
        focal = (FRAME_HEIGHT / 2.0) / math.tan(math.radians(config.camera_vfov_deg) / 2.0)
        box = _box_for(sector, distance_m, name, focal, config)
    return FrameDetections(
        frame_id=frame_id,
        captured_ns=ts - 20 * MS,
        model_started_ns=ts - 10 * MS,
        model_finished_ns=ts,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        detections=[Detection(0, name, confidence, box, sector)],
    )


def snapshot(ts, left=2.0, center=2.0, right=2.0, us1=None, us2=None, left_valid=True):
    return [
        tof("tof-left", left if left_valid else None, ts, valid=left_valid),
        tof("tof-center", center, ts),
        tof("tof-right", right, ts),
        us("us-center-1", center if us1 is None else us1, ts),
        us("us-center-2", center if us2 is None else us2, ts),
    ]


def healthy_engine(config=None, ts=0):
    engine = SmartCaneFusionEngine(config)
    engine.update_sensors(snapshot(ts))
    engine.evaluate(ts)
    return engine


# ---------------------------------------------------------------------------
# fusion_v2: one sensor dies, the rest keep detecting
# ---------------------------------------------------------------------------


def test_a_dead_tof_left_leaves_the_other_sectors_detecting() -> None:
    engine = healthy_engine()
    ts = 100 * MS
    engine.update_sensors(
        [
            tof("tof-left", None, ts, valid=False),
            tof("tof-center", 0.80, ts),
            tof("tof-right", 1.60, ts),
            us("us-center-1", 0.80, ts),
            us("us-center-2", 0.80, ts),
        ]
    )
    decision = engine.evaluate(ts + 10 * MS)
    left, center, right = (
        decision.sectors[Sector.LEFT],
        decision.sectors[Sector.CENTER],
        decision.sectors[Sector.RIGHT],
    )
    assert left.health is SectorHealth.FAULT
    assert left.level is ExtendedHazardLevel.SENSOR_FAULT
    assert center.level is ExtendedHazardLevel.WARNING
    assert center.fused_distance_m == pytest.approx(0.80, abs=0.06)
    assert right.level is ExtendedHazardLevel.CLEAR
    assert decision.overall_level is not ExtendedHazardLevel.SENSOR_FAULT
    assert "tof-left" in decision.degraded_sensors


def test_one_dead_ultrasonic_keeps_center_accuracy() -> None:
    engine = healthy_engine()
    ts = 100 * MS
    engine.update_sensors(
        [
            tof("tof-center", 0.90, ts),
            us("us-center-1", 0.90, ts),
            us("us-center-2", None, ts, valid=False),
        ]
    )
    decision = engine.evaluate(ts + 10 * MS)
    center = decision.sectors[Sector.CENTER]
    assert center.health is SectorHealth.DEGRADED
    assert center.level is ExtendedHazardLevel.WARNING
    assert center.fused_distance_m == pytest.approx(0.90, abs=0.08)
    assert "us-center-2" in center.missing_sensor_ids


def test_all_tof_dead_still_warns_through_the_ultrasonics() -> None:
    engine = healthy_engine()
    ts = 100 * MS
    engine.update_sensors(
        [
            tof("tof-left", None, ts, valid=False),
            tof("tof-center", None, ts, valid=False),
            tof("tof-right", None, ts, valid=False),
            us("us-center-1", 0.70, ts),
            us("us-center-2", 0.70, ts),
        ]
    )
    decision = engine.evaluate(ts + 10 * MS)
    center = decision.sectors[Sector.CENTER]
    assert center.health is SectorHealth.DEGRADED
    assert center.level is ExtendedHazardLevel.WARNING
    assert center.fused_distance_m == pytest.approx(0.70, abs=0.08)
    assert decision.fresh_range_sensor_count == 2


def test_a_dead_mux_channel_does_not_stop_center_detection() -> None:
    engine = healthy_engine()
    for step in range(6):
        ts = (100 + step * 33) * MS
        engine.update_sensors(
            [
                tof("tof-left", None, ts, valid=False),
                tof("tof-right", None, ts, valid=False),
                tof("tof-center", 1.30 - step * 0.10, ts),
                us("us-center-1", 1.30 - step * 0.10, ts),
                us("us-center-2", 1.30 - step * 0.10, ts),
            ]
        )
        decision = engine.evaluate(ts + 5 * MS)
    center = decision.sectors[Sector.CENTER]
    assert center.fused_distance_m == pytest.approx(0.80, abs=0.10)
    assert center.level.value >= ExtendedHazardLevel.WARNING.value
    assert decision.sectors[Sector.LEFT].health is SectorHealth.FAULT
    assert decision.sectors[Sector.RIGHT].health is SectorHealth.FAULT
    assert center.health is SectorHealth.OK


# ---------------------------------------------------------------------------
# fusion_v2: a blind sector is covered by vision instead of going silent
# ---------------------------------------------------------------------------


def test_vision_covers_a_blind_sector_at_medium_range() -> None:
    engine = healthy_engine()
    ts = 100 * MS
    for index in range(2):
        now = ts + index * 33 * MS
        engine.update_vision(vision_frame(Sector.LEFT, 1.4, now, frame_id=index + 1))
        engine.update_sensors(snapshot(now + 5 * MS, left_valid=False))
        decision = engine.evaluate(now + 10 * MS)
    left = decision.sectors[Sector.LEFT]
    assert left.health is SectorHealth.FAULT
    assert left.level is ExtendedHazardLevel.WARNING
    assert left.semantic_labels == ("person",)
    assert left.vision_nearest_m == pytest.approx(1.4, abs=0.3)


def test_blind_sector_needs_persistent_frames_before_critical() -> None:
    engine = healthy_engine()
    ts = 100 * MS
    engine.update_sensors(snapshot(ts, left_valid=False))
    engine.update_vision(vision_frame(Sector.LEFT, 0.45, ts, confidence=0.85))
    first = engine.evaluate(ts + 5 * MS).sectors[Sector.LEFT]
    assert first.level is not ExtendedHazardLevel.CRITICAL

    decided = first
    for index in range(3):
        now = ts + (40 + index * 33) * MS
        engine.update_vision(
            vision_frame(Sector.LEFT, 0.45, now, confidence=0.85, frame_id=index + 2)
        )
        engine.update_sensors(snapshot(now + 5 * MS, left_valid=False))
        decided = engine.evaluate(now + 10 * MS).sectors[Sector.LEFT]
    assert decided.level is ExtendedHazardLevel.CRITICAL


def test_blind_sector_vision_coverage_can_be_disabled() -> None:
    config = SmartCaneFusionConfig(vision_covers_blind_sectors=False)
    engine = healthy_engine(config)
    ts = 100 * MS
    for index in range(3):
        now = ts + index * 33 * MS
        engine.update_vision(
            vision_frame(Sector.LEFT, 1.0, now, frame_id=index + 1, config=config)
        )
        engine.update_sensors(snapshot(now + 5 * MS, left_valid=False))
        left = engine.evaluate(now + 10 * MS).sectors[Sector.LEFT]
    assert left.level is ExtendedHazardLevel.SENSOR_FAULT


# ---------------------------------------------------------------------------
# Runtime: recoverable failures never stop the control loop
# ---------------------------------------------------------------------------


class _FailingUltrasonic:
    def __init__(self, fail_for: int = 10**9):
        self.calls = 0
        self.fail_for = fail_for

    def read_next(self, now_ns=None):
        self.calls += 1
        if self.calls <= self.fail_for:
            raise RuntimeError("simulated HC-SR04 stuck echo")
        from yolo_pi.hardware.hcsr04 import UltrasonicMeasurement

        return UltrasonicMeasurement(
            sensor_id="us-center-1",
            sector=Sector.CENTER,
            timestamp_ns=time.monotonic_ns(),
            distance_m=0.60,
            raw_echo_s=0.0035,
            valid=True,
            no_target=False,
            ambient_temp_c=22.0,
        )

    def close(self) -> None:
        pass


class _FailingIMU:
    def __init__(self):
        self.calls = 0

    def read_raw(self, now_ns=None):
        self.calls += 1
        raise OSError("simulated MPU-6050 dropout")

    def close(self) -> None:
        pass


class _FailingDetector:
    def detect(self, frame, frame_id, captured_ns=None, source=None):
        raise RuntimeError("simulated OpenVINO failure")


class _FailingHapticSink:
    def __init__(self):
        self.frames = 0

    def emit_frame(self, frame) -> None:
        self.frames += 1
        raise OSError("simulated DRV2605L bus error")

    def stop(self) -> None:
        pass


def test_ultrasonic_failure_keeps_the_control_loop_and_tof_detection() -> None:
    runtime = build_simulated_runtime(enable_camera=False)
    runtime.ultrasonic_driver = _FailingUltrasonic()
    summary = runtime.run_for(0.5)
    assert runtime.thread_errors == ()
    assert summary.ticks > 15
    assert summary.source_errors.get("ultrasonic", 0) > 0
    assert summary.sector_hazard_counts["center"] > 0


def test_imu_dropout_keeps_detection_running() -> None:
    runtime = build_simulated_runtime(enable_camera=False)
    runtime.imu_driver = _FailingIMU()
    summary = runtime.run_for(0.5)
    assert runtime.thread_errors == ()
    assert summary.ticks > 15
    assert runtime.imu_errors, "the IMU failure should still be reported"
    assert summary.sector_hazard_counts["center"] > 0


def test_detector_failure_keeps_range_haptics_running() -> None:
    runtime = build_simulated_runtime(enable_camera=True)
    runtime.detector = _FailingDetector()
    summary = runtime.run_for(0.5)
    assert runtime.thread_errors == ()
    assert summary.ticks > 15
    assert runtime.vision_errors
    assert summary.sector_hazard_counts["center"] > 0


def test_haptic_write_failure_keeps_sensing_running() -> None:
    runtime = build_simulated_runtime(enable_camera=False)
    runtime.haptic_sink = _FailingHapticSink()
    summary = runtime.run_for(0.5)
    assert runtime.thread_errors == ()
    assert summary.ticks > 15
    assert summary.source_errors.get("haptic", 0) > 0
    assert runtime.latest_decision is not None


def test_fusion_exception_builds_a_safe_frame_when_nothing_recent_exists() -> None:
    runtime = build_simulated_runtime(enable_camera=False)

    def broken(now_ns):
        raise RuntimeError("simulated fusion bug")

    runtime.fusion.evaluate = broken  # type: ignore[method-assign]
    summary = runtime.run_for(0.3)
    assert runtime.thread_errors == ()
    assert summary.ticks > 8
    assert runtime.metrics.source_errors.get("fusion", 0) > 0
    decision = runtime.latest_decision
    assert decision is not None
    assert decision.overall_level is ExtendedHazardLevel.SENSOR_FAULT


def test_fusion_exception_falls_back_to_the_last_safe_decision() -> None:
    runtime = build_simulated_runtime(enable_camera=False)
    runtime.start()
    try:
        time.sleep(0.15)

        def broken(now_ns):
            raise RuntimeError("simulated fusion bug")

        runtime.fusion.evaluate = broken  # type: ignore[method-assign]
        time.sleep(0.2)
        assert runtime.thread_errors == ()
        assert runtime.metrics.source_errors.get("fusion", 0) > 0
        assert runtime.latest_decision is not None
    finally:
        runtime.stop()


class _WedgedToFArray:
    """A wedged mux whose faulted chip comes back after a per-sensor re-init."""

    def __init__(self):
        self.faulted_sensor_ids = ["tof-left"]
        self.repaired: list = []

    def read_all(self, now_ns=None):
        raise OSError("simulated wedged mux")

    def initialize_sensor(self, sensor_id):
        if sensor_id != "tof-left":
            raise AssertionError(f"healthy chip {sensor_id} must not be re-initialized")
        self.repaired.append(sensor_id)
        self.faulted_sensor_ids = []


class _WedgedToFArrayWithoutPerSensorRepair(_WedgedToFArray):
    initialize_sensor = None

    def initialize(self):
        self.repaired.append("all")
        self.faulted_sensor_ids = []
        return []


def test_a_wedged_tof_bus_is_re_initialized_in_place() -> None:
    runtime = build_simulated_runtime(enable_camera=False)
    array = _WedgedToFArray()
    runtime.tof_array = array
    summary = runtime.run_for(0.3)
    assert runtime.thread_errors == ()
    assert summary.ticks > 8
    assert summary.source_errors.get("tof", 0) > 0
    assert array.repaired == ["tof-left"]
    assert summary.reinit_attempts == 1
    assert summary.reinit_recovered == 1


def test_tof_repair_falls_back_to_the_array_wide_initialize() -> None:
    runtime = build_simulated_runtime(enable_camera=False)
    array = _WedgedToFArrayWithoutPerSensorRepair()
    runtime.tof_array = array
    summary = runtime.run_for(0.3)
    assert runtime.thread_errors == ()
    assert array.repaired == ["all"]
    assert summary.reinit_attempts == 1
    assert summary.reinit_recovered == 1


# ---------------------------------------------------------------------------
# cane: blind-sector vision coverage in the operations engine
# ---------------------------------------------------------------------------


def _cane_fusion(priority=None) -> PriorityFusion:
    expected = {
        Sector.LEFT: ("tof-left",),
        Sector.CENTER: ("tof-center", "us-1", "us-2"),
        Sector.RIGHT: ("tof-right",),
    }
    return PriorityFusion(
        FusionConfig(),
        priority or PriorityConfig(),
        ground_ids=("us-2",),
        sensor_pitch_offsets_deg={"us-1": 0.0, "us-2": 8.0},
        expected_by_sector=expected,
    )


def _cane_ranges(fusion, ts, *, left_dead=True):
    readings = [
        SensorReading("tof-center", SensorKind.TOF, ts, Sector.CENTER, 2.0, True),
        SensorReading("tof-right", SensorKind.TOF, ts, Sector.RIGHT, 2.0, True),
        SensorReading("us-1", SensorKind.ULTRASONIC, ts, Sector.CENTER, 4.0, True),
        SensorReading("us-2", SensorKind.ULTRASONIC, ts, Sector.CENTER, 4.0, True),
    ]
    if left_dead:
        readings.append(SensorReading("tof-left", SensorKind.TOF, ts, Sector.LEFT, None, False))
    else:
        readings.append(SensorReading("tof-left", SensorKind.TOF, ts, Sector.LEFT, 2.0, True))
    fusion.update_sensors(readings)


def test_cane_vision_covers_a_dead_left_tof() -> None:
    priority = PriorityConfig()
    fusion = _cane_fusion(priority)
    ts = 100 * MS
    for step in range(4):
        now = ts + step * 33 * MS
        _cane_ranges(fusion, now)
        fusion.update_vision(
            vision_frame(
                Sector.LEFT,
                1.3,
                now,
                frame_id=step,
                name="stairs",
                box_builder=lambda sector, distance: _cane_box_for(
                    sector, distance, "stairs", priority
                ),
            )
        )
    decision = fusion.evaluate(ts + 200 * MS)
    assert fusion.blind_sectors(ts + 200 * MS) == (Sector.LEFT,)
    assert decision.source == "vision"
    assert decision.sectors == (Sector.LEFT,)
    assert decision.level is HazardLevel.WARNING


def test_cane_blind_sector_critical_needs_confidence_and_persistence() -> None:
    priority = PriorityConfig()

    def run(confidence, frames):
        fusion = _cane_fusion(priority)
        ts = 100 * MS
        for step in range(frames):
            now = ts + step * 33 * MS
            _cane_ranges(fusion, now)
            fusion.update_vision(
                vision_frame(
                    Sector.LEFT,
                    0.45,
                    now,
                    frame_id=step,
                    name="curb",
                    confidence=confidence,
                    box_builder=lambda sector, distance: _cane_box_for(
                        sector, distance, "curb", priority
                    ),
                )
            )
        return fusion.evaluate(ts + 500 * MS)

    assert run(0.85, 4).level is HazardLevel.CRITICAL
    assert run(0.50, 4).level is HazardLevel.WARNING


def test_cane_a_live_range_critical_outranks_blind_sector_vision() -> None:
    priority = PriorityConfig()
    fusion = _cane_fusion(priority)
    ts = 100 * MS
    for step in range(4):
        now = ts + step * 33 * MS
        fusion.update_sensors(
            [
                SensorReading("tof-center", SensorKind.TOF, now, Sector.CENTER, 0.40, True),
                SensorReading("tof-right", SensorKind.TOF, now, Sector.RIGHT, 2.0, True),
                SensorReading("us-1", SensorKind.ULTRASONIC, now, Sector.CENTER, 4.0, True),
                SensorReading("us-2", SensorKind.ULTRASONIC, now, Sector.CENTER, 4.0, True),
                SensorReading("tof-left", SensorKind.TOF, now, Sector.LEFT, None, False),
            ]
        )
        fusion.update_vision(
            vision_frame(
                Sector.LEFT,
                1.3,
                now,
                frame_id=step,
                name="stairs",
                box_builder=lambda sector, distance: _cane_box_for(
                    sector, distance, "stairs", priority
                ),
            )
        )
    decision = fusion.evaluate(ts + 200 * MS)
    assert decision.source == "range"
    assert decision.level is HazardLevel.CRITICAL
    assert decision.sectors == (Sector.CENTER,)


def test_cane_long_range_vision_is_not_blind_sector_coverage() -> None:
    priority = PriorityConfig()
    fusion = _cane_fusion(priority)
    ts = 100 * MS
    for step in range(4):
        now = ts + step * 33 * MS
        _cane_ranges(fusion, now, left_dead=False)
        fusion.update_vision(
            vision_frame(
                Sector.LEFT,
                1.3,
                now,
                frame_id=step,
                name="stairs",
                box_builder=lambda sector, distance: _cane_box_for(
                    sector, distance, "stairs", priority
                ),
            )
        )
    decision = fusion.evaluate(ts + 200 * MS)
    assert fusion.blind_sectors(ts + 200 * MS) == ()
    assert decision.source == "vision"
    assert decision.level is HazardLevel.WARNING
    assert "covers blind" not in decision.reason
