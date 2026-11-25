import math
import threading
import unittest
from dataclasses import replace

from yolo_pi.fusion import HazardLevel, SensorKind, SensorReading
from yolo_pi.fusion_v2 import (
    DEFAULT_CLASS_PRIORS,
    ClassVisionPrior,
    DirectionalHapticFrame,
    DirectionalHazardDecision,
    ExtendedHazardLevel,
    RangeReading,
    RecordingDirectionalHapticSink,
    SectorHealth,
    SmartCaneFusionConfig,
    SmartCaneFusionEngine,
    decision_to_directional_haptics,
    distance_weight_schedule,
    estimate_monocular_distance_m,
)
from yolo_pi.imu import GRAVITY_MPS2, IMUSample, IMUState
from yolo_pi.types import BoundingBox, Detection, FrameDetections, Sector

MS = 1_000_000
SECTORS = (Sector.LEFT, Sector.CENTER, Sector.RIGHT)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_HEIGHT_M = 0.95
CAMERA_PITCH_DEG = 12.0
CAMERA_VFOV_DEG = 48.8
FOCAL_PX = (FRAME_HEIGHT / 2.0) / math.tan(math.radians(CAMERA_VFOV_DEG / 2.0))


def tof(
    sensor_id: str,
    distance_m,
    timestamp_ns: int,
    sector: Sector = Sector.CENTER,
    valid: bool = True,
    no_target: bool = False,
) -> RangeReading:
    return RangeReading(
        sensor_id=sensor_id,
        kind=SensorKind.TOF,
        timestamp_ns=timestamp_ns,
        sector=sector,
        distance_m=distance_m,
        valid=valid,
        no_target=no_target,
    )


def us(
    sensor_id: str,
    distance_m,
    timestamp_ns: int,
    sector: Sector = Sector.CENTER,
    valid: bool = True,
    no_target: bool = False,
) -> RangeReading:
    return RangeReading(
        sensor_id=sensor_id,
        kind=SensorKind.ULTRASONIC,
        timestamp_ns=timestamp_ns,
        sector=sector,
        distance_m=distance_m,
        valid=valid,
        no_target=no_target,
    )


def imu_sample(
    timestamp_ns: int,
    yaw_rate_rps: float = 0.0,
    pitch_deg: float = 12.0,
    pitch_rate_rps: float = 0.0,
) -> IMUSample:
    pitch = math.radians(pitch_deg)
    return IMUSample(
        timestamp_ns=timestamp_ns,
        ax_mps2=-GRAVITY_MPS2 * math.sin(pitch),
        ay_mps2=0.0,
        az_mps2=GRAVITY_MPS2 * math.cos(pitch),
        gx_rps=0.0,
        gy_rps=pitch_rate_rps,
        gz_rps=yaw_rate_rps,
    )


def tilt_engine_down(
    engine: SmartCaneFusionEngine,
    start_ns: int,
    end_pitch_deg: float = 40.0,
    start_pitch_deg: float = 12.0,
    rate_deg_s: float = 140.0,
    step_ms: float = 5.0,
) -> int:
    """Tip the engine's IMU from the walking baseline onto the floor."""

    engine.imu.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=start_pitch_deg)
    engine.update_imu(imu_sample(start_ns, pitch_deg=start_pitch_deg))
    span = end_pitch_deg - start_pitch_deg
    steps = max(int(round(span / (rate_deg_s * step_ms / 1000.0))), 1)
    timestamp_ns = start_ns
    for step in range(1, steps + 1):
        timestamp_ns = start_ns + int(step * step_ms * MS)
        pitch = start_pitch_deg + min(span, rate_deg_s * step * step_ms / 1000.0)
        engine.update_imu(
            imu_sample(
                timestamp_ns,
                pitch_deg=pitch,
                pitch_rate_rps=math.radians(rate_deg_s),
            )
        )
    for _ in range(20):
        timestamp_ns += int(step_ms * MS)
        engine.update_imu(imu_sample(timestamp_ns, pitch_deg=end_pitch_deg))
    return timestamp_ns


def synthetic_detection(
    class_name: str,
    distance_m: float,
    confidence: float = 0.9,
    center_x: float = 320.0,
    class_id: int = 0,
    half_width_px: float = 40.0,
) -> Detection:
    """Build a detection whose geometry implies ``distance_m`` in front."""

    prior = DEFAULT_CLASS_PRIORS[class_name]
    depression = math.degrees(math.atan2(CAMERA_HEIGHT_M, distance_m))
    y2 = FRAME_HEIGHT / 2.0 + FOCAL_PX * math.tan(
        math.radians(depression - CAMERA_PITCH_DEG)
    )
    elevation = math.degrees(
        math.atan2(prior.canonical_height_m - CAMERA_HEIGHT_M, distance_m)
    )
    y1 = FRAME_HEIGHT / 2.0 - FOCAL_PX * math.tan(
        math.radians(elevation + CAMERA_PITCH_DEG)
    )
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        box=BoundingBox(center_x - half_width_px, y1, center_x + half_width_px, y2),
        sector=Sector.UNKNOWN,
    )


def frame(
    frame_id: int,
    captured_ns: int,
    detections,
) -> FrameDetections:
    return FrameDetections(
        frame_id=frame_id,
        captured_ns=captured_ns,
        model_started_ns=captured_ns + 5 * MS,
        model_finished_ns=captured_ns + 10 * MS,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        detections=list(detections),
    )


def all_clear(timestamp_ns: int):
    return [
        tof("tof-left", None, timestamp_ns, Sector.LEFT, no_target=True),
        tof("tof-center", None, timestamp_ns, no_target=True),
        tof("tof-right", None, timestamp_ns, Sector.RIGHT, no_target=True),
        us("us-center-1", None, timestamp_ns, no_target=True),
        us("us-center-2", None, timestamp_ns, no_target=True),
    ]


def snapshot(distance_m: float, timestamp_ns: int, left=None, right=None):
    readings = [
        tof("tof-center", distance_m, timestamp_ns),
        us("us-center-1", distance_m, timestamp_ns),
        us("us-center-2", distance_m, timestamp_ns),
    ]
    readings.append(
        tof("tof-left", None, timestamp_ns, Sector.LEFT, no_target=True)
        if left is None
        else tof("tof-left", left, timestamp_ns, Sector.LEFT)
    )
    readings.append(
        tof("tof-right", None, timestamp_ns, Sector.RIGHT, no_target=True)
        if right is None
        else tof("tof-right", right, timestamp_ns, Sector.RIGHT)
    )
    return readings


def stationary_state(timestamp_ns: int) -> IMUState:
    return IMUState(
        timestamp_ns=timestamp_ns,
        fresh=True,
        calibrated=True,
        pitch_deg=12.0,
        roll_deg=0.0,
        relative_pitch_deg=0.0,
        yaw_rate_rps=0.0,
        sweep_yaw_deg=0.0,
        is_stationary=True,
    )


class DistanceWeightScheduleTests(unittest.TestCase):
    def test_weights_always_sum_to_one(self) -> None:
        for distance in (None, 0.0, 0.2, 0.35, 0.75, 1.0, 1.4, 2.0, 2.4, 3.2, 6.0):
            w_range, w_vision = distance_weight_schedule(distance)
            self.assertAlmostEqual(w_range + w_vision, 1.0, places=9)
            self.assertGreaterEqual(w_range, 0.0)
            self.assertLessEqual(w_range, 1.0)

    def test_short_range_is_range_dominant(self) -> None:
        w_range, w_vision = distance_weight_schedule(0.4)
        self.assertGreaterEqual(w_range, 0.90)
        self.assertLessEqual(w_range, 0.98)
        self.assertLessEqual(w_vision, 0.10)
        closest_range, _ = distance_weight_schedule(0.35)
        self.assertAlmostEqual(closest_range, 0.98, places=9)
        mid_range, mid_vision = distance_weight_schedule(1.0)
        self.assertAlmostEqual(mid_range, 0.90, places=6)
        self.assertAlmostEqual(mid_vision, 0.10, places=6)

    def test_long_range_is_vision_dominant(self) -> None:
        self.assertEqual(distance_weight_schedule(None), (0.0, 1.0))
        w_range, w_vision = distance_weight_schedule(2.0)
        self.assertAlmostEqual(w_range, 0.20, places=6)
        self.assertAlmostEqual(w_vision, 0.80, places=6)
        far_range, far_vision = distance_weight_schedule(2.6)
        self.assertLessEqual(far_range, 1e-9)
        self.assertAlmostEqual(far_vision, 1.0, places=9)

    def test_weights_decrease_monotonically_with_distance(self) -> None:
        previous = 1.01
        distance = 0.3
        while distance <= 2.6:
            w_range, _ = distance_weight_schedule(distance)
            self.assertLessEqual(w_range, previous + 1e-9)
            previous = w_range
            distance += 0.05


class MonocularDistanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SmartCaneFusionConfig()

    def test_person_distance_is_recovered(self) -> None:
        detection = synthetic_detection("person", 2.2)
        estimate = estimate_monocular_distance_m(
            detection, FRAME_HEIGHT, self.config
        )
        self.assertAlmostEqual(estimate, 2.2, delta=0.2)

    def test_stairs_distance_is_recovered(self) -> None:
        detection = synthetic_detection("stairs", 1.6)
        estimate = estimate_monocular_distance_m(
            detection, FRAME_HEIGHT, self.config
        )
        self.assertAlmostEqual(estimate, 1.6, delta=0.15)

    def test_curb_distance_is_recovered(self) -> None:
        detection = synthetic_detection("curb", 1.2)
        estimate = estimate_monocular_distance_m(
            detection, FRAME_HEIGHT, self.config
        )
        self.assertAlmostEqual(estimate, 1.2, delta=0.2)

    def test_estimate_is_clamped_to_the_supported_band(self) -> None:
        tiny = synthetic_detection("person", 0.4, half_width_px=300.0)
        self.assertGreaterEqual(
            estimate_monocular_distance_m(tiny, FRAME_HEIGHT, self.config), 0.35
        )
        far = synthetic_detection("person", 9.0, half_width_px=5.0)
        self.assertLessEqual(
            estimate_monocular_distance_m(far, FRAME_HEIGHT, self.config), 6.0
        )

    def test_tilted_imu_reduces_the_ground_contact_cue(self) -> None:
        detection = synthetic_detection("person", 2.2)
        upright = estimate_monocular_distance_m(
            detection, FRAME_HEIGHT, self.config, stationary_state(0)
        )
        pitched_down = IMUState(
            timestamp_ns=0,
            fresh=True,
            calibrated=True,
            pitch_deg=30.0,
            roll_deg=0.0,
            relative_pitch_deg=18.0,
            yaw_rate_rps=0.0,
            sweep_yaw_deg=0.0,
        )
        tilted = estimate_monocular_distance_m(
            detection, FRAME_HEIGHT, self.config, pitched_down
        )
        self.assertLess(tilted, upright)

    def test_frame_height_must_be_positive(self) -> None:
        detection = synthetic_detection("person", 2.0)
        with self.assertRaises(ValueError):
            estimate_monocular_distance_m(detection, 0, self.config)


class RangeReadingTests(unittest.TestCase):
    def test_missing_distance_becomes_no_target(self) -> None:
        reading = SensorReading(
            sensor_id="tof-center",
            kind=SensorKind.TOF,
            timestamp_ns=5 * MS,
            sector=Sector.CENTER,
            distance_m=None,
            valid=True,
        )
        converted = RangeReading.from_sensor_reading(reading, max_range_m=2.0)
        self.assertTrue(converted.no_target)
        self.assertTrue(converted.valid)
        self.assertFalse(converted.has_measurement())
        self.assertEqual(converted.max_range_m, 2.0)

    def test_invalid_reading_is_not_no_target(self) -> None:
        reading = SensorReading(
            sensor_id="tof-center",
            kind=SensorKind.TOF,
            timestamp_ns=5 * MS,
            sector=Sector.CENTER,
            distance_m=None,
            valid=False,
        )
        converted = RangeReading.from_sensor_reading(reading)
        self.assertFalse(converted.no_target)
        self.assertFalse(converted.has_measurement())

    def test_measurement_requires_a_positive_finite_distance(self) -> None:
        self.assertTrue(tof("a", 0.5, 0).has_measurement())
        self.assertFalse(tof("a", 0.0, 0).has_measurement())
        self.assertFalse(tof("a", -1.0, 0).has_measurement())
        self.assertFalse(tof("a", float("nan"), 0).has_measurement())


class EngineFusionTests(unittest.TestCase):
    def test_inverse_variance_mean_matches_gauss_markov(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 1_000 * MS
        engine.update_sensors([tof("tof-center", 0.80, now), us("us-center-1", 0.90, now)])
        state = engine.evaluate(now).sectors[Sector.CENTER]
        sigma_tof = 0.010 + 0.030 * 0.80
        sigma_us = 0.008 + 0.025 * 0.90
        weight_tof = 1.0 / sigma_tof**2
        weight_us = 1.0 / sigma_us**2
        expected_distance = (weight_tof * 0.80 + weight_us * 0.90) / (
            weight_tof + weight_us
        )
        expected_sigma = math.sqrt(1.0 / (weight_tof + weight_us))
        self.assertAlmostEqual(state.fused_distance_m, expected_distance, places=9)
        self.assertAlmostEqual(state.distance_sigma_m, expected_sigma, places=9)
        expected_range_weight, expected_vision_weight = distance_weight_schedule(
            state.fused_distance_m
        )
        self.assertAlmostEqual(state.range_weight, expected_range_weight, places=9)
        self.assertAlmostEqual(state.vision_weight, expected_vision_weight, places=9)

    def test_sample_age_inflates_the_fused_variance(self) -> None:
        now = 1_500 * MS
        fresh_engine = SmartCaneFusionEngine()
        fresh_engine.update_sensors(
            [tof("tof-center", 0.80, now), us("us-center-1", 0.90, now)]
        )
        fresh = fresh_engine.evaluate(now).sectors[Sector.CENTER]

        aged_engine = SmartCaneFusionEngine()
        aged_engine.update_sensors(
            [tof("tof-center", 0.80, now), us("us-center-1", 0.90, now - 100 * MS)]
        )
        aged = aged_engine.evaluate(now).sectors[Sector.CENTER]

        sigma_tof = 0.010 + 0.030 * 0.80
        sigma_us_aged = 0.008 + 0.025 * 0.90 + 1.2 * 0.1
        weight_tof = 1.0 / sigma_tof**2
        weight_us = 1.0 / sigma_us_aged**2
        expected = (weight_tof * 0.80 + weight_us * 0.90) / (weight_tof + weight_us)
        expected_sigma = math.sqrt(1.0 / (weight_tof + weight_us))
        self.assertAlmostEqual(aged.fused_distance_m, expected, places=6)
        self.assertAlmostEqual(aged.distance_sigma_m, expected_sigma, places=6)
        self.assertLess(aged.fused_distance_m, fresh.fused_distance_m)
        self.assertGreater(aged.distance_sigma_m, fresh.distance_sigma_m)
        self.assertEqual(len(aged.fresh_sensor_ids), 2)

    def test_three_agreeing_sensors_use_one_weighted_mean(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 2_000 * MS
        engine.update_sensors(
            [
                tof("tof-center", 1.20, now),
                us("us-center-1", 1.25, now),
                us("us-center-2", 1.30, now),
            ]
        )
        state = engine.evaluate(now).sectors[Sector.CENTER]
        measurements = [
            (1.20, 0.010 + 0.030 * 1.20),
            (1.25, 0.008 + 0.025 * 1.25),
            (1.30, 0.008 + 0.025 * 1.30),
        ]
        weights = [1.0 / sigma**2 for _, sigma in measurements]
        expected = sum(w * d for w, (d, _) in zip(weights, measurements)) / sum(weights)
        self.assertAlmostEqual(state.fused_distance_m, expected, places=9)
        self.assertEqual(
            state.fresh_sensor_ids, ("tof-center", "us-center-1", "us-center-2")
        )
        self.assertEqual(state.health, SectorHealth.OK)

    def test_single_reading_passes_through_unfiltered(self) -> None:
        engine = SmartCaneFusionEngine()
        engine.update_sensor(tof("tof-center", 1.234, 10 * MS))
        state = engine.evaluate(10 * MS).sectors[Sector.CENTER]
        self.assertAlmostEqual(state.fused_distance_m, 1.234, places=9)

    def test_narrow_pole_requires_two_frames_before_critical(self) -> None:
        engine = SmartCaneFusionEngine()
        levels = []
        for index in range(2):
            timestamp_ns = 100 * MS + index * 100 * MS
            engine.update_sensors(snapshot(1.90, timestamp_ns))
            levels.append(engine.evaluate(timestamp_ns).sectors[Sector.CENTER].level)
        self.assertEqual(levels, [ExtendedHazardLevel.CLEAR, ExtendedHazardLevel.CLEAR])

        timestamp_ns = 300 * MS
        engine.update_sensors(
            [
                tof("tof-center", 0.45, timestamp_ns),
                us("us-center-1", 1.90, timestamp_ns),
                us("us-center-2", 1.90, timestamp_ns),
            ]
        )
        first = engine.evaluate(timestamp_ns).sectors[Sector.CENTER]
        self.assertNotEqual(first.level, ExtendedHazardLevel.CRITICAL)
        self.assertGreater(first.fused_distance_m, 1.0)

        timestamp_ns = 400 * MS
        engine.update_sensors(
            [
                tof("tof-center", 0.45, timestamp_ns),
                us("us-center-1", 1.90, timestamp_ns),
                us("us-center-2", 1.90, timestamp_ns),
            ]
        )
        second = engine.evaluate(timestamp_ns).sectors[Sector.CENTER]
        self.assertEqual(second.level, ExtendedHazardLevel.CRITICAL)
        self.assertAlmostEqual(second.fused_distance_m, 0.45, places=6)

    def test_glass_door_ultrasonic_is_trusted_once_persistent(self) -> None:
        engine = SmartCaneFusionEngine()
        for index in range(2):
            timestamp_ns = 500 * MS + index * 100 * MS
            engine.update_sensors(snapshot(1.90, timestamp_ns))
            engine.evaluate(timestamp_ns)
        first_ns = 700 * MS
        engine.update_sensors(
            [
                tof("tof-center", 1.90, first_ns),
                us("us-center-1", 0.60, first_ns),
                us("us-center-2", 1.90, first_ns),
            ]
        )
        first = engine.evaluate(first_ns).sectors[Sector.CENTER]
        self.assertEqual(first.level, ExtendedHazardLevel.CLEAR)
        self.assertGreater(first.fused_distance_m, 1.4)

        second_ns = 800 * MS
        engine.update_sensors(
            [
                tof("tof-center", 1.90, second_ns),
                us("us-center-1", 0.60, second_ns),
                us("us-center-2", 1.90, second_ns),
            ]
        )
        second = engine.evaluate(second_ns).sectors[Sector.CENTER]
        self.assertEqual(second.level, ExtendedHazardLevel.WARNING)
        self.assertAlmostEqual(second.fused_distance_m, 0.60, places=6)

    def test_single_ping_ghost_is_rejected_by_the_median_filter(self) -> None:
        engine = SmartCaneFusionEngine()
        for index in range(2):
            timestamp_ns = 1_000 * MS + index * 100 * MS
            engine.update_sensors(
                [
                    tof("tof-center", 0.45, timestamp_ns),
                    us("us-center-1", 0.45, timestamp_ns),
                    us("us-center-2", 0.47, timestamp_ns),
                ]
            )
            engine.evaluate(timestamp_ns)
        ghost_ns = 1_200 * MS
        engine.update_sensors(
            [
                tof("tof-center", 0.45, ghost_ns),
                us("us-center-1", 0.45, ghost_ns),
                us("us-center-2", 1.50, ghost_ns),
            ]
        )
        state = engine.evaluate(ghost_ns).sectors[Sector.CENTER]
        self.assertEqual(state.level, ExtendedHazardLevel.CRITICAL)
        self.assertLess(state.fused_distance_m, 0.55)

    def test_two_closer_samples_bypass_the_median(self) -> None:
        engine = SmartCaneFusionEngine()
        for index in range(2):
            timestamp_ns = 1_500 * MS + index * 100 * MS
            engine.update_sensor(tof("tof-center", 1.50, timestamp_ns))
            engine.evaluate(timestamp_ns)
        single_ns = 1_700 * MS
        engine.update_sensor(tof("tof-center", 0.45, single_ns))
        single = engine.evaluate(single_ns).sectors[Sector.CENTER]
        self.assertGreater(single.fused_distance_m, 1.40)
        self.assertNotEqual(single.level, ExtendedHazardLevel.CRITICAL)

        confirmed_ns = 1_800 * MS
        engine.update_sensor(tof("tof-center", 0.45, confirmed_ns))
        confirmed = engine.evaluate(confirmed_ns).sectors[Sector.CENTER]
        self.assertAlmostEqual(confirmed.fused_distance_m, 0.45, places=6)
        self.assertEqual(confirmed.level, ExtendedHazardLevel.CRITICAL)

    def test_farther_dissenter_is_dropped_from_a_two_of_three_agreement(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 2_500 * MS
        engine.update_sensors(
            [
                tof("tof-center", 0.45, now),
                us("us-center-1", 0.45, now),
                us("us-center-2", 1.50, now),
            ]
        )
        state = engine.evaluate(now).sectors[Sector.CENTER]
        self.assertEqual(state.level, ExtendedHazardLevel.CRITICAL)
        self.assertAlmostEqual(state.fused_distance_m, 0.45, places=6)

    def test_floor_echo_is_suppressed_when_the_cane_pitches_down(self) -> None:
        engine = SmartCaneFusionEngine()
        tilted_at = tilt_engine_down(engine, 0)
        imu_state = engine.imu.state(tilted_at)
        self.assertTrue(imu_state.is_pitched_down)
        expected_floor = imu_state.expected_ground_distance_m
        self.assertIsNotNone(expected_floor)

        engine.update_sensors(
            [
                tof("tof-center", expected_floor, tilted_at),
                us("us-center-1", None, tilted_at, no_target=True),
                us("us-center-2", None, tilted_at, no_target=True),
            ]
        )
        state = engine.evaluate(tilted_at + 5 * MS).sectors[Sector.CENTER]
        self.assertEqual(state.level, ExtendedHazardLevel.CLEAR)
        self.assertIsNone(state.fused_distance_m)
        self.assertEqual(state.health, SectorHealth.OK)
        self.assertIn("floor echo", state.reason)

    def test_a_real_obstacle_below_a_floor_echo_is_kept(self) -> None:
        engine = SmartCaneFusionEngine()
        tilted_at = tilt_engine_down(engine, 0)
        engine.update_sensors(
            [
                tof("tof-center", 0.60, tilted_at),
                us("us-center-1", None, tilted_at, no_target=True),
                us("us-center-2", None, tilted_at, no_target=True),
            ]
        )
        state = engine.evaluate(tilted_at + 5 * MS).sectors[Sector.CENTER]
        self.assertEqual(state.level, ExtendedHazardLevel.WARNING)
        self.assertAlmostEqual(state.fused_distance_m, 0.60 * math.cos(math.radians(28.0)), delta=0.05)
        self.assertNotIn("floor echo", state.reason)

    def test_stale_and_invalid_readings_are_missing(self) -> None:
        engine = SmartCaneFusionEngine()
        engine.update_sensor(tof("tof-center", 0.45, 10 * MS))
        engine.update_sensor(tof("tof-left", -1.0, 300 * MS, Sector.LEFT, valid=False))
        decision = engine.evaluate(400 * MS)
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.SENSOR_FAULT)
        self.assertEqual(decision.fresh_range_sensor_count, 0)
        self.assertIn("tof-center", decision.degraded_sensors)
        self.assertIn("tof-left", decision.degraded_sensors)

    def test_no_target_is_a_live_clear_sensor(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 3_000 * MS
        engine.update_sensors(all_clear(now))
        decision = engine.evaluate(now + 10 * MS)
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.CLEAR)
        self.assertNotEqual(decision.overall_level, ExtendedHazardLevel.SENSOR_FAULT)
        self.assertEqual(decision.fresh_range_sensor_count, 5)
        self.assertEqual(decision.degraded_sensors, tuple())
        for sector in SECTORS:
            self.assertEqual(decision.sectors[sector].health, SectorHealth.OK)
            self.assertIsNone(decision.sectors[sector].fused_distance_m)

    def test_total_loss_of_range_sensing_is_a_sensor_fault(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 3_500 * MS
        engine.update_sensors(
            [
                tof("tof-left", -1.0, now, Sector.LEFT, valid=False),
                tof("tof-center", -1.0, now, valid=False),
                tof("tof-right", -1.0, now, Sector.RIGHT, valid=False),
                us("us-center-1", -1.0, now, valid=False),
                us("us-center-2", -1.0, now, valid=False),
            ]
        )
        decision = engine.evaluate(now + 10 * MS)
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.SENSOR_FAULT)
        self.assertEqual(decision.fresh_range_sensor_count, 0)
        self.assertEqual(len(decision.degraded_sensors), 5)
        for sector in SECTORS:
            self.assertEqual(decision.sectors[sector].level, ExtendedHazardLevel.SENSOR_FAULT)
            self.assertEqual(decision.sectors[sector].health, SectorHealth.FAULT)

    def test_partial_degradation_keeps_healthy_sectors_alive(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 4_000 * MS
        engine.update_sensors(
            [
                tof("tof-left", 0.45, now, Sector.LEFT),
                tof("tof-center", None, now, no_target=True),
                us("us-center-1", None, now, no_target=True),
                us("us-center-2", None, now, no_target=True),
                tof("tof-right", -1.0, now, Sector.RIGHT, valid=False),
            ]
        )
        decision = engine.evaluate(now + 10 * MS)
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.CRITICAL)
        self.assertEqual(decision.sectors[Sector.LEFT].level, ExtendedHazardLevel.CRITICAL)
        self.assertEqual(decision.sectors[Sector.LEFT].health, SectorHealth.OK)
        self.assertEqual(decision.sectors[Sector.CENTER].level, ExtendedHazardLevel.CLEAR)
        self.assertEqual(decision.sectors[Sector.CENTER].health, SectorHealth.OK)
        self.assertEqual(decision.sectors[Sector.RIGHT].level, ExtendedHazardLevel.SENSOR_FAULT)
        self.assertEqual(decision.sectors[Sector.RIGHT].health, SectorHealth.FAULT)
        self.assertIn("tof-right", decision.degraded_sensors)
        self.assertEqual(decision.fresh_range_sensor_count, 4)

    def test_one_dead_sector_does_not_raise_the_whole_cane(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 4_500 * MS
        readings = all_clear(now)
        readings = [
            tof("tof-right", -1.0, now, Sector.RIGHT, valid=False)
            if reading.sensor_id == "tof-right"
            else reading
            for reading in readings
        ]
        engine.update_sensors(readings)
        decision = engine.evaluate(now + 10 * MS)
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.CLEAR)
        self.assertEqual(decision.sectors[Sector.RIGHT].level, ExtendedHazardLevel.SENSOR_FAULT)
        self.assertIn("tof-right", decision.degraded_sensors)

    def test_partially_missing_sector_is_degraded(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 4_800 * MS
        engine.update_sensors(
            [
                tof("tof-center", 0.80, now),
                us("us-center-1", 0.82, now),
            ]
        )
        state = engine.evaluate(now).sectors[Sector.CENTER]
        self.assertEqual(state.health, SectorHealth.DEGRADED)
        self.assertEqual(state.missing_sensor_ids, ("us-center-2",))
        self.assertIn("us-center-2", state.missing_sensor_ids)

    def test_nearest_distance_reports_the_closest_sector(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 5_000 * MS
        engine.update_sensors(
            [
                tof("tof-left", 1.30, now, Sector.LEFT),
                tof("tof-center", 0.60, now),
                tof("tof-right", 0.90, now, Sector.RIGHT),
            ]
        )
        decision = engine.evaluate(now)
        self.assertAlmostEqual(decision.nearest_distance_m, 0.60, places=6)
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.WARNING)


class HysteresisTests(unittest.TestCase):
    def _feed(self, engine, distance_m, timestamp_ns):
        engine.update_sensors(snapshot(distance_m, timestamp_ns))
        return engine.evaluate(timestamp_ns).sectors[Sector.CENTER]

    def test_critical_holds_inside_the_release_deadband(self) -> None:
        engine = SmartCaneFusionEngine()
        critical = self._feed(engine, 0.45, 100 * MS)
        self.assertEqual(critical.level, ExtendedHazardLevel.CRITICAL)

        held = self._feed(engine, 0.53, 250 * MS)
        self.assertEqual(held.level, ExtendedHazardLevel.CRITICAL)
        self.assertLess(held.fused_distance_m, 0.58)

        still_held = self._feed(engine, 0.56, 400 * MS)
        self.assertEqual(still_held.level, ExtendedHazardLevel.CRITICAL)

    def test_critical_releases_to_warning_past_the_deadband(self) -> None:
        engine = SmartCaneFusionEngine()
        self._feed(engine, 0.45, 100 * MS)
        released = None
        for index in range(1, 6):
            released = self._feed(engine, 0.62, (100 + index * 150) * MS)
        self.assertEqual(released.level, ExtendedHazardLevel.WARNING)
        self.assertGreater(released.fused_distance_m, 0.58)

    def test_warning_holds_inside_its_release_band(self) -> None:
        engine = SmartCaneFusionEngine()
        warning = self._feed(engine, 0.90, 100 * MS)
        self.assertEqual(warning.level, ExtendedHazardLevel.WARNING)
        held = self._feed(engine, 1.05, 250 * MS)
        self.assertEqual(held.level, ExtendedHazardLevel.WARNING)
        self.assertLess(held.fused_distance_m, 1.12)
        still_held = self._feed(engine, 1.05, 400 * MS)
        self.assertEqual(still_held.level, ExtendedHazardLevel.WARNING)

    def test_warning_releases_to_advisory_past_the_band(self) -> None:
        engine = SmartCaneFusionEngine()
        self._feed(engine, 0.90, 100 * MS)
        released = None
        for index in range(1, 6):
            released = self._feed(engine, 1.25, (100 + index * 150) * MS)
        self.assertEqual(released.level, ExtendedHazardLevel.ADVISORY)
        self.assertGreater(released.fused_distance_m, 1.12)

    def test_advisory_releases_to_clear_once_risk_drops(self) -> None:
        engine = SmartCaneFusionEngine()
        advisory = self._feed(engine, 1.45, 100 * MS)
        self.assertEqual(advisory.level, ExtendedHazardLevel.ADVISORY)
        cleared = None
        for index in range(1, 8):
            cleared = self._feed(engine, 2.40, (100 + index * 150) * MS)
        self.assertEqual(cleared.level, ExtendedHazardLevel.CLEAR)

    def test_escalation_needs_no_dwell(self) -> None:
        engine = SmartCaneFusionEngine()
        state = self._feed(engine, 0.42, 10 * MS)
        self.assertEqual(state.level, ExtendedHazardLevel.CRITICAL)


class TrackingTests(unittest.TestCase):
    def _approach(self, blocks):
        engine = SmartCaneFusionEngine()
        timestamp_ns = 1_000 * MS
        counter = 0
        state = None
        for distance in blocks:
            for _ in range(4):
                timestamp_ns = 1_000 * MS + counter * 50 * MS
                engine.update_sensors(
                    [
                        tof("tof-center", distance, timestamp_ns),
                        us("us-center-1", None, timestamp_ns, no_target=True),
                        us("us-center-2", None, timestamp_ns, no_target=True),
                    ]
                )
                counter += 1
            state = engine.evaluate(timestamp_ns).sectors[Sector.CENTER]
        return state

    def test_ttc_escalates_before_the_distance_threshold(self) -> None:
        state = self._approach([2.00, 1.40, 0.95])
        self.assertEqual(state.level, ExtendedHazardLevel.WARNING)
        self.assertGreater(state.fused_distance_m, 1.00)
        self.assertIsNotNone(state.ttc_s)
        self.assertLess(state.ttc_s, 0.85)
        self.assertGreater(state.ttc_s, 0.40)
        self.assertTrue(state.ttc_triggered)
        self.assertGreater(state.closing_velocity_mps, 1.0)

    def test_receding_target_reports_no_time_to_contact(self) -> None:
        state = self._approach([0.95, 1.25, 1.55])
        self.assertIsNone(state.ttc_s)
        self.assertFalse(state.ttc_triggered)
        self.assertLess(state.closing_velocity_mps, 0.0)

    def test_stationary_target_has_no_time_to_contact(self) -> None:
        engine = SmartCaneFusionEngine()
        for index in range(6):
            timestamp_ns = 2_000 * MS + index * 100 * MS
            engine.update_sensor(tof("tof-center", 1.20, timestamp_ns))
            state = engine.evaluate(timestamp_ns).sectors[Sector.CENTER]
        self.assertIsNone(state.ttc_s)
        self.assertAlmostEqual(state.closing_velocity_mps, 0.0, delta=0.05)


class VisionTests(unittest.TestCase):
    def test_single_frame_flicker_does_not_advise(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 1_000 * MS
        engine.update_sensors(all_clear(now))
        engine.update_vision(frame(1, now, [synthetic_detection("person", 2.2, center_x=100)]))
        decision = engine.evaluate(now + 10 * MS)
        self.assertEqual(decision.sectors[Sector.LEFT].level, ExtendedHazardLevel.CLEAR)
        self.assertEqual(decision.sectors[Sector.LEFT].semantic_labels, ("person",))
        self.assertFalse(decision.sectors[Sector.LEFT].vision_triggered)

    def test_two_frames_trigger_a_left_advisory(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 1_000 * MS
        engine.update_sensors(all_clear(now))
        engine.update_vision(frame(1, now, [synthetic_detection("person", 2.2, center_x=100)]))
        engine.evaluate(now + 10 * MS)
        later = now + 80 * MS
        engine.update_sensors(all_clear(later))
        engine.update_vision(
            frame(2, later, [synthetic_detection("person", 2.2, center_x=100)])
        )
        decision = engine.evaluate(later + 10 * MS)
        self.assertEqual(decision.sectors[Sector.LEFT].level, ExtendedHazardLevel.ADVISORY)
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.ADVISORY)
        self.assertTrue(decision.sectors[Sector.LEFT].vision_triggered)
        self.assertFalse(decision.sectors[Sector.LEFT].range_triggered)
        self.assertEqual(decision.sectors[Sector.CENTER].level, ExtendedHazardLevel.CLEAR)
        self.assertEqual(decision.sectors[Sector.RIGHT].level, ExtendedHazardLevel.CLEAR)

    def test_repeated_evaluation_of_one_frame_counts_once(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 1_000 * MS
        engine.update_sensors(all_clear(now))
        engine.update_vision(frame(1, now, [synthetic_detection("person", 2.2, center_x=100)]))
        engine.evaluate(now + 10 * MS)
        decision = engine.evaluate(now + 20 * MS)
        self.assertEqual(decision.sectors[Sector.LEFT].level, ExtendedHazardLevel.CLEAR)

    def test_range_blind_stairs_produce_a_warning(self) -> None:
        engine = SmartCaneFusionEngine()
        start = 2_000 * MS
        decision = None
        for index in range(2):
            timestamp_ns = start + index * 80 * MS
            engine.update_sensors(all_clear(timestamp_ns))
            engine.update_vision(
                frame(index + 1, timestamp_ns, [synthetic_detection("stairs", 1.6)])
            )
            decision = engine.evaluate(timestamp_ns + 10 * MS)
        state = decision.sectors[Sector.CENTER]
        self.assertEqual(state.level, ExtendedHazardLevel.WARNING)
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.WARNING)
        self.assertTrue(state.vision_triggered)
        self.assertFalse(state.range_triggered)
        self.assertEqual(state.semantic_labels, ("stairs",))

    def test_range_blind_curb_produces_a_warning(self) -> None:
        engine = SmartCaneFusionEngine()
        start = 2_500 * MS
        decision = None
        for index in range(2):
            timestamp_ns = start + index * 80 * MS
            engine.update_sensors(all_clear(timestamp_ns))
            engine.update_vision(
                frame(index + 1, timestamp_ns, [synthetic_detection("curb", 1.2)])
            )
            decision = engine.evaluate(timestamp_ns + 10 * MS)
        self.assertEqual(
            decision.sectors[Sector.CENTER].level, ExtendedHazardLevel.WARNING
        )

    def test_vision_alone_never_reaches_critical(self) -> None:
        engine = SmartCaneFusionEngine()
        start = 3_000 * MS
        for index in range(2):
            timestamp_ns = start + index * 80 * MS
            engine.update_sensors(all_clear(timestamp_ns))
            engine.update_vision(
                frame(index + 1, timestamp_ns, [synthetic_detection("stairs", 0.6)])
            )
            decision = engine.evaluate(timestamp_ns + 10 * MS)
        self.assertEqual(
            decision.sectors[Sector.CENTER].level, ExtendedHazardLevel.WARNING
        )
        self.assertNotEqual(
            decision.sectors[Sector.CENTER].level, ExtendedHazardLevel.CRITICAL
        )

    def test_class_without_warning_privilege_stays_advisory(self) -> None:
        engine = SmartCaneFusionEngine()
        start = 3_500 * MS
        for index in range(2):
            timestamp_ns = start + index * 80 * MS
            engine.update_sensors(all_clear(timestamp_ns))
            engine.update_vision(
                frame(index + 1, timestamp_ns, [synthetic_detection("person", 0.7)])
            )
            decision = engine.evaluate(timestamp_ns + 10 * MS)
        self.assertEqual(
            decision.sectors[Sector.CENTER].level, ExtendedHazardLevel.ADVISORY
        )

    def test_vision_never_suppresses_a_short_range_hazard(self) -> None:
        engine = SmartCaneFusionEngine()
        start = 4_000 * MS
        for index in range(2):
            timestamp_ns = start + index * 80 * MS
            engine.update_sensors(snapshot(0.42, timestamp_ns))
            engine.update_vision(
                frame(index + 1, timestamp_ns, [synthetic_detection("door", 3.0)])
            )
            decision = engine.evaluate(timestamp_ns + 10 * MS)
        state = decision.sectors[Sector.CENTER]
        self.assertEqual(state.level, ExtendedHazardLevel.CRITICAL)
        self.assertAlmostEqual(state.fused_distance_m, 0.42, delta=0.02)
        self.assertGreater(state.range_weight, 0.9)
        self.assertLess(state.vision_weight, 0.1)

    def test_stale_vision_releases_the_advisory(self) -> None:
        engine = SmartCaneFusionEngine()
        start = 5_000 * MS
        for index in range(2):
            timestamp_ns = start + index * 80 * MS
            engine.update_sensors(all_clear(timestamp_ns))
            engine.update_vision(
                frame(index + 1, timestamp_ns, [synthetic_detection("person", 2.2, center_x=100)])
            )
            engine.evaluate(timestamp_ns + 10 * MS)
        later = start + 700 * MS
        engine.update_sensors(all_clear(later))
        decision = engine.evaluate(later)
        self.assertEqual(decision.sectors[Sector.LEFT].level, ExtendedHazardLevel.CLEAR)
        self.assertEqual(decision.sectors[Sector.LEFT].semantic_labels, tuple())

    def test_yaw_lag_compensation_moves_the_vision_sector(self) -> None:
        def sectors_for(yaw_rate_rps: float):
            engine = SmartCaneFusionEngine()
            engine.imu.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
            engine.update_imu(imu_sample(0, yaw_rate_rps=0.0))
            now = 1_000 * MS
            engine.update_imu(imu_sample(now, yaw_rate_rps=yaw_rate_rps))
            engine.update_sensors(all_clear(now))
            for index in range(2):
                captured = now - 200 * MS + index * 10 * MS
                engine.update_vision(
                    frame(
                        index + 1,
                        captured,
                        [synthetic_detection("person", 2.2, center_x=210.0)],
                    )
                )
            return engine.evaluate(now + 20 * MS)

        static = sectors_for(0.0)
        self.assertEqual(static.sectors[Sector.LEFT].semantic_labels, ("person",))
        self.assertEqual(static.sectors[Sector.CENTER].semantic_labels, tuple())

        yawed = sectors_for(1.0)
        self.assertEqual(yawed.sectors[Sector.CENTER].semantic_labels, ("person",))
        self.assertEqual(yawed.sectors[Sector.LEFT].semantic_labels, tuple())

    def test_low_confidence_detections_are_dropped(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 6_000 * MS
        engine.update_sensors(all_clear(now))
        for index in range(2):
            timestamp_ns = now + index * 80 * MS
            engine.update_sensors(all_clear(timestamp_ns))
            engine.update_vision(
                frame(
                    index + 1,
                    timestamp_ns,
                    [synthetic_detection("person", 2.0, confidence=0.30)],
                )
            )
            decision = engine.evaluate(timestamp_ns + 10 * MS)
        self.assertEqual(decision.sectors[Sector.CENTER].semantic_labels, tuple())
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.CLEAR)

    def test_distant_detections_exceed_the_vision_range(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 6_500 * MS
        engine.update_sensors(all_clear(now))
        for index in range(2):
            timestamp_ns = now + index * 80 * MS
            engine.update_sensors(all_clear(timestamp_ns))
            engine.update_vision(
                frame(
                    index + 1,
                    timestamp_ns,
                    [synthetic_detection("person", 5.0, confidence=0.95)],
                )
            )
            decision = engine.evaluate(timestamp_ns + 10 * MS)
        self.assertEqual(decision.sectors[Sector.CENTER].semantic_labels, tuple())


class DirectionalHapticTests(unittest.TestCase):
    def _decision(self, readings, now):
        engine = SmartCaneFusionEngine()
        engine.update_sensors(readings)
        return engine.evaluate(now + 10 * MS)

    def test_directional_haptics_drive_each_motor_independently(self) -> None:
        now = 1_000 * MS
        decision = self._decision(
            [
                tof("tof-left", 0.45, now, Sector.LEFT),
                tof("tof-center", None, now, no_target=True),
                us("us-center-1", None, now, no_target=True),
                us("us-center-2", None, now, no_target=True),
                tof("tof-right", 0.90, now, Sector.RIGHT),
            ],
            now,
        )
        self.assertEqual(decision.sectors[Sector.LEFT].level, ExtendedHazardLevel.CRITICAL)
        self.assertEqual(decision.sectors[Sector.CENTER].level, ExtendedHazardLevel.CLEAR)
        self.assertEqual(decision.sectors[Sector.RIGHT].level, ExtendedHazardLevel.WARNING)

        haptics = decision_to_directional_haptics(decision)
        self.assertIsInstance(haptics, DirectionalHapticFrame)
        self.assertEqual(set(haptics.motors), set(SECTORS))

        left = haptics.motors[Sector.LEFT]
        self.assertEqual(left.level, ExtendedHazardLevel.CRITICAL)
        self.assertEqual(left.intensity, 1.0)
        self.assertEqual(left.pulse_period_ms, 100)
        self.assertEqual(left.duty_cycle, 0.70)
        self.assertEqual(left.waveform_id, 47)

        right = haptics.motors[Sector.RIGHT]
        self.assertEqual(right.level, ExtendedHazardLevel.WARNING)
        self.assertGreaterEqual(right.intensity, 0.60)
        self.assertLessEqual(right.intensity, 0.88)
        self.assertGreaterEqual(right.pulse_period_ms, 160)
        self.assertLessEqual(right.pulse_period_ms, 250)
        self.assertEqual(right.duty_cycle, 0.45)
        self.assertEqual(right.waveform_id, 14)

        center = haptics.motors[Sector.CENTER]
        self.assertEqual(center.level, ExtendedHazardLevel.CLEAR)
        self.assertEqual(center.intensity, 0.0)
        self.assertIsNone(center.pulse_period_ms)
        self.assertEqual(center.duty_cycle, 0.0)
        self.assertEqual(center.waveform_id, 0)

    def test_advisory_uses_the_soft_cue(self) -> None:
        now = 1_500 * MS
        decision = self._decision(
            [
                tof("tof-left", 1.45, now, Sector.LEFT),
                tof("tof-center", None, now, no_target=True),
                us("us-center-1", None, now, no_target=True),
                us("us-center-2", None, now, no_target=True),
                tof("tof-right", None, now, Sector.RIGHT, no_target=True),
            ],
            now,
        )
        self.assertEqual(decision.sectors[Sector.LEFT].level, ExtendedHazardLevel.ADVISORY)
        advisory = decision_to_directional_haptics(decision).motors[Sector.LEFT]
        self.assertGreaterEqual(advisory.intensity, 0.28)
        self.assertLessEqual(advisory.intensity, 0.48)
        self.assertEqual(advisory.pulse_period_ms, 520)
        self.assertEqual(advisory.duty_cycle, 0.25)
        self.assertEqual(advisory.waveform_id, 1)

    def test_total_fault_pulses_all_motors(self) -> None:
        now = 2_000 * MS
        decision = self._decision(
            [
                tof("tof-left", -1.0, now, Sector.LEFT, valid=False),
                tof("tof-center", -1.0, now, valid=False),
                tof("tof-right", -1.0, now, Sector.RIGHT, valid=False),
                us("us-center-1", -1.0, now, valid=False),
                us("us-center-2", -1.0, now, valid=False),
            ],
            now,
        )
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.SENSOR_FAULT)
        haptics = decision_to_directional_haptics(decision)
        for sector in SECTORS:
            motor = haptics.motors[sector]
            self.assertEqual(motor.level, ExtendedHazardLevel.SENSOR_FAULT)
            self.assertAlmostEqual(motor.intensity, 0.35, places=6)
            self.assertEqual(motor.pulse_period_ms, 1000)
            self.assertAlmostEqual(motor.duty_cycle, 0.20, places=6)
            self.assertEqual(motor.waveform_id, 10)

    def test_partial_fault_is_silent_unless_degraded_pulse_is_enabled(self) -> None:
        now = 2_500 * MS
        readings = [
            tof("tof-left", 0.45, now, Sector.LEFT),
            tof("tof-center", None, now, no_target=True),
            us("us-center-1", None, now, no_target=True),
            us("us-center-2", None, now, no_target=True),
            tof("tof-right", -1.0, now, Sector.RIGHT, valid=False),
        ]
        decision = self._decision(readings, now)
        self.assertEqual(decision.sectors[Sector.RIGHT].level, ExtendedHazardLevel.SENSOR_FAULT)
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.CRITICAL)
        silent = decision_to_directional_haptics(decision).motors[Sector.RIGHT]
        self.assertEqual(silent.level, ExtendedHazardLevel.CLEAR)
        self.assertEqual(silent.intensity, 0.0)

        ping = decision_to_directional_haptics(
            decision, SmartCaneFusionConfig(degraded_sector_pulse=True)
        ).motors[Sector.RIGHT]
        self.assertAlmostEqual(ping.intensity, 0.35, places=6)
        self.assertEqual(ping.pulse_period_ms, 1000)
        self.assertEqual(ping.waveform_id, 10)

    def test_stationary_softens_warning_but_never_critical(self) -> None:
        now = 3_000 * MS
        warning = self._decision(
            [
                tof("tof-left", 0.90, now, Sector.LEFT),
                tof("tof-center", 0.42, now),
                us("us-center-1", 0.42, now),
                us("us-center-2", 0.42, now),
                tof("tof-right", None, now, Sector.RIGHT, no_target=True),
            ],
            now,
        )
        moving_frame = decision_to_directional_haptics(warning)
        still_decision = replace(warning, imu_state=stationary_state(now))
        still_frame = decision_to_directional_haptics(still_decision)

        self.assertAlmostEqual(
            still_frame.motors[Sector.LEFT].intensity,
            moving_frame.motors[Sector.LEFT].intensity * 0.65,
            places=6,
        )
        self.assertEqual(still_frame.motors[Sector.CENTER].intensity, 1.0)
        self.assertEqual(
            still_frame.motors[Sector.CENTER].waveform_id,
            moving_frame.motors[Sector.CENTER].waveform_id,
        )

    def test_haptic_frame_serialises_three_motors(self) -> None:
        now = 4_000 * MS
        decision = self._decision(all_clear(now), now)
        payload = decision_to_directional_haptics(decision).as_dict()
        self.assertEqual(set(payload["motors"]), {"left", "center", "right"})
        self.assertEqual(payload["overall_level"], "clear")
        for sector in ("left", "center", "right"):
            self.assertIn("waveform_id", payload["motors"][sector])

    def test_recording_sink_collects_frames(self) -> None:
        now = 4_500 * MS
        decision = self._decision(all_clear(now), now)
        sink = RecordingDirectionalHapticSink()
        sink.emit_frame(decision_to_directional_haptics(decision))
        self.assertEqual(len(sink.frames), 1)
        self.assertFalse(sink.stopped)
        sink.stop()
        self.assertTrue(sink.stopped)


class LegacyMappingTests(unittest.TestCase):
    def _decision(self, readings, now) -> DirectionalHazardDecision:
        engine = SmartCaneFusionEngine()
        engine.update_sensors(readings)
        return engine.evaluate(now + 10 * MS)

    def test_critical_maps_to_legacy_critical_with_the_sector(self) -> None:
        now = 1_000 * MS
        decision = self._decision(
            [
                tof("tof-left", 0.42, now, Sector.LEFT),
                tof("tof-center", None, now, no_target=True),
                us("us-center-1", None, now, no_target=True),
                us("us-center-2", None, now, no_target=True),
                tof("tof-right", None, now, Sector.RIGHT, no_target=True),
            ],
            now,
        )
        legacy = decision.to_legacy_decision()
        self.assertEqual(legacy.level, HazardLevel.CRITICAL)
        self.assertEqual(legacy.sectors, (Sector.LEFT,))
        self.assertAlmostEqual(legacy.nearest_distance_m, 0.42, delta=0.02)
        self.assertTrue(legacy.range_triggered)
        self.assertEqual(legacy.fresh_range_sensor_count, 5)
        self.assertEqual(legacy.timestamp_ns, decision.timestamp_ns)

    def test_advisory_maps_to_legacy_clear(self) -> None:
        now = 1_500 * MS
        decision = self._decision(
            [
                tof("tof-left", 1.45, now, Sector.LEFT),
                tof("tof-center", None, now, no_target=True),
                us("us-center-1", None, now, no_target=True),
                us("us-center-2", None, now, no_target=True),
                tof("tof-right", None, now, Sector.RIGHT, no_target=True),
            ],
            now,
        )
        self.assertEqual(decision.sectors[Sector.LEFT].level, ExtendedHazardLevel.ADVISORY)
        legacy = decision.to_legacy_decision()
        self.assertEqual(legacy.level, HazardLevel.CLEAR)
        self.assertEqual(legacy.sectors, tuple())

    def test_total_fault_maps_to_unknown_sector(self) -> None:
        now = 2_000 * MS
        decision = self._decision(
            [
                tof("tof-left", -1.0, now, Sector.LEFT, valid=False),
                tof("tof-center", -1.0, now, valid=False),
                tof("tof-right", -1.0, now, Sector.RIGHT, valid=False),
                us("us-center-1", -1.0, now, valid=False),
                us("us-center-2", -1.0, now, valid=False),
            ],
            now,
        )
        legacy = decision.to_legacy_decision()
        self.assertEqual(legacy.level, HazardLevel.SENSOR_FAULT)
        self.assertEqual(legacy.sectors, (Sector.UNKNOWN,))
        self.assertEqual(legacy.fresh_range_sensor_count, 0)

    def test_decision_dict_exposes_three_sectors_and_imu(self) -> None:
        now = 3_000 * MS
        decision = self._decision(all_clear(now), now)
        payload = decision.as_dict()
        self.assertEqual(set(payload["sectors"]), {"left", "center", "right"})
        self.assertEqual(payload["overall_level"], "clear")
        self.assertIn("imu", payload)
        self.assertFalse(payload["imu"]["fresh"])
        center = payload["sectors"]["center"]
        self.assertEqual(center["health"], "ok")
        self.assertEqual(center["level"], "clear")


class EngineLifecycleTests(unittest.TestCase):
    def test_reset_clears_sensor_tracker_and_vision_state(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 1_000 * MS
        engine.update_sensors(snapshot(0.42, now))
        engine.update_vision(frame(1, now, [synthetic_detection("person", 2.0)]))
        self.assertEqual(
            engine.evaluate(now).sectors[Sector.CENTER].level,
            ExtendedHazardLevel.CRITICAL,
        )
        engine.reset()
        decision = engine.evaluate(now + 10 * MS)
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.SENSOR_FAULT)
        self.assertEqual(decision.sectors[Sector.CENTER].semantic_labels, tuple())
        self.assertIsNone(decision.sectors[Sector.CENTER].fused_distance_m)

    def test_empty_expected_sensors_disables_fault_reporting(self) -> None:
        config = SmartCaneFusionConfig(expected_sensors={})
        engine = SmartCaneFusionEngine(config)
        decision = engine.evaluate(1_000 * MS)
        self.assertEqual(decision.overall_level, ExtendedHazardLevel.CLEAR)
        self.assertEqual(decision.expected_range_sensor_count, 0)
        self.assertEqual(decision.fresh_range_sensor_count, 0)
        self.assertEqual(decision.degraded_sensors, tuple())

    def test_custom_expected_sensor_map(self) -> None:
        config = SmartCaneFusionConfig(
            expected_sensors={"sonar-left": Sector.LEFT, "sonar-right": Sector.RIGHT}
        )
        engine = SmartCaneFusionEngine(config)
        now = 2_000 * MS
        engine.update_sensors(
            [
                us("sonar-left", 0.45, now, Sector.LEFT),
                us("sonar-right", 1.80, now, Sector.RIGHT),
            ]
        )
        decision = engine.evaluate(now)
        self.assertEqual(decision.expected_range_sensor_count, 2)
        self.assertEqual(decision.sectors[Sector.LEFT].level, ExtendedHazardLevel.CRITICAL)
        self.assertEqual(decision.sectors[Sector.LEFT].health, SectorHealth.OK)
        self.assertEqual(decision.sectors[Sector.CENTER].health, SectorHealth.OK)

    def test_sensor_reading_objects_are_accepted(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 3_000 * MS
        engine.update_sensor(
            SensorReading(
                sensor_id="tof-center",
                kind=SensorKind.TOF,
                timestamp_ns=now,
                sector=Sector.CENTER,
                distance_m=0.48,
            )
        )
        state = engine.evaluate(now).sectors[Sector.CENTER]
        self.assertEqual(state.level, ExtendedHazardLevel.CRITICAL)

    def test_backwards_sensor_timestamps_are_ignored(self) -> None:
        engine = SmartCaneFusionEngine()
        now = 4_000 * MS
        engine.update_sensor(tof("tof-center", 0.45, now))
        engine.update_sensor(tof("tof-center", 1.60, now - 100 * MS))
        state = engine.evaluate(now).sectors[Sector.CENTER]
        self.assertAlmostEqual(state.fused_distance_m, 0.45, places=6)

    def test_calibrate_imu_feeds_the_shared_processor(self) -> None:
        engine = SmartCaneFusionEngine()
        bias = engine.calibrate_imu([imu_sample(index * 10 * MS) for index in range(12)])
        self.assertEqual(bias, (0.0, 0.0, 0.0))
        self.assertTrue(engine.imu.calibrated)
        state = engine.evaluate(200 * MS)
        self.assertFalse(state.imu_state.fresh)


class ThreadSafetyTests(unittest.TestCase):
    def test_concurrent_updates_and_evaluations_stay_consistent(self) -> None:
        engine = SmartCaneFusionEngine()
        errors = []
        start = threading.Barrier(5)

        def range_worker(offset_ms: int) -> None:
            try:
                start.wait()
                for index in range(150):
                    timestamp_ns = 1_000 * MS + index * MS
                    engine.update_sensors(
                        [
                            tof("tof-center", 0.60 + 0.001 * offset_ms, timestamp_ns),
                            us("us-center-1", 0.62, timestamp_ns),
                            us("us-center-2", 0.64, timestamp_ns),
                            tof("tof-left", 1.30, timestamp_ns, Sector.LEFT),
                            tof("tof-right", 1.90, timestamp_ns, Sector.RIGHT),
                        ]
                    )
            except Exception as error:  # pragma: no cover - failure path
                errors.append(error)

        def evaluate_worker() -> None:
            try:
                start.wait()
                for index in range(150):
                    decision = engine.evaluate(1_000 * MS + index * MS)
                    if set(decision.sectors) != set(SECTORS):
                        errors.append(AssertionError("missing sector"))
                    if decision.overall_level not in ExtendedHazardLevel:
                        errors.append(AssertionError("bad level"))
            except Exception as error:  # pragma: no cover - failure path
                errors.append(error)

        def imu_worker() -> None:
            try:
                start.wait()
                for index in range(150):
                    engine.update_imu(imu_sample(1_000 * MS + index * MS))
            except Exception as error:  # pragma: no cover - failure path
                errors.append(error)

        def vision_worker() -> None:
            try:
                start.wait()
                for index in range(150):
                    captured = 1_000 * MS + index * MS
                    engine.update_vision(
                        frame(index + 1, captured, [synthetic_detection("person", 2.4)])
                    )
            except Exception as error:  # pragma: no cover - failure path
                errors.append(error)

        threads = [
            threading.Thread(target=range_worker, args=(0,)),
            threading.Thread(target=range_worker, args=(1,)),
            threading.Thread(target=evaluate_worker),
            threading.Thread(target=imu_worker),
            threading.Thread(target=vision_worker),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)
        self.assertEqual(errors, [])
        final = engine.evaluate(1_150 * MS)
        self.assertEqual(set(final.sectors), set(SECTORS))


class ConfigTests(unittest.TestCase):
    def test_default_priors_cover_the_fifteen_cane_classes(self) -> None:
        expected = {
            "stairs": (1.00, 0.32, 1.20, True),
            "vehicle": (0.95, 0.35, 1.55, True),
            "bicycle_motorcycle": (0.90, 0.35, 1.35, True),
            "overhead_obstacle": (0.90, 0.35, 0.80, True),
            "curb": (0.85, 0.32, 0.20, True),
            "barrier": (0.78, 0.35, 1.00, False),
            "person": (0.75, 0.35, 1.68, False),
            "pole": (0.72, 0.35, 2.20, False),
            "dog": (0.70, 0.35, 0.55, False),
            "slope": (0.70, 0.35, 1.00, False),
            "generic_obstacle": (0.65, 0.38, 0.90, False),
            "tree": (0.62, 0.38, 2.50, False),
            "cone": (0.60, 0.38, 0.65, False),
            "signboard": (0.55, 0.40, 1.50, False),
            "door": (0.50, 0.40, 2.00, False),
        }
        self.assertEqual(len(DEFAULT_CLASS_PRIORS), 15)
        for name, (severity, min_confidence, height, can_warn) in expected.items():
            prior = DEFAULT_CLASS_PRIORS[name]
            self.assertIsInstance(prior, ClassVisionPrior)
            self.assertAlmostEqual(prior.severity, severity, places=6)
            self.assertAlmostEqual(prior.min_confidence, min_confidence, places=6)
            self.assertAlmostEqual(prior.canonical_height_m, height, places=6)
            self.assertEqual(prior.can_trigger_warning, can_warn)

    def test_default_expected_sensor_map(self) -> None:
        config = SmartCaneFusionConfig()
        self.assertEqual(
            config.expected_sensors,
            {
                "tof-left": Sector.LEFT,
                "tof-center": Sector.CENTER,
                "tof-right": Sector.RIGHT,
                "us-center-1": Sector.CENTER,
                "us-center-2": Sector.CENTER,
            },
        )
        self.assertEqual(config.stationary_noncritical_scale, 0.65)
        self.assertFalse(config.degraded_sector_pulse)
        self.assertEqual(config.closer_outlier_persist_count, 2)

    def test_threshold_ordering_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            SmartCaneFusionConfig(critical_distance_m=0.0)
        with self.assertRaises(ValueError):
            SmartCaneFusionConfig(critical_distance_m=1.20, warning_distance_m=1.00)
        with self.assertRaises(ValueError):
            SmartCaneFusionConfig(warning_distance_m=2.00, advisory_distance_m=1.90)
        with self.assertRaises(ValueError):
            SmartCaneFusionConfig(critical_release_m=0.40)
        with self.assertRaises(ValueError):
            SmartCaneFusionConfig(warning_release_m=0.80)
        with self.assertRaises(ValueError):
            SmartCaneFusionConfig(advisory_risk_on=0.10, advisory_risk_off=0.20)
        with self.assertRaises(ValueError):
            SmartCaneFusionConfig(sensor_stale_after_ms=0.0)
        with self.assertRaises(ValueError):
            SmartCaneFusionConfig(center_fraction=1.0)
        with self.assertRaises(ValueError):
            SmartCaneFusionConfig(ttc_critical_s=0.90, ttc_warning_s=0.80)
        with self.assertRaises(ValueError):
            SmartCaneFusionConfig(vision_persist_frames=0)
        with self.assertRaises(ValueError):
            SmartCaneFusionConfig(stationary_noncritical_scale=1.5)

    def test_config_is_immutable(self) -> None:
        config = SmartCaneFusionConfig()
        with self.assertRaises(Exception):
            config.critical_distance_m = 0.40


if __name__ == "__main__":
    unittest.main()
