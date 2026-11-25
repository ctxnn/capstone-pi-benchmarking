import math
import unittest

from yolo_pi.cane.config import (
    CaneConfig,
    ConfigError,
    PriorityConfig,
    config_from_dict,
)
from yolo_pi.cane.patterns import build_cycle, cycle_length
from yolo_pi.cane.priority import (
    PriorityFusion,
    command_for,
    estimate_distance_m,
)
from yolo_pi.fusion import FusionConfig, HazardLevel, SensorKind, SensorReading
from yolo_pi.types import BoundingBox, Detection, FrameDetections, Sector

MS = 1_000_000
PRIORITY = PriorityConfig()
FOCAL = 240.0 / math.tan(math.radians(PRIORITY.vertical_fov_deg) / 2)


def person(sector, distance, confidence=0.9, name="person", height_m=1.7):
    box_h = FOCAL * height_m / distance
    cx = {Sector.LEFT: 100.0, Sector.CENTER: 320.0, Sector.RIGHT: 540.0}[sector]
    return Detection(0, name, confidence,
                     BoundingBox(cx - 30, 240 - box_h / 2, cx + 30, 240 + box_h / 2), sector)


def frame(ts, *detections):
    return FrameDetections(1, ts, ts, ts, 640, 480, list(detections))


def tof(sensor_id, sector, distance, ts):
    return SensorReading(sensor_id, SensorKind.TOF, ts, sector, distance)


def us(sensor_id, distance, ts, sector=Sector.CENTER):
    return SensorReading(sensor_id, SensorKind.ULTRASONIC, ts, sector, distance)


def make():
    return PriorityFusion(FusionConfig(), PRIORITY, overhead_ids=["us-2"])


def clear_ranges(fusion, ts):
    for sensor_id, sector in (("tof-left", Sector.LEFT), ("tof-center", Sector.CENTER),
                              ("tof-right", Sector.RIGHT)):
        fusion.update_sensor(tof(sensor_id, sector, 2.0, ts))
    fusion.update_sensor(us("us-1", 4.0, ts))
    fusion.update_sensor(us("us-2", 4.0, ts))


class DistanceEstimateTests(unittest.TestCase):
    def test_a_person_at_three_metres_is_estimated_close_to_three(self) -> None:
        estimate = estimate_distance_m(person(Sector.CENTER, 3.0), 640, 480, PRIORITY)
        self.assertAlmostEqual(estimate, 3.0, places=2)

    def test_a_box_clipped_by_the_frame_is_pulled_into_the_alert_band(self) -> None:
        clipped = Detection(0, "person", 0.9, BoundingBox(300, 0, 340, 120), Sector.CENTER)
        self.assertLessEqual(estimate_distance_m(clipped, 640, 480, PRIORITY), PRIORITY.far_max_m)

    def test_tiny_boxes_give_no_estimate(self) -> None:
        sliver = Detection(0, "person", 0.9, BoundingBox(0, 100, 5, 101), Sector.LEFT)
        self.assertIsNone(estimate_distance_m(sliver, 640, 480, PRIORITY))

    def test_unknown_classes_use_the_default_height(self) -> None:
        self.assertEqual(PRIORITY.height_for("unicorn"), PRIORITY.default_height_m)


class NearRangeHasPriorityTests(unittest.TestCase):
    def test_a_close_range_hit_wins_over_a_far_camera_detection(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        fusion.update_sensor(tof("tof-left", Sector.LEFT, 0.4, now))
        for i in range(3):
            fusion.update_vision(frame(now + i, person(Sector.RIGHT, 3.0)))
        decision = fusion.evaluate(now + 5 * MS)
        self.assertEqual((decision.level, decision.source), (HazardLevel.CRITICAL, "range"))
        self.assertEqual(decision.sectors, (Sector.LEFT,))

    def test_the_forward_ultrasonic_counts_as_short_range(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        fusion.update_sensor(us("us-1", 0.7, now))
        decision = fusion.evaluate(now + MS)
        self.assertEqual((decision.level, decision.source), (HazardLevel.WARNING, "range"))

    def test_the_camera_cannot_override_clear_range_sensors_in_the_near_band(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        for i in range(3):
            fusion.update_vision(frame(now + i, person(Sector.CENTER, 0.8)))
        self.assertEqual(fusion.evaluate(now + 5 * MS).level, HazardLevel.CLEAR)


class FarCameraTests(unittest.TestCase):
    def test_a_confirmed_far_detection_pulses_its_own_sector(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        fusion.update_vision(frame(now, person(Sector.RIGHT, 3.0)))
        fusion.update_vision(frame(now + 30 * MS, person(Sector.RIGHT, 3.0)))
        decision = fusion.evaluate(now + 40 * MS)
        self.assertEqual((decision.level, decision.source), (HazardLevel.WARNING, "vision"))
        self.assertEqual(decision.sectors, (Sector.RIGHT,))
        self.assertEqual(decision.semantic_labels, ("person",))

    def test_one_frame_is_not_enough(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        fusion.update_vision(frame(now, person(Sector.LEFT, 3.0)))
        self.assertEqual(fusion.evaluate(now + MS).level, HazardLevel.CLEAR)

    def test_a_detection_that_disappears_resets_the_confirmation(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        fusion.update_vision(frame(now, person(Sector.LEFT, 3.0)))
        fusion.update_vision(frame(now + 10 * MS))
        fusion.update_vision(frame(now + 20 * MS, person(Sector.LEFT, 3.0)))
        self.assertEqual(fusion.evaluate(now + 25 * MS).level, HazardLevel.CLEAR)

    def test_low_confidence_and_beyond_far_max_are_ignored(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        for i in range(3):
            fusion.update_vision(frame(now + i * MS, person(Sector.CENTER, 3.0, confidence=0.4),
                                       person(Sector.LEFT, 6.0)))
        self.assertEqual(fusion.evaluate(now + 5 * MS).level, HazardLevel.CLEAR)

    def test_the_nearest_detection_decides_the_sector(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        for i in range(3):
            fusion.update_vision(frame(now + i * MS, person(Sector.LEFT, 3.8), person(Sector.RIGHT, 2.2)))
        decision = fusion.evaluate(now + 5 * MS)
        self.assertEqual(decision.sectors, (Sector.RIGHT,))

    def test_a_stale_frame_is_ignored(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        for i in range(3):
            fusion.update_vision(frame(now + i, person(Sector.CENTER, 3.0)))
        later = now + 600 * MS
        clear_ranges(fusion, later)
        self.assertEqual(fusion.evaluate(later).level, HazardLevel.CLEAR)

    def test_far_alerts_are_softer_and_slower_than_range_alerts(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        for i in range(3):
            fusion.update_vision(frame(now + i * MS, person(Sector.CENTER, 3.0)))
        haptics = CaneConfig().haptics
        far = command_for(fusion.evaluate(now + 5 * MS), PRIORITY, FusionConfig(), haptics)
        fusion.update_sensor(tof("tof-center", Sector.CENTER, 0.3, now + 6 * MS))
        near = command_for(fusion.evaluate(now + 7 * MS), PRIORITY, FusionConfig(), haptics)
        self.assertEqual((far.source, near.source), ("vision", "range"))
        self.assertLess(far.intensity, near.intensity)
        self.assertGreater(cycle_length(build_cycle(far, haptics)), 1.5 * cycle_length(build_cycle(near, haptics)))

    def test_when_range_sensors_are_dead_the_camera_also_covers_the_near_band(self) -> None:
        fusion, now = make(), 10 * MS
        for i in range(3):
            fusion.update_vision(frame(now + i * MS, person(Sector.CENTER, 0.8)))
        decision = fusion.evaluate(now + 5 * MS)
        self.assertEqual((decision.level, decision.source), (HazardLevel.WARNING, "vision"))
        command = command_for(decision, PRIORITY, FusionConfig(), CaneConfig().haptics)
        self.assertGreater(command.intensity, PRIORITY.far_intensity)  # a range-style alert, not the gentle one

    def test_dead_ranges_and_nothing_seen_is_a_fault(self) -> None:
        decision = make().evaluate(10 * MS)
        self.assertEqual((decision.level, decision.source), (HazardLevel.SENSOR_FAULT, "fault"))


class OverheadTests(unittest.TestCase):
    def test_the_overhead_ultrasonic_pulses_the_centre_motor_only(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        fusion.update_sensor(us("us-2", 0.8, now))
        decision = fusion.evaluate(now + MS)
        self.assertEqual((decision.level, decision.source), (HazardLevel.WARNING, "overhead"))
        self.assertEqual(decision.sectors, (Sector.CENTER,))

    def test_overhead_readings_never_count_as_forward_ranges(self) -> None:
        fusion, now = make(), 10 * MS
        fusion.update_sensor(us("us-2", 3.0, now))  # only the overhead sensor is alive
        self.assertEqual(fusion.evaluate(now + MS).level, HazardLevel.SENSOR_FAULT)

    def test_a_clear_overhead_sensor_is_silent(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        self.assertEqual(fusion.evaluate(now + MS).level, HazardLevel.CLEAR)

    def test_a_more_severe_forward_hazard_wins_over_overhead(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        fusion.update_sensor(us("us-2", 0.8, now))
        fusion.update_sensor(tof("tof-left", Sector.LEFT, 0.3, now))
        decision = fusion.evaluate(now + MS)
        self.assertEqual((decision.level, decision.sectors), (HazardLevel.CRITICAL, (Sector.LEFT,)))

    def test_equal_severity_merges_the_sectors(self) -> None:
        fusion, now = make(), 10 * MS
        clear_ranges(fusion, now)
        fusion.update_sensor(us("us-2", 0.8, now))
        fusion.update_sensor(tof("tof-left", Sector.LEFT, 0.8, now))
        decision = fusion.evaluate(now + MS)
        self.assertEqual(decision.sectors, (Sector.CENTER, Sector.LEFT))


class PriorityConfigTests(unittest.TestCase):
    def test_roles_and_priority_load_from_json(self) -> None:
        config = config_from_dict({
            "ultrasonic": {"sensors": [
                {"id": "a", "trig_gpio": 23, "echo_gpio": 24, "sector": "center", "role": "forward"},
                {"id": "b", "trig_gpio": 27, "echo_gpio": 22, "sector": "center", "role": "overhead"},
            ]},
            "priority": {"far_max_m": 3.5, "class_heights_m": {"person": 1.8}},
        })
        self.assertEqual([s.role for s in config.ultrasonic.sensors], ["forward", "overhead"])
        self.assertEqual(config.priority.height_for("person"), 1.8)
        self.assertEqual(config.priority.height_for("dog"), 0.5)

    def test_bad_values_are_rejected(self) -> None:
        for payload in (
            {"priority": {"far_max_m": 0.9}},
            {"priority": {"confirm_frames": 0}},
            {"priority": {"typo": 1}},
            {"imu": {"jolt_angle_deg": 1}},
            {"ultrasonic": {"sensors": [
                {"id": "a", "trig_gpio": 23, "echo_gpio": 24, "sector": "center", "role": "up"}]}},
            {"ultrasonic": {"sensors": [
                {"id": "a", "trig_gpio": 23, "echo_gpio": 24, "sector": "center", "role": "overhead"},
                {"id": "b", "trig_gpio": 27, "echo_gpio": 22, "sector": "center", "role": "overhead"}]}},
        ):
            with self.assertRaises(ConfigError, msg=payload):
                config_from_dict(payload)

    def test_the_shipped_config_marks_us_2_as_ground(self) -> None:
        from yolo_pi.cane.config import load_config
        config = load_config("configs/cane-hardware.json")
        self.assertEqual({s.sensor_id: s.role for s in config.ultrasonic.sensors},
                         {"us-1": "forward", "us-2": "ground"})

    def test_single_ping_ultrasonic_ghost_spike_is_rejected(self) -> None:
        fusion = PriorityFusion(FusionConfig(), PRIORITY)
        for index in range(6):
            ts = (index + 1) * 33 * MS
            spike = 0.32 if index == 3 else 1.50
            fusion.update_sensors([
                tof("tof-left", Sector.LEFT, 1.50, ts),
                tof("tof-center", Sector.CENTER, 1.50, ts),
                tof("tof-right", Sector.RIGHT, 1.50, ts),
                us("us-1", spike, ts),
                us("us-2", 1.50, ts),
            ])
            decision = fusion.evaluate(ts)
            self.assertEqual(decision.level, HazardLevel.CLEAR)

    def test_downward_tilt_suppresses_floor_echo_on_ground_angled_us2(self) -> None:
        from yolo_pi.imu import IMUSample

        fusion = PriorityFusion(
            FusionConfig(),
            PRIORITY,
            ground_ids=("us-2",),
            sensor_pitch_offsets_deg={"us-2": 8.0},
        )
        g = 9.80665
        pitch_deg = 42.0
        rad = math.radians(pitch_deg)
        for index in range(40):
            ts = (index + 1) * 20 * MS
            fusion.update_imu(
                IMUSample(ts, -g * math.sin(rad), 0.0, -g * math.cos(rad), 0.0, 0.0, 0.0)
            )
        ts = 900 * MS
        floor_us1 = 0.75 / math.sin(math.radians(pitch_deg))
        floor_us2 = 0.75 / math.sin(math.radians(pitch_deg + 8.0))
        fusion.update_sensors([
            tof("tof-center", Sector.CENTER, floor_us1, ts),
            us("us-1", floor_us1, ts),
            us("us-2", floor_us2, ts),
        ])
        self.assertEqual(fusion.evaluate(ts).level, HazardLevel.CLEAR)

    def test_fast_approaching_target_warns_early_via_ttc(self) -> None:
        fusion = PriorityFusion(FusionConfig(), PRIORITY)
        for index, dist in enumerate((1.45, 1.35, 1.25)):
            ts = (index + 1) * 50 * MS
            fusion.update_sensor(tof("tof-center", Sector.CENTER, dist, ts))
        decision = fusion.evaluate(160 * MS)
        self.assertEqual(decision.level, HazardLevel.WARNING)
        self.assertIn("TTC", decision.reason)


if __name__ == "__main__":
    unittest.main()
