import unittest

from yolo_pi.fusion import (
    FusionConfig,
    FusionEngine,
    HazardLevel,
    RecordingHapticSink,
    SensorKind,
    SensorReading,
    decision_to_haptic,
)
from yolo_pi.types import BoundingBox, Detection, FrameDetections, Sector


MS = 1_000_000


def range_reading(
    sensor_id: str,
    distance_m: float,
    timestamp_ns: int,
    sector: Sector = Sector.CENTER,
    valid: bool = True,
) -> SensorReading:
    return SensorReading(
        sensor_id=sensor_id,
        kind=SensorKind.TOF,
        timestamp_ns=timestamp_ns,
        sector=sector,
        distance_m=distance_m,
        valid=valid,
    )


class FusionTests(unittest.TestCase):
    def test_close_range_warns_without_vision(self) -> None:
        engine = FusionEngine()
        engine.update_sensor(range_reading("tof-center", 0.45, 100 * MS))
        decision = engine.evaluate(101 * MS)
        self.assertEqual(decision.level, HazardLevel.CRITICAL)
        self.assertTrue(decision.range_triggered)
        self.assertEqual(decision.semantic_labels, tuple())

    def test_nearest_sensor_controls_severity_and_sector(self) -> None:
        engine = FusionEngine()
        engine.update_sensors(
            [
                range_reading("tof-left", 1.8, 100 * MS, Sector.LEFT),
                range_reading("tof-right", 0.8, 100 * MS, Sector.RIGHT),
            ]
        )
        decision = engine.evaluate(120 * MS)
        self.assertEqual(decision.level, HazardLevel.WARNING)
        self.assertEqual(decision.sectors, (Sector.RIGHT,))
        self.assertEqual(decision.nearest_distance_m, 0.8)

    def test_stale_or_invalid_ranges_produce_sensor_fault(self) -> None:
        engine = FusionEngine(FusionConfig(sensor_stale_after_ms=100))
        engine.update_sensors(
            [
                range_reading("old", 0.2, 1 * MS),
                range_reading("invalid", -1.0, 150 * MS),
            ]
        )
        decision = engine.evaluate(200 * MS)
        self.assertEqual(decision.level, HazardLevel.SENSOR_FAULT)
        self.assertFalse(decision.range_triggered)

    def test_vision_enriches_but_does_not_create_range_warning(self) -> None:
        engine = FusionEngine()
        vision = FrameDetections(
            frame_id=1,
            captured_ns=90 * MS,
            model_started_ns=95 * MS,
            model_finished_ns=100 * MS,
            width=300,
            height=300,
            detections=[
                Detection(
                    0,
                    "person",
                    0.9,
                    BoundingBox(100, 0, 200, 250),
                    Sector.CENTER,
                )
            ],
        )
        engine.update_vision(vision)
        engine.update_sensor(range_reading("far", 2.0, 100 * MS))
        clear = engine.evaluate(110 * MS)
        self.assertEqual(clear.level, HazardLevel.CLEAR)
        self.assertFalse(clear.range_triggered)

        engine.update_sensor(range_reading("near", 0.7, 120 * MS))
        warning = engine.evaluate(125 * MS)
        self.assertEqual(warning.level, HazardLevel.WARNING)
        self.assertEqual(warning.semantic_labels, ("person",))

    def test_haptic_sink_records_mapped_command(self) -> None:
        engine = FusionEngine()
        engine.update_sensor(range_reading("near", 0.4, 10 * MS, Sector.LEFT))
        decision = engine.evaluate(11 * MS)
        sink = RecordingHapticSink()
        sink.emit(decision_to_haptic(decision))
        self.assertEqual(len(sink.commands), 1)
        self.assertEqual(sink.commands[0].intensity, 1.0)
        self.assertEqual(sink.commands[0].sectors, (Sector.LEFT,))


if __name__ == "__main__":
    unittest.main()
