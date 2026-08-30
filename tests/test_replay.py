import tempfile
import unittest
from pathlib import Path

from yolo_pi.fusion import FusionEngine, HazardLevel
from yolo_pi.replay import load_sensor_jsonl, replay_readings, write_decisions_jsonl


class ReplayTests(unittest.TestCase):
    def test_jsonl_round_trip(self) -> None:
        content = "\n".join(
            [
                '{"sensor_id":"left","kind":"tof","timestamp_ns":1000000,"sector":"left","distance_m":2.0}',
                '{"sensor_id":"left","kind":"tof","timestamp_ns":2000000,"sector":"left","distance_m":0.4}',
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "readings.jsonl"
            output_path = Path(directory) / "decisions.jsonl"
            input_path.write_text(content + "\n", encoding="utf-8")
            decisions = replay_readings(load_sensor_jsonl(input_path), FusionEngine())
            self.assertEqual(decisions[0].level, HazardLevel.CLEAR)
            self.assertEqual(decisions[1].level, HazardLevel.CRITICAL)
            write_decisions_jsonl(decisions, output_path)
            output = output_path.read_text(encoding="utf-8")
            self.assertIn('"level": "critical"', output)

    def test_rejects_non_monotonic_trace(self) -> None:
        content = "\n".join(
            [
                '{"sensor_id":"x","kind":"tof","timestamp_ns":2,"sector":"center","distance_m":1.0}',
                '{"sensor_id":"x","kind":"tof","timestamp_ns":1,"sector":"center","distance_m":1.0}',
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(content + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                list(load_sensor_jsonl(path))


if __name__ == "__main__":
    unittest.main()
