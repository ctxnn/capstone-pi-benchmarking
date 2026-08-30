import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

from yolo_pi.benchmark import BenchmarkInput, percentile, run_benchmark
from yolo_pi.types import FrameDetections


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(
        self,
        frame: Any,
        frame_id: int,
        captured_ns: Optional[int] = None,
        source: Optional[str] = None,
    ) -> FrameDetections:
        self.calls += 1
        captured_ns = 0 if captured_ns is None else captured_ns
        duration_ns = self.calls * 1_000_000
        return FrameDetections(
            frame_id=frame_id,
            captured_ns=captured_ns,
            model_started_ns=captured_ns + 100_000,
            model_finished_ns=captured_ns + 100_000 + duration_ns,
            width=10,
            height=10,
            backend="fake",
            model_id="fixture-v1",
            stage_ms={"inference": float(self.calls)},
            source=source,
        )


class BenchmarkTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([0.0, 10.0], 0.5), 5.0)
        self.assertEqual(percentile([1.0], 0.99), 1.0)

    def test_warmups_are_excluded_and_report_is_written(self) -> None:
        detector = FakeDetector()
        report = run_benchmark(
            detector,
            [BenchmarkInput("a", "a.jpg"), BenchmarkInput("b", "b.jpg")],
            warmup_runs=2,
            repetitions=2,
        )
        self.assertEqual(detector.calls, 6)
        self.assertEqual(len(report.observations), 4)
        self.assertEqual(report.observations[0].model_call_ms, 3.0)
        payload = report.as_dict()
        self.assertEqual(payload["warmup_runs_excluded"], 2)
        self.assertEqual(payload["model_call_ms"]["count"], 4.0)
        self.assertIn("not Raspberry Pi evidence", payload["scope"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"
            report.write_json(path)
            self.assertTrue(path.exists())
            self.assertIn('"schema_version": 1', path.read_text(encoding="utf-8"))

    def test_rejects_empty_inputs(self) -> None:
        with self.assertRaises(ValueError):
            run_benchmark(FakeDetector(), [])


if __name__ == "__main__":
    unittest.main()
