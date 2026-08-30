import unittest

from yolo_pi.types import BoundingBox, Detection, FrameDetections, Sector


class BoundingBoxTests(unittest.TestCase):
    def test_geometry(self) -> None:
        box = BoundingBox(10.0, 20.0, 30.0, 60.0)
        self.assertEqual(box.center_x, 20.0)
        self.assertEqual(box.center_y, 40.0)
        self.assertEqual(box.area, 800.0)

    def test_rejects_inverted_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(10.0, 20.0, 9.0, 60.0)


class DetectionTests(unittest.TestCase):
    def test_confidence_range_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            Detection(0, "person", 1.01, BoundingBox(0, 0, 1, 1))

    def test_frame_timing_and_serialization(self) -> None:
        detection = Detection(
            0,
            "person",
            0.9,
            BoundingBox(0, 0, 100, 100),
            Sector.CENTER,
        )
        frame = FrameDetections(
            frame_id=7,
            captured_ns=1_000_000,
            model_started_ns=2_000_000,
            model_finished_ns=5_000_000,
            width=640,
            height=480,
            detections=[detection],
            backend="fake",
            model_id="fixture",
        )
        self.assertEqual(frame.model_call_ms, 3.0)
        self.assertEqual(frame.frame_age_at_model_finish_ms, 4.0)
        payload = frame.as_dict()
        self.assertEqual(payload["detections"][0]["sector"], "center")
        self.assertEqual(payload["model_id"], "fixture")


if __name__ == "__main__":
    unittest.main()
