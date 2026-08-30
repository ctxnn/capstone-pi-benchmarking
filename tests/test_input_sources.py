import time
import unittest
from pathlib import Path
from time import perf_counter_ns

from yolo_pi.input_sources import ImageSequenceSource
from yolo_pi.picamera2_source import Picamera2LatestFrameSource


class FakeRequest:
    def __init__(self, value: int) -> None:
        self.value = value
        self.timestamp = perf_counter_ns()

    def get_metadata(self):
        return {"SensorTimestamp": self.timestamp}

    def make_array(self, stream: str):
        self.stream = stream
        return [[self.value]]

    def release(self) -> None:
        return None


class FakeCamera:
    def __init__(self, camera_num: int) -> None:
        self.camera_num = camera_num
        self.running = False
        self.value = 0

    def create_video_configuration(self, **kwargs):
        self.configuration_kwargs = kwargs
        return kwargs

    def configure(self, configuration) -> None:
        self.configuration = configuration

    def start(self) -> None:
        self.running = True

    def capture_request(self, flush: bool = False):
        if not self.running:
            raise RuntimeError("camera stopped")
        self.value += 1
        time.sleep(0.002)
        return FakeRequest(self.value)

    def stop(self) -> None:
        self.running = False


class InputSourceTests(unittest.TestCase):
    def test_saved_images_cycle_without_camera_timing(self) -> None:
        source = ImageSequenceSource([Path("one.jpg"), Path("two.jpg")])
        source.start()
        first = source.next_frame()
        source.next_frame()
        third = source.next_frame()
        self.assertEqual(first.payload, Path("one.jpg"))
        self.assertEqual(third.payload, Path("one.jpg"))
        self.assertIsNone(first.capture_ms)
        self.assertEqual(first.source_type, "images")

    def test_picamera_adapter_returns_fresh_latest_frames(self) -> None:
        source = Picamera2LatestFrameSource(
            width=320,
            height=240,
            fps=30,
            _camera_factory=FakeCamera,
        )
        source.start()
        first = source.next_frame(timeout_s=1)
        time.sleep(0.015)
        second = source.next_frame(timeout_s=1)
        source.close()
        self.assertEqual(first.source_type, "camera")
        self.assertGreater(second.frame_id, first.frame_id)
        self.assertGreaterEqual(second.dropped_since_last, 1)
        self.assertIsNotNone(second.capture_ms)
        self.assertIsNotNone(second.frame_age_at_available_ms)


if __name__ == "__main__":
    unittest.main()
