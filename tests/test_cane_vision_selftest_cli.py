import io
import json
import signal
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from yolo_pi.cane.cli import main
from yolo_pi.cane.config import CaneConfig, VisionConfig
from yolo_pi.cane.selftest import run_selftest
from yolo_pi.cane.sim import SimClock, build_simulated_hardware
from yolo_pi.cane.vision import VisionWorker, model_input_size
from yolo_pi.types import FrameDetections

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Frame:
    frame_id: int
    payload: object
    captured_ns: int
    source: str = "fake"


class FakeCamera:
    def __init__(self) -> None:
        self.count = 0
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def next_frame(self, timeout_s=10.0):
        time.sleep(0.01)
        self.count += 1
        return Frame(self.count, object(), time.perf_counter_ns())

    def close(self):
        self.closed = True


class FakeDetector:
    def detect(self, payload, frame_id, captured_ns=None, source=None):
        now = time.perf_counter_ns()
        return FrameDetections(frame_id, captured_ns or now, now, now + 2_000_000, 640, 480)


class VisionSink:
    def __init__(self) -> None:
        self.results = []

    def update_vision(self, detections) -> None:
        self.results.append(detections)


def make_worker(sink, **kwargs):
    config = VisionConfig(restart_backoff_s=0.05, idle_interval_s=0.2, **kwargs.pop("config", {}))
    camera = kwargs.pop("camera", None) or FakeCamera()
    detector_factory = kwargs.pop("detector_factory", FakeDetector)
    return VisionWorker(config, sink, camera_factory=lambda: camera,
                        detector_factory=detector_factory, **kwargs), camera


class VisionTests(unittest.TestCase):
    def test_frames_flow_to_the_sink_and_the_camera_is_released(self) -> None:
        sink = VisionSink()
        worker, camera = make_worker(sink)
        worker.start()
        time.sleep(0.3)
        worker.stop()
        worker.join(timeout=3)
        self.assertGreater(len(sink.results), 5)
        self.assertTrue(camera.started and camera.closed)
        self.assertEqual(worker.stats()["frames"], len(sink.results))

    def test_the_detector_idles_while_the_cane_is_still(self) -> None:
        busy_sink, idle_sink = VisionSink(), VisionSink()
        busy, _ = make_worker(busy_sink)
        idle, _ = make_worker(idle_sink, is_still=lambda: True)
        for worker in (busy, idle):
            worker.start()
        time.sleep(0.6)
        for worker in (busy, idle):
            worker.stop()
            worker.join(timeout=3)
        self.assertLess(len(idle_sink.results), len(busy_sink.results) / 3)
        self.assertGreater(len(idle_sink.results), 0)

    def test_idle_can_be_disabled(self) -> None:
        sink = VisionSink()
        worker, _ = make_worker(sink, config={"idle_when_still": False}, is_still=lambda: True)
        worker.start()
        time.sleep(0.3)
        worker.stop()
        worker.join(timeout=3)
        self.assertGreater(len(sink.results), 5)

    def test_a_detector_failure_is_survived_and_retried(self) -> None:
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise FileNotFoundError("model missing")
            return FakeDetector()

        sink = VisionSink()
        worker, _ = make_worker(sink, detector_factory=flaky)
        worker.start()
        time.sleep(0.5)
        worker.stop()
        worker.join(timeout=3)
        self.assertGreaterEqual(worker.restarts, 1)
        self.assertGreater(len(sink.results), 0)
        self.assertIn("FileNotFoundError", worker.last_error)

    def test_model_input_size_is_read_from_the_exported_metadata(self) -> None:
        base = ROOT / "artifacts" / "models" / "cane-v1-fast"
        for size in (416, 320):
            model = base / f"ov-fp32-{size}" / "best_openvino_model"
            if not (model / "metadata.yaml").is_file():
                self.skipTest("model artifacts not present")
            self.assertEqual(model_input_size(model), size)
        self.assertEqual(model_input_size(Path("/nonexistent")), 416)


class SelftestTests(unittest.TestCase):
    def run_checks(self, hardware, **kwargs):
        return run_selftest(CaneConfig(), hardware, duration_s=0.3, ultrasonic_rounds=3,
                            sleep=lambda _s: None, out=None, **kwargs)

    def test_simulated_cane_passes_with_no_failures(self) -> None:
        checks = self.run_checks(build_simulated_hardware(CaneConfig(), clock=SimClock()))
        by_name = {c.name: c.status for c in checks}
        self.assertNotIn("FAIL", by_name.values())
        self.assertEqual(by_name["tof tof-left"], "PASS")
        self.assertEqual(by_name["haptic right"], "PASS")
        self.assertEqual(by_name["haptic overhead"], "PASS")
        self.assertEqual(by_name["imu gyro bias"], "PASS")
        self.assertEqual(by_name["ultrasonic us-1"], "WARN")  # empty room: no echo to prove the wiring

    def test_a_sensor_that_never_reports_is_a_failure(self) -> None:
        hardware = build_simulated_hardware(CaneConfig(), clock=SimClock())

        class Silent:
            def poll(self):
                return None

        hardware.tof["tof-center"] = Silent()
        by_name = {c.name: c.status for c in self.run_checks(hardware)}
        self.assertEqual(by_name["tof tof-center"], "FAIL")
        self.assertEqual(by_name["tof tof-left"], "PASS")

    def test_startup_errors_and_a_missing_motor_are_failures(self) -> None:
        hardware = build_simulated_hardware(CaneConfig(), clock=SimClock())
        hardware.errors.append("DRV2605L (main bus): no device")
        hardware.haptic = None
        checks = self.run_checks(hardware)
        statuses = {c.name: c.status for c in checks}
        self.assertEqual(statuses["haptic driver"], "FAIL")
        self.assertTrue(any(c.name == "startup" and c.status == "FAIL" for c in checks))

    def test_a_motor_write_error_is_reported_not_raised(self) -> None:
        hardware = build_simulated_hardware(CaneConfig(), clock=SimClock())

        def boom(_value):
            raise OSError("bus")

        hardware.haptic.set_rtp = boom
        statuses = {c.name: c.status for c in self.run_checks(hardware)}
        self.assertEqual(statuses["haptic left"], "FAIL")


class CliTests(unittest.TestCase):
    def write_config(self, tmp, extra=None):
        payload = {"runtime": {"event_log": None}, "haptics": {"startup_sweep": False}}
        payload.update(extra or {})
        path = Path(tmp) / "cane.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_show_config_prints_the_wiring(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["show-config"])
        self.assertEqual(code, 0)
        described = json.loads(buffer.getvalue())
        self.assertEqual(described["mux_address"], "0x70")
        self.assertEqual(described["haptics"]["taps"], {"left": 1, "center": 2, "right": 3})

    def test_a_bad_config_exits_2_with_a_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(tmp, {"tof": {"typo_key": 1}})
            self.assertEqual(main(["--config", path, "show-config"]), 2)

    def test_simulated_run_exits_zero(self) -> None:
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = self.write_config(tmp)
                self.assertEqual(
                    main(["--config", path, "run", "--simulate", "--no-vision", "--duration", "0.4"]), 0)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)

    def test_simulated_selftest_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(tmp)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(["--config", path, "selftest", "--simulate", "--seconds", "0.3"])
            self.assertEqual(code, 0)
            self.assertIn("checks:", buffer.getvalue())

    def test_real_hardware_commands_fail_cleanly_without_a_pi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(tmp)
            self.assertEqual(main(["--config", path, "scan"]), 3)


if __name__ == "__main__":
    unittest.main()
