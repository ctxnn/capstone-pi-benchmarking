from __future__ import annotations

import threading
from dataclasses import replace
from time import perf_counter_ns
from typing import Any, Callable, Optional

from .input_sources import BenchmarkFrame
from .pipeline import FramePacket, LatestFrameBuffer


class Picamera2LatestFrameSource:
    """IMX219 source with one-slot overwrite semantics and no stale-frame queue."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        camera_num: int = 0,
        pixel_format: str = "RGB888",
        _camera_factory: Optional[Callable[[int], Any]] = None,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("camera width, height, and fps must be positive")
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_num = camera_num
        self.pixel_format = pixel_format
        self._camera_factory = _camera_factory
        self._camera: Any = None
        self._buffer = LatestFrameBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._last_frame_id: Optional[int] = None
        self._last_overwritten = 0

    def _new_camera(self) -> Any:
        if self._camera_factory is not None:
            return self._camera_factory(self.camera_num)
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is required only for --source camera on Raspberry Pi"
            ) from exc
        return Picamera2(self.camera_num)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._camera = self._new_camera()
        try:
            configuration = self._camera.create_video_configuration(
                main={"size": (self.width, self.height), "format": self.pixel_format},
                controls={"FrameRate": self.fps},
                buffer_count=3,
            )
            self._camera.configure(configuration)
            self._camera.start()
        except BaseException:
            self.close()
            raise
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="picamera2-latest-frame",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        frame_id = 0
        try:
            while not self._stop.is_set():
                capture_started_ns = perf_counter_ns()
                request = self._camera.capture_request(flush=True)
                try:
                    metadata = request.get_metadata()
                    payload = request.make_array("main")
                finally:
                    request.release()
                available_ns = perf_counter_ns()
                sensor_timestamp = metadata.get("SensorTimestamp")
                captured_ns = int(sensor_timestamp) if sensor_timestamp is not None else capture_started_ns
                # libcamera normally uses a compatible monotonic clock. If a
                # platform reports another clock domain, retain honest local timing.
                age_ns = available_ns - captured_ns
                if age_ns < 0 or age_ns > 10_000_000_000:
                    captured_ns = capture_started_ns
                frame = BenchmarkFrame(
                    frame_id=frame_id,
                    payload=payload,
                    source=f"picamera2:{self.camera_num}",
                    source_type="camera",
                    captured_ns=captured_ns,
                    capture_started_ns=capture_started_ns,
                    available_ns=available_ns,
                )
                self._buffer.publish(
                    FramePacket(
                        frame_id=frame_id,
                        captured_ns=captured_ns,
                        payload=frame,
                        source=frame.source,
                    )
                )
                frame_id += 1
        except BaseException as exc:
            self._error = exc
        finally:
            self._buffer.close()

    def next_frame(self, timeout_s: float = 10.0) -> BenchmarkFrame:
        packet = self._buffer.wait_for_next(self._last_frame_id, timeout_s=timeout_s)
        if packet is None:
            if self._error is not None:
                raise RuntimeError("Picamera2 capture loop failed") from self._error
            raise TimeoutError(f"no fresh camera frame within {timeout_s} seconds")
        self._last_frame_id = packet.frame_id
        overwritten = self._buffer.overwritten_count
        dropped = max(0, overwritten - self._last_overwritten)
        self._last_overwritten = overwritten
        return replace(packet.payload, dropped_since_last=dropped)

    def close(self) -> None:
        self._stop.set()
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._buffer.close()
        if self._camera is not None:
            # stop() leaves libcamera acquired in Configured state. Release it
            # before the next benchmark row creates another Picamera2 instance.
            self._camera.close()
            self._camera = None

    def __enter__(self) -> "Picamera2LatestFrameSource":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
