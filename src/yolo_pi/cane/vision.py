"""Camera and detector thread that feeds detections to the fusion engine.

Close in (inside the warning distance) the range sensors decide and a detection
only names what they found. Farther out (up to ``priority.far_max_m``) a
confirmed detection raises a gentle alert itself; see ``priority.py``. Either
way, a camera or model failure here must never stop the cane. This worker
catches every error, logs it, backs off, and tries again.

While the IMU says the cane is resting, the worker drops to about one frame per
second instead of running the detector flat out, which saves battery.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

from ..types import FrameDetections
from .config import VisionConfig

LOGGER = logging.getLogger("yolo_pi.cane.vision")


class VisionSink(Protocol):
    def update_vision(self, detections: FrameDetections) -> None: ...


def model_input_size(model_path: Path, default: int = 416) -> int:
    """Read the export size from an OpenVINO ``metadata.yaml`` (no YAML library needed)."""

    metadata = model_path / "metadata.yaml"
    if metadata.is_file():
        match = re.search(r"imgsz:\s*\n\s*-\s*(\d+)", metadata.read_text(encoding="utf-8"))
        if match:
            return int(match.group(1))
    return default


def create_camera(config: VisionConfig) -> Any:  # pragma: no cover - needs a Pi
    from ..picamera2_source import Picamera2LatestFrameSource

    return Picamera2LatestFrameSource(
        width=config.width,
        height=config.height,
        fps=config.camera_fps,
        camera_num=config.camera_num,
    )


def create_detector(config: VisionConfig) -> Any:  # pragma: no cover - needs a Pi
    from ..pi_benchmark import configure_threads

    model = Path(config.model)
    if not model.exists():
        raise FileNotFoundError(f"vision model not found: {model}")
    configure_threads(config.threads)
    resolved_imgsz = config.imgsz or model_input_size(model)
    is_openvino = (
        model.is_dir() and (model / "best.xml").is_file()
    ) or model.suffix.lower() == ".xml" or model.name.endswith("_openvino_model")
    if is_openvino:
        try:
            from ..openvino_detector import OpenVINODetector

            return OpenVINODetector(
                model_path=model,
                imgsz=resolved_imgsz,
                confidence=config.confidence,
                center_fraction=config.center_fraction,
                num_threads=config.threads,
            )
        except Exception as exc:
            LOGGER.warning("native OpenVINODetector unavailable (%s); falling back to Ultralytics", exc)

    from ..detector import UltralyticsDetector

    return UltralyticsDetector(
        model_path=model,
        imgsz=resolved_imgsz,
        confidence=config.confidence,
        center_fraction=config.center_fraction,
        runtime_threads=config.threads,
    )


class VisionWorker(threading.Thread):
    def __init__(
        self,
        config: VisionConfig,
        sink: VisionSink,
        is_still: Callable[[], bool] = lambda: False,
        camera_factory: Callable[[], Any] = None,  # type: ignore[assignment]
        detector_factory: Callable[[], Any] = None,  # type: ignore[assignment]
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(name="vision", daemon=True)
        self._config = config
        self._sink = sink
        self._is_still = is_still
        self._camera_factory = camera_factory or (lambda: create_camera(config))
        self._detector_factory = detector_factory or (lambda: create_detector(config))
        self._clock = clock
        self._halt = threading.Event()
        self.ready = threading.Event()
        self.frames = 0
        self.restarts = 0
        self.last_error: Optional[str] = None
        self._model_ms_total = 0.0
        self._last_model_ms = 0.0

    def stop(self) -> None:
        self._halt.set()

    def stats(self) -> Dict[str, object]:
        return {
            "frames": self.frames,
            "restarts": self.restarts,
            "mean_model_ms": round(self._model_ms_total / self.frames, 1) if self.frames else None,
            "last_model_ms": round(self._last_model_ms, 1),
            "idle": bool(self._config.idle_when_still and self._is_still()),
            "last_error": self.last_error,
        }

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                self._run_once()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.error("vision stopped: %s; retrying in %.0f s",
                             self.last_error, self._config.restart_backoff_s)
            self.ready.clear()
            if self._halt.wait(self._config.restart_backoff_s):
                break
            self.restarts += 1

    def _run_once(self) -> None:
        detector = self._detector_factory()
        camera = self._camera_factory()
        camera.start()
        try:
            self.ready.set()
            while not self._halt.is_set():
                if self._config.idle_when_still and self._is_still():
                    if self._halt.wait(self._config.idle_interval_s):
                        return
                frame = camera.next_frame(timeout_s=5.0)
                result = detector.detect(
                    frame.payload,
                    frame.frame_id,
                    captured_ns=frame.captured_ns,
                    source=frame.source,
                )
                self._sink.update_vision(result)
                self.frames += 1
                self._last_model_ms = result.model_call_ms
                self._model_ms_total += result.model_call_ms
        finally:
            try:
                camera.close()
            except Exception:
                pass
