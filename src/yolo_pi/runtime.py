"""Live multi-threaded runtime for the SmartCane.

Three concerns run on separate threads so none of them can hold up another:

1. A deadline-driven fast control loop (default 50 Hz) that polls the three
   VL53L0X time-of-flight sensors and the MPU-6050, folds both into the fusion
   engine, and emits a directional haptic frame every tick.
2. An alternating HC-SR04 thread that pings one ultrasonic sensor at a time in
   the background. An ultrasonic echo wait costs tens of milliseconds, which
   would destroy a 20 ms control period if it ran inside the control loop.
3. A camera capture thread plus a vision inference thread. The capture thread
   keeps only the newest frame, so a slow model never builds a queue of old
   images, and the vision thread drops frames that are already too old to be
   worth inferring on.

Every driver is duck-typed: the same runtime drives the real Pi 5 drivers and
the in-memory simulation used for soak tests on a laptop.
"""

from __future__ import annotations

import json
import logging
import math
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

from .fusion import SensorKind
from .fusion_v2 import (
    DirectionalHapticFrame,
    DirectionalHapticSink,
    ExtendedHazardLevel,
    RangeReading,
    SmartCaneFusionConfig,
    SmartCaneFusionEngine,
    decision_to_directional_haptics,
)
from .imu import IMUSample
from .types import BoundingBox, Detection, FrameDetections, Sector

_SECTORS = (Sector.LEFT, Sector.CENTER, Sector.RIGHT)
_DEFAULT_WINDOW = 512
_HARDWARE_MAX_RANGE_M = 2.0
_ULTRASONIC_MAX_RANGE_M = 3.5


class ToFArrayLike(Protocol):
    def read_all(self, now_ns: Optional[int] = None) -> Sequence[Any]:
        """Return one measurement per time-of-flight sensor."""

    def close(self) -> None:
        """Release the bus and sensors."""


class UltrasonicDriverLike(Protocol):
    def read_next(self, now_ns: Optional[int] = None) -> Any:
        """Ping the next sensor in rotation and return its measurement."""

    def close(self) -> None:
        """Release the GPIO pins."""


class IMUDriverLike(Protocol):
    def read_raw(self, now_ns: Optional[int] = None) -> Any:
        """Return one raw sample exposing ``as_imu_fields()``."""

    def close(self) -> None:
        """Park the inertial sensor."""


class FrameSourceLike(Protocol):
    def next_frame(self, timeout_s: float = 1.0) -> Any:
        """Block until a new frame is available."""

    def close(self) -> None:
        """Stop capturing and close the camera."""


class DetectorLike(Protocol):
    def detect(
        self,
        frame: Any,
        frame_id: int,
        captured_ns: Optional[int] = None,
        source: Optional[str] = None,
    ) -> FrameDetections:
        """Run inference on one captured frame."""


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyStats:
    """Summary of one latency distribution."""

    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


@dataclass(frozen=True)
class RuntimeSummary:
    """Snapshot of a runtime session, safe to log or assert on."""

    duration_s: float
    ticks: int
    vision_frames: int
    frame_drops: int
    degraded_events: int
    sector_hazard_counts: Dict[str, int]
    sensor_to_haptic_ms: LatencyStats
    fusion_eval_ms: LatencyStats
    vision_call_ms: LatencyStats
    loop_jitter_ms: LatencyStats
    vision_age_ms: float

    @property
    def tick_rate_hz(self) -> float:
        return self.ticks / self.duration_s if self.duration_s > 0.0 else 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "duration_s": round(self.duration_s, 3),
            "ticks": self.ticks,
            "tick_rate_hz": round(self.tick_rate_hz, 2),
            "vision_frames": self.vision_frames,
            "frame_drops": self.frame_drops,
            "degraded_events": self.degraded_events,
            "vision_age_ms": round(self.vision_age_ms, 3),
            "sector_hazard_counts": dict(self.sector_hazard_counts),
            "sensor_to_haptic_ms": self.sensor_to_haptic_ms.as_dict(),
            "fusion_eval_ms": self.fusion_eval_ms.as_dict(),
            "vision_call_ms": self.vision_call_ms.as_dict(),
            "loop_jitter_ms": self.loop_jitter_ms.as_dict(),
        }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * weight)


def _stats(values: Iterable[float]) -> LatencyStats:
    ordered = sorted(values)
    if not ordered:
        return LatencyStats(0, 0.0, 0.0, 0.0, 0.0)
    return LatencyStats(
        count=len(ordered),
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        p99_ms=_percentile(ordered, 0.99),
        max_ms=float(ordered[-1]),
    )


class RuntimeMetrics:
    """Thread-safe rolling telemetry for the fast loop and the vision thread."""

    def __init__(self, window: int = _DEFAULT_WINDOW) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self._lock = threading.Lock()
        self._window = int(window)
        self._sensor_to_haptic: Deque[float] = deque(maxlen=window)
        self._fusion_eval: Deque[float] = deque(maxlen=window)
        self._vision_call: Deque[float] = deque(maxlen=window)
        self._loop_jitter: Deque[float] = deque(maxlen=window)
        self._started_ns: Optional[int] = None
        self._ticks = 0
        self._vision_frames = 0
        self._frame_drops = 0
        self._degraded_events = 0
        self._degraded_latched = False
        self._vision_age_ms = 0.0
        self._sector_counts: Dict[str, int] = {sector.value: 0 for sector in _SECTORS}

    def begin(self, now_ns: Optional[int] = None) -> None:
        """Reset every window and counter for a fresh session."""

        with self._lock:
            for window in (
                self._sensor_to_haptic,
                self._fusion_eval,
                self._vision_call,
                self._loop_jitter,
            ):
                window.clear()
            self._started_ns = now_ns if now_ns is not None else time.monotonic_ns()
            self._ticks = 0
            self._vision_frames = 0
            self._frame_drops = 0
            self._degraded_events = 0
            self._degraded_latched = False
            self._vision_age_ms = 0.0
            for sector in _SECTORS:
                self._sector_counts[sector.value] = 0

    def record_tick(
        self,
        *,
        sensor_to_haptic_ms: float,
        fusion_eval_ms: float,
        loop_jitter_ms: float,
        vision_age_ms: float,
        decision: Any,
    ) -> None:
        with self._lock:
            self._ticks += 1
            self._sensor_to_haptic.append(max(0.0, float(sensor_to_haptic_ms)))
            self._fusion_eval.append(max(0.0, float(fusion_eval_ms)))
            self._loop_jitter.append(float(loop_jitter_ms))
            self._vision_age_ms = max(0.0, float(vision_age_ms))
            for sector in _SECTORS:
                state = decision.sectors.get(sector)
                if state is None:
                    continue
                if int(state.level) >= int(ExtendedHazardLevel.WARNING):
                    self._sector_counts[sector.value] += 1
            degraded = bool(getattr(decision, "degraded_sensors", ()))
            if degraded and not self._degraded_latched:
                self._degraded_events += 1
            self._degraded_latched = degraded

    def record_vision(self, call_ms: float, age_ms: float) -> None:
        with self._lock:
            self._vision_frames += 1
            self._vision_call.append(max(0.0, float(call_ms)))
            self._vision_age_ms = max(0.0, float(age_ms))

    def record_vision_drop(self, count: int = 1) -> None:
        with self._lock:
            self._frame_drops += max(0, int(count))

    @property
    def ticks(self) -> int:
        with self._lock:
            return self._ticks

    @property
    def vision_frames(self) -> int:
        with self._lock:
            return self._vision_frames

    @property
    def frame_drops(self) -> int:
        with self._lock:
            return self._frame_drops

    @property
    def vision_age_ms(self) -> float:
        with self._lock:
            return self._vision_age_ms

    def snapshot(self, now_ns: Optional[int] = None) -> RuntimeSummary:
        with self._lock:
            if self._started_ns is None:
                duration_s = 0.0
            else:
                end_ns = now_ns if now_ns is not None else time.monotonic_ns()
                duration_s = max(0.0, (end_ns - self._started_ns) / 1e9)
            return RuntimeSummary(
                duration_s=duration_s,
                ticks=self._ticks,
                vision_frames=self._vision_frames,
                frame_drops=self._frame_drops,
                degraded_events=self._degraded_events,
                sector_hazard_counts=dict(self._sector_counts),
                sensor_to_haptic_ms=_stats(self._sensor_to_haptic),
                fusion_eval_ms=_stats(self._fusion_eval),
                vision_call_ms=_stats(self._vision_call),
                loop_jitter_ms=_stats(self._loop_jitter),
                vision_age_ms=self._vision_age_ms,
            )


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class SmartCaneRuntime:
    """Owns the sensor threads, the fusion engine, and the haptic sink."""

    def __init__(
        self,
        *,
        fusion: SmartCaneFusionEngine,
        haptic_sink: DirectionalHapticSink,
        tof_array: Optional[ToFArrayLike] = None,
        imu_driver: Optional[IMUDriverLike] = None,
        ultrasonic_driver: Optional[UltrasonicDriverLike] = None,
        frame_source: Optional[FrameSourceLike] = None,
        detector: Optional[DetectorLike] = None,
        control_loop_hz: float = 50.0,
        max_frame_age_before_infer_ms: float = 200.0,
        imu_calibration_duration_s: float = 0.5,
        log_jsonl_path: Optional[Any] = None,
        closeables: Sequence[Any] = (),
        name: str = "smart-cane",
    ) -> None:
        if control_loop_hz <= 0.0:
            raise ValueError("control_loop_hz must be positive")
        if max_frame_age_before_infer_ms < 0.0:
            raise ValueError("max_frame_age_before_infer_ms must not be negative")
        if imu_calibration_duration_s < 0.0:
            raise ValueError("imu_calibration_duration_s must not be negative")
        self.fusion = fusion
        self.haptic_sink = haptic_sink
        self.tof_array = tof_array
        self.imu_driver = imu_driver
        self.ultrasonic_driver = ultrasonic_driver
        self.frame_source = frame_source
        self.detector = detector
        self.control_loop_hz = float(control_loop_hz)
        self.max_frame_age_before_infer_ms = float(max_frame_age_before_infer_ms)
        self.imu_calibration_duration_s = float(imu_calibration_duration_s)
        self.log_jsonl_path = Path(log_jsonl_path) if log_jsonl_path else None
        self.metrics = RuntimeMetrics()

        self._resources: List[Any] = list(closeables)
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._thread_errors: List[BaseException] = []
        self._vision_errors: List[BaseException] = []
        self._imu_errors: List[BaseException] = []
        self._threads: Dict[str, threading.Thread] = {}
        self._running = False
        self._closed = False
        self._log_file: Optional[Any] = None
        self._log_lock = threading.Lock()

        self._frame_condition = threading.Condition()
        self._pending_frame: Any = None
        self._pending_overwrites = 0
        self._latest_decision: Any = None
        self._latest_haptic_frame: Optional[DirectionalHapticFrame] = None
        self._latest_detections: Optional[FrameDetections] = None

        self._name = name

    # -- public state ----------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def thread_errors(self) -> Tuple[BaseException, ...]:
        with self._error_lock:
            return tuple(self._thread_errors)

    @property
    def vision_errors(self) -> Tuple[BaseException, ...]:
        with self._error_lock:
            return tuple(self._vision_errors)

    @property
    def imu_errors(self) -> Tuple[BaseException, ...]:
        with self._error_lock:
            return tuple(self._imu_errors)

    @property
    def latest_decision(self) -> Any:
        with self._state_lock:
            return self._latest_decision

    @property
    def latest_haptic_frame(self) -> Optional[DirectionalHapticFrame]:
        with self._state_lock:
            return self._latest_haptic_frame

    @property
    def latest_detections(self) -> Optional[FrameDetections]:
        with self._state_lock:
            return self._latest_detections

    def summary(self) -> RuntimeSummary:
        return self.metrics.snapshot()

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "SmartCaneRuntime":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def start(self) -> None:
        """Calibrate, spawn every thread, and begin the control loop."""

        with self._lifecycle_lock:
            if self._running:
                return
            if self._closed:
                raise RuntimeError(
                    f"{self._name} runtime cannot be restarted after stop()"
                )
            self._stop_event.clear()
            with self._error_lock:
                self._thread_errors.clear()
                self._vision_errors.clear()
                self._imu_errors.clear()
            with self._frame_condition:
                self._pending_frame = None
                self._pending_overwrites = 0
            self.metrics.begin()
            try:
                if self.log_jsonl_path is not None:
                    self.log_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                    self._log_file = self.log_jsonl_path.open(
                        "w", encoding="utf-8", buffering=1
                    )
                self._calibrate_imu_once()
                if self.frame_source is not None and self.detector is not None:
                    starter = getattr(self.frame_source, "start", None)
                    if callable(starter):
                        starter()
                    self._spawn("camera", self._camera_loop)
                    self._spawn("vision", self._vision_loop)
                if self.ultrasonic_driver is not None:
                    self._spawn("ultrasonic", self._ultrasonic_loop)
                if callable(getattr(self.haptic_sink, "tick_watchdog", None)):
                    self._spawn("watchdog", self._watchdog_loop)
                self._spawn("control", self._control_loop)
            except BaseException:
                self.stop()
                raise
            self._running = True

    def stop(self) -> None:
        """Stop every thread and park the motors, whatever went wrong."""

        with self._lifecycle_lock:
            if self._closed and not self._running:
                return
            self._stop_event.set()
            with self._frame_condition:
                self._frame_condition.notify_all()
            current = threading.current_thread()
            for name in ("vision", "camera", "ultrasonic", "watchdog", "control"):
                thread = self._threads.get(name)
                if thread is None or thread is current:
                    continue
                thread.join(timeout=2.0)
            self._threads.clear()
            self._shutdown_resources()
            self._running = False
            self._closed = True

    def run_for(self, duration_s: float) -> RuntimeSummary:
        """Run for ``duration_s`` seconds, then tear down and return metrics."""

        if duration_s < 0.0:
            raise ValueError("duration_s must not be negative")
        deadline = time.monotonic() + duration_s
        self.start()
        try:
            while not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._stop_event.wait(min(remaining, 0.05))
        finally:
            self.stop()
        self._raise_control_error()
        return self.summary()

    def install_signal_handlers(self) -> None:
        """Turn SIGINT and SIGTERM into a clean stop request."""

        def _handle(_signum: int, _frame: Any) -> None:
            self._stop_event.set()

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signum, _handle)
            except (ValueError, OSError):
                # Not on the main thread, or the platform has no such signal.
                continue

    # -- threads ---------------------------------------------------------

    def _spawn(self, name: str, target: Any) -> None:
        thread = threading.Thread(
            target=target, name=f"{self._name}-{name}", daemon=True
        )
        thread.start()
        self._threads[name] = thread

    def _control_loop(self) -> None:
        period_ns = int(round(1e9 / self.control_loop_hz))
        next_deadline_ns = time.monotonic_ns()
        first = True
        try:
            while not self._stop_event.is_set():
                loop_start_ns = time.monotonic_ns()
                jitter_ms = 0.0 if first else (loop_start_ns - next_deadline_ns) / 1e6
                first = False

                oldest_sample_ns = self._poll_range_sensors(loop_start_ns)
                imu_sample_ns = self._poll_imu(loop_start_ns)
                if imu_sample_ns is not None:
                    oldest_sample_ns = (
                        imu_sample_ns
                        if oldest_sample_ns is None
                        else min(oldest_sample_ns, imu_sample_ns)
                    )

                fusion_started_ns = time.monotonic_ns()
                decision = self.fusion.evaluate(loop_start_ns)
                fusion_ms = (time.monotonic_ns() - fusion_started_ns) / 1e6

                haptic_frame = decision_to_directional_haptics(
                    decision, self.fusion.config
                )
                self.haptic_sink.emit_frame(haptic_frame)
                haptic_done_ns = time.monotonic_ns()

                reference_ns = (
                    loop_start_ns if oldest_sample_ns is None else oldest_sample_ns
                )
                sensor_to_haptic_ms = max(
                    0.0, (haptic_done_ns - int(reference_ns)) / 1e6
                )

                with self._state_lock:
                    self._latest_decision = decision
                    self._latest_haptic_frame = haptic_frame

                self.metrics.record_tick(
                    sensor_to_haptic_ms=sensor_to_haptic_ms,
                    fusion_eval_ms=fusion_ms,
                    loop_jitter_ms=jitter_ms,
                    vision_age_ms=self._vision_age_ms(loop_start_ns),
                    decision=decision,
                )
                self._write_log_line(decision, haptic_frame, sensor_to_haptic_ms, fusion_ms)

                next_deadline_ns += period_ns
                now_ns = time.monotonic_ns()
                if next_deadline_ns <= now_ns:
                    # Fell behind: realign instead of bursting through a backlog.
                    next_deadline_ns = now_ns
                    continue
                self._stop_event.wait(
                    min((next_deadline_ns - now_ns) / 1e9, period_ns / 1e9)
                )
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            self._record_error(exc)
            self._stop_event.set()
            with self._frame_condition:
                self._frame_condition.notify_all()

    def _ultrasonic_loop(self) -> None:
        driver = self.ultrasonic_driver
        if driver is None:
            return
        try:
            while not self._stop_event.is_set():
                measurement = driver.read_next()
                self.fusion.update_sensor(
                    self._range_reading(
                        measurement, SensorKind.ULTRASONIC, _ULTRASONIC_MAX_RANGE_M
                    )
                )
        except BaseException as exc:  # noqa: BLE001
            self._record_error(exc)
            self._stop_event.set()

    def _camera_loop(self) -> None:
        source = self.frame_source
        if source is None:
            return
        try:
            while not self._stop_event.is_set():
                try:
                    frame = source.next_frame(timeout_s=0.25)
                except TimeoutError:
                    continue
                with self._frame_condition:
                    if self._pending_frame is not None:
                        self._pending_overwrites += 1
                    self._pending_frame = frame
                    self._frame_condition.notify_all()
        except BaseException as exc:  # noqa: BLE001 - vision is optional
            with self._error_lock:
                self._vision_errors.append(exc)

    def _vision_loop(self) -> None:
        detector = self.detector
        if detector is None:
            return
        try:
            while not self._stop_event.is_set():
                with self._frame_condition:
                    while self._pending_frame is None and not self._stop_event.is_set():
                        self._frame_condition.wait(0.1)
                    frame = self._pending_frame
                    self._pending_frame = None
                    overwritten = self._pending_overwrites
                    self._pending_overwrites = 0
                if overwritten:
                    self.metrics.record_vision_drop(overwritten)
                if frame is None or self._stop_event.is_set():
                    continue

                captured_ns = getattr(frame, "captured_ns", None)
                age_ms = (
                    max(0.0, (time.monotonic_ns() - int(captured_ns)) / 1e6)
                    if captured_ns is not None
                    else 0.0
                )
                if age_ms > self.max_frame_age_before_infer_ms:
                    self.metrics.record_vision_drop(1)
                    continue

                started_ns = time.monotonic_ns()
                try:
                    detections = detector.detect(
                        frame.payload,
                        int(frame.frame_id),
                        captured_ns=captured_ns,
                        source=getattr(frame, "source", None),
                    )
                except BaseException as exc:  # noqa: BLE001 - keep the cane alive
                    with self._error_lock:
                        self._vision_errors.append(exc)
                    continue
                call_ms = (time.monotonic_ns() - started_ns) / 1e6
                self.fusion.update_vision(detections)
                with self._state_lock:
                    self._latest_detections = detections
                self.metrics.record_vision(call_ms, age_ms)
        except BaseException as exc:  # noqa: BLE001
            with self._error_lock:
                self._vision_errors.append(exc)

    # -- sensor plumbing -------------------------------------------------

    def _poll_range_sensors(self, now_ns: int) -> Optional[int]:
        if self.tof_array is None:
            return None
        oldest: Optional[int] = None
        for measurement in self.tof_array.read_all(now_ns):
            reading = self._range_reading(measurement, SensorKind.TOF, _HARDWARE_MAX_RANGE_M)
            self.fusion.update_sensor(reading)
            oldest = (
                reading.timestamp_ns
                if oldest is None
                else min(oldest, reading.timestamp_ns)
            )
        return oldest

    def _watchdog_loop(self) -> None:
        ticker = getattr(self.haptic_sink, "tick_watchdog", None)
        if not callable(ticker):
            return
        while not self._stop_event.is_set():
            try:
                ticker(time.monotonic_ns())
            except Exception as exc:  # noqa: BLE001 - keep watchdog alive
                LOGGER.warning("haptic watchdog tick failed: %s", exc)
            self._stop_event.wait(0.05)

    def _poll_imu(self, now_ns: int) -> Optional[int]:
        if self.imu_driver is None:
            return None
        try:
            sample = _to_imu_sample(self.imu_driver.read_raw(now_ns))
        except Exception as exc:  # noqa: BLE001 - IMU faults degrade gracefully
            with self._error_lock:
                self._imu_errors.append(exc)
            LOGGER.warning("IMU read failed; falling back to neutral attitude: %s", exc)
            return None
        self.fusion.update_imu(sample)
        return sample.timestamp_ns

    def _calibrate_imu_once(self) -> None:
        if self.imu_driver is None or self.imu_calibration_duration_s <= 0.0:
            return
        deadline = time.monotonic() + self.imu_calibration_duration_s
        samples: List[IMUSample] = []
        while time.monotonic() < deadline and not self._stop_event.is_set():
            try:
                samples.append(_to_imu_sample(self.imu_driver.read_raw()))
            except Exception as exc:  # noqa: BLE001 - calibration is best effort
                with self._error_lock:
                    self._imu_errors.append(exc)
                LOGGER.warning("IMU startup calibration aborted: %s", exc)
                return
            time.sleep(0.005)
        if not samples:
            return
        self.fusion.calibrate_imu(samples)
        for sample in samples:
            self.fusion.update_imu(sample)

    def _vision_age_ms(self, now_ns: int) -> float:
        with self._state_lock:
            detections = self._latest_detections
        if detections is None:
            return math.inf
        age_ms = (now_ns - detections.model_finished_ns) / 1e6
        if not math.isfinite(age_ms):
            return math.inf
        return max(0.0, age_ms)

    @staticmethod
    def _range_reading(
        measurement: Any, kind: SensorKind, max_range_m: float
    ) -> RangeReading:
        return RangeReading(
            sensor_id=str(measurement.sensor_id),
            kind=kind,
            timestamp_ns=int(measurement.timestamp_ns),
            sector=measurement.sector,
            distance_m=measurement.distance_m,
            valid=bool(measurement.valid),
            no_target=bool(measurement.no_target),
            max_range_m=max_range_m,
        )

    # -- logging and teardown --------------------------------------------

    def _write_log_line(
        self,
        decision: Any,
        haptic_frame: DirectionalHapticFrame,
        sensor_to_haptic_ms: float,
        fusion_ms: float,
    ) -> None:
        if self._log_file is None:
            return
        record = {
            "t_ns": decision.timestamp_ns,
            "sensor_to_haptic_ms": round(sensor_to_haptic_ms, 4),
            "fusion_eval_ms": round(fusion_ms, 4),
            "decision": decision.as_dict(),
            "haptics": haptic_frame.as_dict(),
        }
        line = json.dumps(record, separators=(",", ":"))
        with self._log_lock:
            if self._log_file is None:
                return
            try:
                self._log_file.write(line + "\n")
            except (OSError, ValueError) as exc:
                with self._error_lock:
                    self._thread_errors.append(exc)
                self._log_file = None

    def _record_error(self, exc: BaseException) -> None:
        with self._error_lock:
            self._thread_errors.append(exc)

    def _raise_control_error(self) -> None:
        errors = self.thread_errors
        if errors:
            raise RuntimeError(
                f"{self._name} runtime thread failed: {errors[0]!r}"
            ) from errors[0]

    def _shutdown_resources(self) -> None:
        """Park the motors first, then release every device."""

        sink = self.haptic_sink
        try:
            sink.stop()
        except BaseException as exc:  # noqa: BLE001 - teardown must continue
            with self._error_lock:
                self._thread_errors.append(exc)
        closer = getattr(sink, "close", None)
        if callable(closer):
            try:
                closer()
            except BaseException:  # noqa: BLE001
                pass
        if self.frame_source is not None:
            try:
                self.frame_source.close()
            except BaseException:  # noqa: BLE001
                pass
        for resource in reversed(self._resources):
            try:
                resource.close()
            except BaseException:  # noqa: BLE001 - one dead bus must not block the rest
                continue
        with self._log_lock:
            if self._log_file is not None:
                try:
                    self._log_file.flush()
                    self._log_file.close()
                except (OSError, ValueError):
                    pass
                self._log_file = None


def _to_imu_sample(raw: Any) -> IMUSample:
    fields = getattr(raw, "as_imu_fields", None)
    if callable(fields):
        return IMUSample(**fields())
    ax, ay, az = raw.accel_mps2
    gx, gy, gz = raw.gyro_rad_s
    return IMUSample(
        timestamp_ns=int(raw.timestamp_ns),
        ax_mps2=float(ax),
        ay_mps2=float(ay),
        az_mps2=float(az),
        gx_rps=float(gx),
        gy_rps=float(gy),
        gz_rps=float(gz),
        temp_c=getattr(raw, "temperature_c", None),
        valid=bool(getattr(raw, "valid", True)),
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class SimulatedObstacle:
    """One obstacle closing on the cane along its shaft axis."""

    sector: Sector
    start_distance_m: float = 1.90
    closing_speed_mps: float = 2.60
    stop_distance_m: float = 0.28
    label: str = "generic_obstacle"


class SimulatedWorld:
    """Deterministic scene shared by the simulated ToF, ultrasonic, and camera."""

    def __init__(self, obstacles: Optional[Sequence[SimulatedObstacle]] = None) -> None:
        self._lock = threading.Lock()
        self._obstacles: List[SimulatedObstacle] = list(
            obstacles
            if obstacles is not None
            else [SimulatedObstacle(Sector.CENTER, label="pole")]
        )
        self._origin_ns = time.monotonic_ns()

    def elapsed_s(self, now_ns: Optional[int] = None) -> float:
        now = now_ns if now_ns is not None else time.monotonic_ns()
        return max(0.0, (now - self._origin_ns) / 1e9)

    def distance_m(
        self, sector: Sector, now_ns: Optional[int] = None
    ) -> Optional[float]:
        with self._lock:
            obstacles = [item for item in self._obstacles if item.sector == sector]
        if not obstacles:
            return None
        elapsed_s = self.elapsed_s(now_ns)
        nearest: Optional[float] = None
        for obstacle in obstacles:
            distance = max(
                obstacle.stop_distance_m,
                obstacle.start_distance_m - obstacle.closing_speed_mps * elapsed_s,
            )
            if nearest is None or distance < nearest:
                nearest = distance
        return nearest

    def sector_obstacles(self, sector: Sector) -> List[SimulatedObstacle]:
        with self._lock:
            return [item for item in self._obstacles if item.sector == sector]

    def set_obstacles(self, obstacles: Sequence[SimulatedObstacle]) -> None:
        with self._lock:
            self._obstacles = list(obstacles)


class SimulatedToFArray:
    """Three VL53L0X stand-ins reading the simulated world."""

    def __init__(self, world: SimulatedWorld, noise_m: float = 0.006) -> None:
        self.world = world
        self.noise_m = float(noise_m)
        self._sensors = (
            ("tof-left", Sector.LEFT),
            ("tof-center", Sector.CENTER),
            ("tof-right", Sector.RIGHT),
        )
        self.closed = False

    def read_all(self, now_ns: Optional[int] = None) -> List[Any]:
        from .hardware.vl53l0x import ToFMeasurement

        timestamp_ns = now_ns if now_ns is not None else time.monotonic_ns()
        measurements: List[Any] = []
        for index, (sensor_id, sector) in enumerate(self._sensors):
            distance = self.world.distance_m(sector, timestamp_ns)
            if distance is None or distance > _HARDWARE_MAX_RANGE_M:
                measurements.append(
                    ToFMeasurement(
                        sensor_id=sensor_id,
                        sector=sector,
                        timestamp_ns=timestamp_ns,
                        distance_m=None,
                        raw_mm=None,
                        range_status=0,
                        valid=True,
                        no_target=True,
                    )
                )
                continue
            phase = self.world.elapsed_s(timestamp_ns) * 41.0 + index
            noisy = max(0.02, distance + self.noise_m * math.sin(phase))
            measurements.append(
                ToFMeasurement(
                    sensor_id=sensor_id,
                    sector=sector,
                    timestamp_ns=timestamp_ns,
                    distance_m=noisy,
                    raw_mm=int(round(noisy * 1000.0)),
                    range_status=11,
                    valid=True,
                    no_target=False,
                )
            )
        return measurements

    def close(self) -> None:
        self.closed = True


class SimulatedUltrasonicDriver:
    """Alternating HC-SR04 stand-ins with an optional per-ping delay."""

    def __init__(
        self,
        world: SimulatedWorld,
        read_delay_s: float = 0.0,
        noise_m: float = 0.004,
    ) -> None:
        self.world = world
        self.read_delay_s = max(0.0, float(read_delay_s))
        self.noise_m = float(noise_m)
        self._sensors = (("us-center-1", Sector.CENTER), ("us-center-2", Sector.CENTER))
        self._next_index = 0
        self.closed = False
        self.reads = 0

    def read_next(self, now_ns: Optional[int] = None) -> Any:
        from .hardware.hcsr04 import UltrasonicMeasurement

        if self.read_delay_s:
            time.sleep(self.read_delay_s)
        index = self._next_index
        self._next_index = (index + 1) % len(self._sensors)
        sensor_id, sector = self._sensors[index]
        self.reads += 1
        timestamp_ns = now_ns if now_ns is not None else time.monotonic_ns()
        distance = self.world.distance_m(sector, timestamp_ns)
        if distance is None or distance > _ULTRASONIC_MAX_RANGE_M:
            return UltrasonicMeasurement(
                sensor_id=sensor_id,
                sector=sector,
                timestamp_ns=timestamp_ns,
                distance_m=None,
                echo_s=None,
                valid=True,
                no_target=True,
            )
        noisy = max(
            0.02,
            distance + self.noise_m * math.sin(self.world.elapsed_s(timestamp_ns) * 29.0 + index),
        )
        return UltrasonicMeasurement(
            sensor_id=sensor_id,
            sector=sector,
            timestamp_ns=timestamp_ns,
            distance_m=noisy,
            echo_s=2.0 * noisy / 343.0,
            valid=True,
            no_target=False,
        )

    def close(self) -> None:
        self.closed = True


class SimulatedIMUDriver:
    """A still MPU-6050 with a touch of sensor noise."""

    def __init__(self, world: SimulatedWorld) -> None:
        self.world = world
        self.closed = False

    def read_raw(self, now_ns: Optional[int] = None) -> Any:
        from .hardware.mpu6050 import RawIMUSample

        timestamp_ns = now_ns if now_ns is not None else time.monotonic_ns()
        elapsed_s = self.world.elapsed_s(timestamp_ns)
        tilt = 0.004 * math.sin(elapsed_s * 3.1)
        return RawIMUSample(
            timestamp_ns=timestamp_ns,
            accel_mps2=(0.02 * tilt, 0.02 * tilt, 9.80665),
            gyro_rad_s=(0.001 * tilt, 0.001 * tilt, 0.001 * tilt),
            temperature_c=31.5,
            accel_raw=(0, 0, 16384),
            gyro_raw=(0, 0, 0),
            valid=True,
        )

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class SimulatedSceneFrame:
    """Opaque camera payload for :class:`SimulatedDetector`."""

    frame_id: int
    width: int
    height: int
    distance_m: Optional[float]
    sector: Sector
    label: str


class SimulatedFrameSource:
    """A camera at a fixed frame rate whose scene shrinks as the obstacle nears."""

    def __init__(
        self,
        world: SimulatedWorld,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        sector: Sector = Sector.CENTER,
        emit_rgb_array: bool = False,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        self.world = world
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.sector = sector
        self.emit_rgb_array = bool(emit_rgb_array)
        self._lock = threading.Lock()
        self._next_ns = 0
        self._frame_id = 0
        self._stop = threading.Event()
        self._closed = False
        self._rgb_frame: Any = None
        if self.emit_rgb_array:
            import numpy as np

            self._rgb_frame = np.full((self.height, self.width, 3), 114, dtype=np.uint8)

    def start(self) -> None:
        self._next_ns = time.monotonic_ns()

    def next_frame(self, timeout_s: float = 1.0) -> Any:
        from .input_sources import BenchmarkFrame

        period_ns = int(round(1e9 / self.fps))
        deadline_ns = time.monotonic_ns() + int(max(0.0, timeout_s) * 1e9)
        while not self._stop.is_set():
            now_ns = time.monotonic_ns()
            if now_ns >= self._next_ns:
                break
            if now_ns >= deadline_ns:
                raise TimeoutError(f"no simulated frame within {timeout_s} seconds")
            time.sleep(min((self._next_ns - now_ns) / 1e9, 0.01))
        if self._stop.is_set():
            raise TimeoutError("frame source closed")
        with self._lock:
            frame_id = self._frame_id
            self._frame_id += 1
        captured_ns = time.monotonic_ns()
        self._next_ns = max(self._next_ns + period_ns, captured_ns)
        if self.emit_rgb_array and self._rgb_frame is not None:
            payload: Any = self._rgb_frame
        else:
            obstacles = self.world.sector_obstacles(self.sector)
            distance = self.world.distance_m(self.sector, captured_ns)
            payload = SimulatedSceneFrame(
                frame_id=frame_id,
                width=self.width,
                height=self.height,
                distance_m=distance,
                sector=self.sector,
                label=obstacles[0].label if obstacles else "generic_obstacle",
            )
        return BenchmarkFrame(
            frame_id=frame_id,
            payload=payload,
            source="simulated-camera",
            source_type="simulated",
            captured_ns=captured_ns,
            capture_started_ns=captured_ns,
            available_ns=captured_ns,
        )

    def close(self) -> None:
        self._stop.set()
        self._closed = True


class SimulatedDetector:
    """Projects the simulated obstacle into a plausible detection box."""

    def detect(
        self,
        frame: Any,
        frame_id: int,
        captured_ns: Optional[int] = None,
        source: Optional[str] = None,
    ) -> FrameDetections:
        started_ns = time.monotonic_ns()
        scene = frame if isinstance(frame, SimulatedSceneFrame) else None
        width = scene.width if scene is not None else 640
        height = scene.height if scene is not None else 480
        detections: List[Detection] = []
        if scene is not None and scene.distance_m is not None and scene.distance_m <= 3.8:
            focal_px = (height / 2.0) / math.tan(math.radians(48.8 / 2.0))
            apparent_px = max(
                focal_px * 2.20 / max(scene.distance_m, 0.3), 24.0
            )
            bottom = 0.82 * height
            center_x = 0.5 * width
            if scene.sector == Sector.LEFT:
                center_x = 0.22 * width
            elif scene.sector == Sector.RIGHT:
                center_x = 0.78 * width
            detections.append(
                Detection(
                    class_id=0,
                    class_name=scene.label,
                    confidence=0.82,
                    box=BoundingBox(
                        x1=max(0.0, center_x - apparent_px / 2.0),
                        y1=max(0.0, bottom - apparent_px),
                        x2=min(float(width), center_x + apparent_px / 2.0),
                        y2=bottom,
                    ),
                    sector=scene.sector,
                )
            )
        finished_ns = time.monotonic_ns()
        return FrameDetections(
            frame_id=frame_id,
            captured_ns=captured_ns if captured_ns is not None else started_ns,
            model_started_ns=started_ns,
            model_finished_ns=finished_ns,
            width=width,
            height=height,
            detections=detections,
            backend="simulated",
            model_id="simulated-detector",
            source=source,
        )


class SimulatedHapticSink:
    """Records frames, tracks per-motor intensity, and proves teardown."""

    def __init__(self, max_frames: int = 4096) -> None:
        self._lock = threading.Lock()
        self.frames: Deque[DirectionalHapticFrame] = deque(maxlen=max_frames)
        self.frame_count = 0
        self.stopped = False
        self.stop_calls = 0
        self.intensities: Dict[Sector, float] = {sector: 0.0 for sector in _SECTORS}
        self.levels: Dict[Sector, ExtendedHazardLevel] = {
            sector: ExtendedHazardLevel.CLEAR for sector in _SECTORS
        }

    def emit_frame(self, frame: DirectionalHapticFrame) -> None:
        with self._lock:
            if self.stopped:
                return
            self.frames.append(frame)
            self.frame_count += 1
            for sector in _SECTORS:
                command = frame.motors.get(sector)
                if command is None:
                    continue
                self.intensities[sector] = float(command.intensity)
                self.levels[sector] = command.level

    def tick_watchdog(self, now_ns: int) -> bool:
        return False

    def stop(self) -> None:
        with self._lock:
            self.stop_calls += 1
            self.stopped = True
            for sector in _SECTORS:
                self.intensities[sector] = 0.0
                self.levels[sector] = ExtendedHazardLevel.CLEAR

    def close(self) -> None:
        self.stop()

    @property
    def max_intensity(self) -> float:
        with self._lock:
            return max(self.intensities.values()) if self.intensities else 0.0

    @property
    def all_motors_zeroed(self) -> bool:
        with self._lock:
            return all(value == 0.0 for value in self.intensities.values())


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _fusion_config_from_hardware(config: Any) -> SmartCaneFusionConfig:
    timing = config.timing
    expected: Dict[str, Sector] = {}
    pitch_offsets: Dict[str, float] = {}
    for spec in getattr(config, "tof_sensors", ()):
        expected[spec.sensor_id] = spec.sector
    for spec in getattr(config, "ultrasonic_sensors", ()):
        expected[spec.sensor_id] = spec.sector
        offset = float(getattr(spec, "pitch_offset_deg", 0.0) or 0.0)
        if offset:
            pitch_offsets[spec.sensor_id] = offset
    kwargs: Dict[str, Any] = {
        "critical_distance_m": timing.critical_distance_m,
        "warning_distance_m": timing.warning_distance_m,
        "sensor_stale_after_ms": timing.sensor_stale_after_ms,
        "vision_stale_after_ms": timing.vision_stale_after_ms,
    }
    if expected:
        kwargs["expected_sensors"] = expected
    if pitch_offsets:
        kwargs["sensor_pitch_offsets_deg"] = pitch_offsets
    return SmartCaneFusionConfig(**kwargs)


def _build_detector(model_path: str, num_threads: int) -> Any:
    """Prefer the OpenVINO adapter, falling back to Ultralytics."""

    from .fusion_v2 import DEFAULT_CLASS_PRIORS

    class_thresholds = {
        name: prior.min_confidence for name, prior in DEFAULT_CLASS_PRIORS.items()
    }
    try:
        from .openvino_detector import OpenVINODetector

        return OpenVINODetector(
            model_path,
            num_threads=num_threads,
            class_thresholds=class_thresholds,
        )
    except (ImportError, AttributeError, TypeError, RuntimeError, OSError):
        from .detector import UltralyticsDetector

        return UltralyticsDetector(model_path, runtime_threads=num_threads)


def build_hardware_runtime(
    config: Any,
    enable_camera: bool = True,
    model_path: Optional[str] = None,
    num_threads: int = 2,
    log_jsonl_path: Optional[Any] = None,
    *,
    haptic_mode: str = "rtp",
    control_loop_hz: Optional[float] = None,
    fusion_config: Optional[SmartCaneFusionConfig] = None,
) -> SmartCaneRuntime:
    """Wire the real Pi 5 hardware into one :class:`SmartCaneRuntime`."""

    from .hardware import (
        DRV2605LController,
        HCSR04PairDriver,
        MPU6050Driver,
        SMBus2Adapter,
        TCA9548AMux,
        VL53L0XArray,
        create_gpio_backend,
    )

    timing = config.timing
    bus = SMBus2Adapter(config.i2c_bus)
    mux = TCA9548AMux(bus, config.mux_address)
    opened: List[Any] = [bus, mux]
    haptics = None
    try:
        tof_array = VL53L0XArray(mux, config.tof_sensors)
        opened.append(tof_array)
        tof_array.initialize()
        tof_array.start_continuous()
        imu_driver = MPU6050Driver(mux, config.imu)
        opened.append(imu_driver)
        imu_driver.initialize()
        gpio = create_gpio_backend(config.gpio_chip)
        ultrasonic = HCSR04PairDriver(
            gpio,
            config.ultrasonic_sensors,
            guard_interval_ms=timing.ultrasonic_guard_interval_ms,
            timeout_ms=timing.ultrasonic_timeout_ms,
            ambient_temp_c=timing.ambient_temp_c,
        )
        opened.append(ultrasonic)
        haptics = DRV2605LController(
            mux,
            config.haptic_motors,
            mode=haptic_mode,
            watchdog_timeout_ms=timing.haptic_watchdog_timeout_ms,
        )
        haptics.initialize()

        frame_source = None
        detector = None
        if enable_camera:
            from .picamera2_source import Picamera2LatestFrameSource

            camera = config.camera
            frame_source = Picamera2LatestFrameSource(
                width=camera.width,
                height=camera.height,
                fps=camera.fps,
                pixel_format=camera.pixel_format,
            )
            detector = _build_detector(
                model_path or camera.model_path, num_threads or camera.num_threads
            )
    except BaseException:
        # A half-built rig must not leave a motor buzzing or the bus selected.
        if haptics is not None:
            try:
                haptics.stop()
            except BaseException:
                pass
        for resource in reversed(opened):
            try:
                resource.close()
            except BaseException:
                continue
        raise

    return SmartCaneRuntime(
        fusion=SmartCaneFusionEngine(fusion_config or _fusion_config_from_hardware(config)),
        haptic_sink=haptics,
        tof_array=tof_array,
        imu_driver=imu_driver,
        ultrasonic_driver=ultrasonic,
        frame_source=frame_source,
        detector=detector,
        control_loop_hz=control_loop_hz or timing.control_loop_hz,
        imu_calibration_duration_s=0.5,
        log_jsonl_path=log_jsonl_path,
        closeables=(tof_array, ultrasonic, imu_driver, mux, bus),
        name="smart-cane-hardware",
    )


def build_simulated_runtime(
    config: Optional[Any] = None,
    detector: Optional[Any] = None,
    enable_camera: bool = False,
    log_jsonl_path: Optional[Any] = None,
    *,
    world: Optional[SimulatedWorld] = None,
    control_loop_hz: float = 50.0,
    ultrasonic_read_delay_s: float = 0.0,
    max_frame_age_before_infer_ms: float = 200.0,
    imu_calibration_duration_s: float = 0.0,
) -> SmartCaneRuntime:
    """Run the identical four-thread runtime against in-memory drivers."""

    if config is None:
        from .hardware.config import default_pi5_config

        config = default_pi5_config()
    scene = world or SimulatedWorld()
    tof_array = SimulatedToFArray(scene)
    ultrasonic = SimulatedUltrasonicDriver(scene, read_delay_s=ultrasonic_read_delay_s)
    imu_driver = SimulatedIMUDriver(scene)
    haptic_sink = SimulatedHapticSink()
    frame_source = None
    active_detector = None
    if enable_camera:
        active_detector = detector or SimulatedDetector()
        emit_rgb = not isinstance(active_detector, SimulatedDetector)
        frame_source = SimulatedFrameSource(
            scene,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            emit_rgb_array=emit_rgb,
        )

    return SmartCaneRuntime(
        fusion=SmartCaneFusionEngine(_fusion_config_from_hardware(config)),
        haptic_sink=haptic_sink,
        tof_array=tof_array,
        imu_driver=imu_driver,
        ultrasonic_driver=ultrasonic,
        frame_source=frame_source,
        detector=active_detector,
        control_loop_hz=control_loop_hz,
        max_frame_age_before_infer_ms=max_frame_age_before_infer_ms,
        imu_calibration_duration_s=imu_calibration_duration_s,
        log_jsonl_path=log_jsonl_path,
        closeables=(tof_array, ultrasonic, imu_driver),
        name="smart-cane-simulated",
    )
