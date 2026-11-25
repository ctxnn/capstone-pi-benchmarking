"""Background pollers that turn raw sensor reads into fusion ``SensorReading`` values.

There are three pollers, each on its own thread so a slow device never delays
another: the three ToF sensors, the two ultrasonic sensors, and the IMU.
Timestamps are ``time.perf_counter_ns()`` taken right after each read, the same
clock the vision path and the fusion engine use.

Errors never propagate out of a poller. A failing sensor simply stops producing
new readings; the fusion engine then ages its last reading out after 250 ms and
ignores it, and reports a sensor fault only if *no* range sensor is left.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Mapping, Optional, Protocol, Sequence

from ..fusion import SensorKind, SensorReading
from ..types import Sector
from .config import TofConfig, UltrasonicConfig
from .imu import JoltDetector, JoltEvent, MotionMonitor, Mpu6050
from .tof import RangeSample, Vl53l0xDevice
from .ultrasonic import Hcsr04

LOGGER = logging.getLogger("yolo_pi.cane.sensors")


class SensorSink(Protocol):
    def update_sensor(self, reading: SensorReading) -> None: ...


@dataclass
class SensorHealth:
    sensor_id: str
    ok_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_ok_ns: Optional[int] = None
    last_distance_m: Optional[float] = None
    last_error: Optional[str] = None
    restarts: int = 0

    def record_ok(self, now_ns: int, distance_m: Optional[float]) -> None:
        self.ok_count += 1
        self.consecutive_errors = 0
        self.last_ok_ns = now_ns
        self.last_distance_m = distance_m

    def record_error(self, error: BaseException) -> None:
        self.error_count += 1
        self.consecutive_errors += 1
        self.last_error = f"{type(error).__name__}: {error}"

    def as_dict(self, now_ns: int) -> Dict[str, object]:
        age_ms = None if self.last_ok_ns is None else round((now_ns - self.last_ok_ns) / 1e6, 1)
        return {
            "ok": self.ok_count,
            "errors": self.error_count,
            "age_ms": age_ms,
            "distance_m": None if self.last_distance_m is None else round(self.last_distance_m, 3),
            "last_error": self.last_error,
            "restarts": self.restarts,
        }


class _Poller(threading.Thread):
    """Runs ``step()`` in a loop until stopped; ``step`` returns seconds to wait."""

    def __init__(self, name: str, clock_ns: Callable[[], int] = time.perf_counter_ns) -> None:
        super().__init__(name=name, daemon=True)
        self._halt = threading.Event()
        self._clock_ns = clock_ns
        self.crashed: Optional[BaseException] = None

    def step(self) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self) -> None:
        try:
            while not self._halt.is_set():
                delay = self.step()
                if delay > 0:
                    self._halt.wait(delay)
        except BaseException as exc:  # a poller bug must be visible, not silent
            self.crashed = exc
            LOGGER.exception("%s poller crashed", self.name)

    def stop(self) -> None:
        self._halt.set()


class ToFPoller(_Poller):
    def __init__(
        self,
        devices: Mapping[str, Vl53l0xDevice],
        sectors: Mapping[str, Sector],
        sink: SensorSink,
        config: TofConfig,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        super().__init__("tof-poller", clock_ns)
        self._devices = dict(devices)
        self._sectors = dict(sectors)
        self._sink = sink
        self._config = config
        self._next_retry_ns: Dict[str, int] = {}
        self.health: Dict[str, SensorHealth] = {name: SensorHealth(name) for name in self._devices}

    def step(self) -> float:
        for sensor_id, device in self._devices.items():
            health = self.health[sensor_id]
            now_ns = self._clock_ns()
            if health.consecutive_errors >= self._config.reinit_after_errors:
                self._maybe_restart(sensor_id, device, health, now_ns)
                continue
            try:
                sample = device.poll()
            except Exception as exc:
                health.record_error(exc)
                if health.consecutive_errors == 1:
                    LOGGER.warning("%s read failed: %s", sensor_id, exc)
                continue
            if sample is None:
                continue
            self._publish(sensor_id, sample)
        return self._config.poll_interval_ms / 1000.0

    def _publish(self, sensor_id: str, sample: RangeSample) -> None:
        now_ns = self._clock_ns()
        self._sink.update_sensor(
            SensorReading(
                sensor_id=sensor_id,
                kind=SensorKind.TOF,
                timestamp_ns=now_ns,
                sector=self._sectors[sensor_id],
                distance_m=sample.distance_m,
                valid=sample.valid,
            )
        )
        self.health[sensor_id].record_ok(now_ns, sample.distance_m)

    def _maybe_restart(
        self, sensor_id: str, device: Vl53l0xDevice, health: SensorHealth, now_ns: int
    ) -> None:
        if now_ns < self._next_retry_ns.get(sensor_id, 0):
            return
        self._next_retry_ns[sensor_id] = now_ns + int(self._config.reinit_backoff_s * 1e9)
        try:
            device.restart()
        except Exception as exc:
            health.record_error(exc)
            LOGGER.warning("%s restart failed: %s", sensor_id, exc)
            return
        health.restarts += 1
        health.consecutive_errors = 0
        LOGGER.warning("%s restarted after repeated read errors", sensor_id)


class UltrasonicPoller(_Poller):
    def __init__(
        self,
        devices: Mapping[str, Hcsr04],
        sink: SensorSink,
        config: UltrasonicConfig,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        super().__init__("ultrasonic-poller", clock_ns)
        self._devices: List[Hcsr04] = list(devices.values())
        self._sink = sink
        self._config = config
        self._index = 0
        self.health: Dict[str, SensorHealth] = {name: SensorHealth(name) for name in devices}

    def step(self) -> float:
        if not self._devices:
            return 0.5
        device = self._devices[self._index % len(self._devices)]
        self._index += 1
        sensor_id = device.spec.sensor_id
        health = self.health[sensor_id]
        try:
            sample = device.measure()
        except Exception as exc:
            health.record_error(exc)
            if health.consecutive_errors == 1:
                LOGGER.warning("%s measurement failed: %s", sensor_id, exc)
            return self._config.spacing_ms / 1000.0
        now_ns = self._clock_ns()
        self._sink.update_sensor(
            SensorReading(
                sensor_id=sensor_id,
                kind=SensorKind.ULTRASONIC,
                timestamp_ns=now_ns,
                sector=device.spec.sector,
                distance_m=sample.distance_m,
                valid=sample.valid,
            )
        )
        health.record_ok(now_ns, sample.distance_m)
        return self._config.spacing_ms / 1000.0


class ImuPoller(_Poller):
    def __init__(
        self,
        imu: Mpu6050,
        monitor: MotionMonitor,
        poll_interval_ms: float,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        jolt: Optional[JoltDetector] = None,
        imu_sink: Optional[Any] = None,
    ) -> None:
        super().__init__("imu-poller", clock_ns)
        self._imu = imu
        self.monitor = monitor
        self.jolt = jolt
        self._imu_sink = imu_sink
        self._events: Deque[JoltEvent] = deque(maxlen=4)
        self._events_lock = threading.Lock()
        self._interval_s = poll_interval_ms / 1000.0
        self.health = SensorHealth("imu")

    def step(self) -> float:
        try:
            sample = self._imu.read()
        except Exception as exc:
            self.health.record_error(exc)
            if self.health.consecutive_errors == 1:
                LOGGER.warning("IMU read failed: %s", exc)
            # Fail towards "moving": the detector stays at full rate if the IMU is gone.
            self.monitor.is_still = False
            return max(self._interval_s, 0.2)
        now_ns = self._clock_ns()
        self.monitor.update(sample, now_ns)
        if self._imu_sink is not None and hasattr(self._imu_sink, "update_imu"):
            try:
                self._imu_sink.update_imu(sample, now_ns)
            except Exception:
                pass
        if self.jolt is not None:
            event = self.jolt.update(sample, now_ns)
            if event is not None:
                with self._events_lock:
                    self._events.append(event)
        self.health.record_ok(now_ns, None)
        return self._interval_s

    def pop_jolt(self) -> Optional[JoltEvent]:
        """The oldest jolt not yet handled, or None."""

        with self._events_lock:
            return self._events.popleft() if self._events else None


class SensorHub:
    """Owns the pollers and gives the runtime one place to start, stop and inspect them."""

    def __init__(
        self,
        pollers: Sequence[_Poller],
        tof: Optional[ToFPoller] = None,
        ultrasonic: Optional[UltrasonicPoller] = None,
        imu: Optional[ImuPoller] = None,
    ) -> None:
        self._pollers = list(pollers)
        self.tof = tof
        self.ultrasonic = ultrasonic
        self.imu = imu

    def pop_jolt(self) -> Optional[JoltEvent]:
        return self.imu.pop_jolt() if self.imu is not None else None

    @property
    def is_still(self) -> bool:
        return bool(self.imu is not None and self.imu.monitor.is_still)

    def start(self) -> None:
        for poller in self._pollers:
            poller.start()

    def stop(self) -> None:
        for poller in self._pollers:
            poller.stop()
        for poller in self._pollers:
            if poller.is_alive():
                poller.join(timeout=2.0)

    def crashed(self) -> List[str]:
        return [poller.name for poller in self._pollers if poller.crashed is not None]

    def health(self, now_ns: Optional[int] = None) -> Dict[str, Dict[str, object]]:
        now_ns = time.perf_counter_ns() if now_ns is None else now_ns
        report: Dict[str, Dict[str, object]] = {}
        for poller in (self.tof, self.ultrasonic):
            if poller is not None:
                for sensor_id, health in poller.health.items():
                    report[sensor_id] = health.as_dict(now_ns)
        if self.imu is not None:
            report["imu"] = self.imu.health.as_dict(now_ns)
            report["imu"]["still"] = self.imu.monitor.is_still
        return report
