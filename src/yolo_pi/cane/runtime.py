"""The cane's main loop: fuse the newest readings, decide, and drive the motors.

Threads in a running cane:

* ToF, ultrasonic and IMU pollers write readings into the fusion engine;
* the vision worker writes detections into the same engine;
* this loop evaluates the engine at ``loop_hz`` and sends the result to the
  haptic driver, which pulses the motors on its own thread.

The fusion engine itself is not thread-safe, so it is wrapped in a lock here
rather than changing the tested ``fusion.py``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from ..fusion import (
    FusionConfig,
    HapticCommand,
    HapticSink,
    HazardDecision,
    HazardLevel,
    SensorReading,
)
from ..types import FrameDetections
from .config import CaneConfig, PriorityConfig, RuntimeConfig
from .imu import JoltEvent
from .priority import CaneDecision, PriorityFusion, command_for

LOGGER = logging.getLogger("yolo_pi.cane.runtime")


class LockedFusion:
    """Thread-safe facade over :class:`PriorityFusion` (many writers, one reader)."""

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        priority: Optional[PriorityConfig] = None,
        overhead_ids: Iterable[str] = (),
        *,
        ground_ids: Iterable[str] = (),
        sensor_pitch_offsets_deg: Optional[Dict[str, float]] = None,
    ) -> None:
        self._engine = PriorityFusion(
            config,
            priority,
            overhead_ids,
            ground_ids=ground_ids,
            sensor_pitch_offsets_deg=sensor_pitch_offsets_deg,
        )
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: CaneConfig) -> "LockedFusion":
        overhead = [s.sensor_id for s in config.ultrasonic.sensors if s.role == "overhead"]
        ground = [s.sensor_id for s in config.ultrasonic.sensors if s.role == "ground"]
        offsets = {
            s.sensor_id: float(getattr(s, "pitch_offset_deg", 0.0))
            for s in config.ultrasonic.sensors
            if hasattr(s, "pitch_offset_deg")
        }
        return cls(
            config.fusion,
            config.priority,
            overhead,
            ground_ids=ground,
            sensor_pitch_offsets_deg=offsets,
        )

    def update_imu(self, sample: Any, now_ns: Optional[int] = None) -> Any:
        with self._lock:
            return self._engine.update_imu(sample, now_ns)

    def update_sensor(self, reading: SensorReading) -> None:
        with self._lock:
            self._engine.update_sensor(reading)

    def update_sensors(self, readings: Iterable[SensorReading]) -> None:
        with self._lock:
            self._engine.update_sensors(readings)

    def update_vision(self, detections: FrameDetections) -> None:
        with self._lock:
            self._engine.update_vision(detections)

    def evaluate(self, now_ns: int) -> CaneDecision:
        with self._lock:
            return self._engine.evaluate(now_ns)


_HELD_LEVELS = (HazardLevel.WARNING, HazardLevel.CRITICAL)


class HazardSmoother:
    """Escalate at once, release slowly.

    A target hovering at exactly the warning or critical distance would
    otherwise flip the pattern on every reading. The smoother never delays a
    rise in severity, but holds WARNING or CRITICAL for ``hold_ms`` after the
    last reading that justified it before stepping down towards CLEAR.
    """

    def __init__(self, hold_ms: float) -> None:
        self._hold_ns = int(hold_ms * 1e6)
        self._current: Optional[HazardDecision] = None
        self._confirmed_ns = 0

    def apply(self, decision: HazardDecision) -> HazardDecision:
        current = self._current
        now = decision.timestamp_ns
        if current is not None and current.level in _HELD_LEVELS:
            if decision.level == current.level:
                self._current, self._confirmed_ns = decision, now
                return decision
            stepping_down = decision.level in (HazardLevel.CLEAR, HazardLevel.WARNING) and (
                decision.level.value < current.level.value
            )
            if stepping_down and now - self._confirmed_ns < self._hold_ns:
                return replace(current, timestamp_ns=now)
        self._current, self._confirmed_ns = decision, now
        return decision


def read_cpu_temperature_c() -> Optional[float]:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except (OSError, ValueError):
        return None


def read_throttled_flags() -> Optional[str]:
    """``vcgencmd get_throttled``: non-zero means under-voltage or thermal limits were hit."""

    try:
        output = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1.0, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return output.split("=")[-1] if "=" in output else None


class EventLog:
    """Append-only JSON lines: one record per hazard change plus periodic status."""

    def __init__(self, path: Optional[str]) -> None:
        self._handle = None
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._handle = target.open("a", encoding="utf-8", buffering=1)

    def write(self, kind: str, payload: Dict[str, Any]) -> None:
        if self._handle is None:
            return
        record = {"kind": kind, "wall_time": round(time.time(), 3), **payload}
        try:
            self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:  # a full or failing SD card must not stop the cane
            self._handle = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@dataclass
class LoopStats:
    iterations: int = 0
    overruns: int = 0
    emits: int = 0
    jolts: int = 0
    max_loop_ms: float = 0.0


class CaneRuntime:
    def __init__(
        self,
        config: CaneConfig,
        fusion: LockedFusion,
        sink: HapticSink,
        *,
        health: Callable[[], Dict[str, Any]] = lambda: {},
        imu_events: Callable[[], Optional[JoltEvent]] = lambda: None,
        vision_stats: Callable[[], Dict[str, Any]] = lambda: {},
        hardware_ok: Callable[[], bool] = lambda: True,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        sleep: Callable[[float], None] = time.sleep,
        event_log: Optional[EventLog] = None,
    ) -> None:
        self._config = config
        self._runtime: RuntimeConfig = config.runtime
        self._fusion = fusion
        self._sink = sink
        self._health = health
        self._imu_events = imu_events
        self._vision_stats = vision_stats
        self._hardware_ok = hardware_ok
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._events = event_log if event_log is not None else EventLog(config.runtime.event_log)
        self._smoother = HazardSmoother(config.runtime.release_hold_ms)
        self.stats = LoopStats()
        self.last_decision: Optional[HazardDecision] = None
        self.last_command: Optional[HapticCommand] = None

    def tick(self, now_ns: Optional[int] = None) -> HapticCommand:
        """One fusion pass. Public so tests and the simulator can step it directly."""

        now_ns = self._clock_ns() if now_ns is None else now_ns
        decision = self._smoother.apply(self._fusion.evaluate(now_ns))
        command = command_for(decision, self._config.priority, self._config.fusion, self._config.haptics)
        previous = self.last_decision
        self.last_decision, self.last_command = decision, command
        self._sink.emit(command)
        self.stats.emits += 1
        self._handle_jolt(now_ns)
        if previous is None or (
            previous.level, previous.sectors, getattr(previous, "source", None)
        ) != (decision.level, decision.sectors, getattr(decision, "source", None)):
            self._events.write("decision", decision.as_dict())
            LOGGER.info(
                "%s (%s) sectors=%s nearest=%s labels=%s",
                decision.level.name, getattr(decision, "source", "range"), [s.value for s in decision.sectors],
                None if decision.nearest_distance_m is None else round(decision.nearest_distance_m, 2),
                list(decision.semantic_labels),
            )
        return command

    def _handle_jolt(self, now_ns: int) -> None:
        """Pulse every motor once per sudden tilt. Nothing else the IMU sees vibrates."""

        event = self._imu_events()
        if event is None:
            return
        burst = getattr(self._sink, "burst", None)
        if callable(burst):
            burst(self._config.imu.jolt_intensity, self._config.imu.jolt_pulse_ms)
        self.stats.jolts += 1
        self._events.write("jolt", {"angle_deg": round(event.angle_deg, 1), "timestamp_ns": now_ns})
        LOGGER.info("jolt: tilt changed %.0f degrees suddenly; pulsing all motors", event.angle_deg)

    def status(self, now_ns: Optional[int] = None) -> Dict[str, Any]:
        now_ns = self._clock_ns() if now_ns is None else now_ns
        decision = self.last_decision
        return {
            "level": None if decision is None else decision.level.name.lower(),
            "nearest_distance_m": None if decision is None else decision.nearest_distance_m,
            "fresh_range_sensors": None if decision is None else decision.fresh_range_sensor_count,
            "loop": {
                "iterations": self.stats.iterations,
                "overruns": self.stats.overruns,
                "max_loop_ms": round(self.stats.max_loop_ms, 2),
                "jolts": self.stats.jolts,
            },
            "source": getattr(decision, "source", None),
            "sensors": self._health(),
            "vision": self._vision_stats(),
            "cpu_temp_c": read_cpu_temperature_c(),
            "throttled": read_throttled_flags(),
        }

    def run(self, stop: threading.Event) -> None:
        period_ns = int(1e9 / self._runtime.loop_hz)
        status_ns = int(self._runtime.status_interval_s * 1e9)
        next_tick = self._clock_ns()
        next_status = next_tick + status_ns
        try:
            while not stop.is_set():
                started = self._clock_ns()
                if not self._hardware_ok():
                    raise RuntimeError("a sensor thread crashed; stopping so the service restarts")
                self.tick(started)
                self.stats.iterations += 1
                finished = self._clock_ns()
                self.stats.max_loop_ms = max(self.stats.max_loop_ms, (finished - started) / 1e6)
                if started >= next_status:
                    status = self.status(started)
                    self._events.write("status", status)
                    LOGGER.info("status: %s", json.dumps(status, sort_keys=True, default=str))
                    next_status = started + status_ns
                next_tick += period_ns
                lag_ns = finished - next_tick
                if lag_ns > period_ns:  # fell behind: do not try to catch up in a burst
                    self.stats.overruns += 1
                    next_tick = finished
                self._sleep(max(0.0, (next_tick - self._clock_ns()) / 1e9))
        finally:
            self._events.close()
