"""A simulated cane: scripted obstacles, fake devices, no hardware.

This runs the *same* pollers, fusion engine, smoother and haptic driver as the
real cane, with only the leaf devices swapped. It exists so the whole software
stack can be exercised on a laptop (``yolo-pi-cane run --simulate``) and in the
test suite, and so a student can see what each hazard pattern looks like before
touching the harness.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..types import BoundingBox, Detection, FrameDetections, Sector
from .config import CaneConfig, TofConfig, UltrasonicConfig, UltrasonicSensorSpec
from .hardware import HardwareSet
from .imu import ImuSample
from .tof import RangeSample

# What the sensors see, by key:
#   Sector            distance (m) of the nearest obstacle in that sector, or None when clear
#   OVERHEAD          distance (m) of something above the path (the overhead ultrasonic)
#   (FAR, Sector)     distance (m) of an object only the camera can see, in that sector
OVERHEAD = "overhead"
FAR = "far"
WorldState = Dict[object, Optional[float]]
World = Callable[[float], Optional[WorldState]]  # None means every sensor has failed

SIM_DURATION_S = 21.0
JOLT_START_S, JOLT_RAMP_S, JOLT_ANGLE_DEG = 14.0, 0.15, 50.0


def demo_world(t: float) -> Optional[WorldState]:
    """A 21-second walk: a far object, near hazards on each side, an overhead branch,
    a knock to the cane, then total sensor loss."""

    state: WorldState = {Sector.LEFT: None, Sector.CENTER: None, Sector.RIGHT: None}
    if 1.0 <= t < 4.0:
        state[(FAR, Sector.CENTER)] = 3.6 - (t - 1.0) / 3.0 * 1.2  # camera only: 3.6 m to 2.4 m
    if 4.5 <= t < 7.5:
        state[Sector.LEFT] = 1.4 - (t - 4.5) / 3.0 * 1.05  # 1.40 m closing to 0.35 m
    elif 8.0 <= t < 10.5:
        state[Sector.CENTER] = 1.2 - (t - 8.0) / 2.5 * 0.8  # 1.20 m closing to 0.40 m
    elif 11.0 <= t < 13.0:
        state[OVERHEAD] = 0.8
    elif 17.0 <= t < 18.5:
        state[Sector.RIGHT] = 0.8
    elif t >= 19.0:
        return None
    return state


def sim_tilt_deg(t: float) -> float:
    """Cane tilt: a sudden 50 degree knock at 14 s that eases back by 17 s."""

    if t < JOLT_START_S:
        return 0.0
    if t < JOLT_START_S + JOLT_RAMP_S:
        return JOLT_ANGLE_DEG * (t - JOLT_START_S) / JOLT_RAMP_S
    if t < 15.5:
        return JOLT_ANGLE_DEG
    if t < 17.0:
        return JOLT_ANGLE_DEG * (17.0 - t) / 1.5
    return 0.0


class SimClock:
    """Wall-clock seconds since construction."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def __call__(self) -> float:
        return time.monotonic() - self._start


class SimTofDevice:
    def __init__(self, sector: Sector, world: World, clock: Callable[[], float], config: TofConfig) -> None:
        self.sector, self._world, self._clock, self._config = sector, world, clock, config
        self._last_poll = -1.0

    def start(self) -> None:
        return None

    def restart(self) -> None:
        return None

    def close(self) -> None:
        return None

    def poll(self) -> Optional[RangeSample]:
        now = self._clock()
        if now - self._last_poll < 0.033:  # a new sample about every 33 ms, like the real chip
            return None
        state = self._world(now)
        if state is None:
            return None  # a dead sensor simply stops producing readings
        self._last_poll = now
        distance = state.get(self.sector)
        if distance is None:
            return RangeSample(self._config.clear_distance_m, True, True, 8190.0)
        return RangeSample(min(distance, self._config.clear_distance_m), True, False, distance * 1000.0)


class SimHcsr04:
    def __init__(
        self,
        spec: UltrasonicSensorSpec,
        world: World,
        clock: Callable[[], float],
        config: UltrasonicConfig,
    ) -> None:
        self.spec, self._world, self._clock, self._config = spec, world, clock, config
        self.echo_count = 0
        self.no_echo_streak = 0

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def measure(self) -> RangeSample:
        state = self._world(self._clock())
        if state is None:
            raise RuntimeError("simulated ultrasonic failure")
        distance = state.get(OVERHEAD if self.spec.role == "overhead" else self.spec.sector)
        if distance is None or distance >= self._config.max_range_m:
            self.no_echo_streak += 1
            return RangeSample(self._config.max_range_m, True, True, None)
        self.echo_count += 1
        self.no_echo_streak = 0
        return RangeSample(max(distance, self._config.min_range_m), True, False, distance)


@dataclass
class SimMotorDriver:
    name: str
    clock: Callable[[], float]
    events: List[Tuple[float, int]] = field(default_factory=list)

    def configure(self) -> None:
        return None

    def set_rtp(self, value: int) -> None:
        self.events.append((self.clock(), int(value)))

    def stop(self) -> None:
        self.set_rtp(0)

    def standby(self) -> None:
        self.set_rtp(0)


class SimImu:
    """A cane that lies still except for the scripted knock (a tilt about the X axis)."""

    def __init__(self, clock: Optional[Callable[[], float]] = None) -> None:
        self.who_am_i = 0x68
        self._clock = clock

    def read(self) -> ImuSample:
        if self._clock is None:
            return ImuSample((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 30.0)
        t = self._clock()
        theta = math.radians(sim_tilt_deg(t))
        rate_dps = (sim_tilt_deg(t + 0.005) - sim_tilt_deg(t - 0.005)) / 0.01
        return ImuSample((0.0, math.sin(theta), math.cos(theta)), (rate_dps, 0.0, 0.0), 30.0)


def build_simulated_hardware(
    config: CaneConfig,
    world: World = demo_world,
    clock: Optional[Callable[[], float]] = None,
) -> HardwareSet:
    clock = clock or SimClock()
    hardware = HardwareSet()
    for spec in config.tof.sensors:
        hardware.tof[spec.sensor_id] = SimTofDevice(spec.sector, world, clock, config.tof)  # type: ignore[assignment]
    for spec in config.ultrasonic.sensors:
        hardware.ultrasonic[spec.sensor_id] = SimHcsr04(spec, world, clock, config.ultrasonic)  # type: ignore[assignment]
    hardware.haptic = SimMotorDriver("haptic", clock)  # type: ignore[assignment]
    if config.imu.enabled:
        hardware.imu = SimImu(clock)  # type: ignore[assignment]
    return hardware


class SimVisionWorker(threading.Thread):
    """Publishes a 'pole' detection in whichever sector currently has a close obstacle."""

    def __init__(self, sink, world: World, clock: Callable[[], float]) -> None:
        super().__init__(name="sim-vision", daemon=True)
        self._sink, self._world, self._clock = sink, world, clock
        self._halt = threading.Event()
        self.ready = threading.Event()
        self.frames = 0
        self.restarts = 0

    def stats(self) -> Dict[str, object]:
        return {"frames": self.frames, "restarts": self.restarts, "simulated": True}

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        self.ready.set()
        frame_id = 0
        while not self._halt.wait(0.1):
            state = self._world(self._clock()) or {}
            detections = []
            centers = {Sector.LEFT: 100.0, Sector.CENTER: 320.0, Sector.RIGHT: 540.0}
            focal_px = 240.0 / math.tan(math.radians(48.8) / 2.0)
            for key, distance in state.items():
                if distance is None:
                    continue
                if isinstance(key, Sector) and distance < 1.5:
                    cx = centers[key]
                    detections.append(
                        Detection(3, "pole", 0.8, BoundingBox(cx - 40, 120, cx + 40, 420), key)
                    )
                elif isinstance(key, tuple) and key[0] == FAR:
                    cx = centers[key[1]]
                    height = min(440.0, focal_px * 1.7 / distance)  # a 1.7 m person
                    detections.append(
                        Detection(0, "person", 0.85,
                                  BoundingBox(cx - height / 6, 240 - height / 2, cx + height / 6, 240 + height / 2),
                                  key[1])
                    )
            now = time.perf_counter_ns()
            self._sink.update_vision(
                FrameDetections(
                    frame_id=frame_id, captured_ns=now, model_started_ns=now,
                    model_finished_ns=now, width=640, height=480,
                    detections=detections, backend="simulated", model_id="simulated",
                )
            )
            frame_id += 1
            self.frames += 1
