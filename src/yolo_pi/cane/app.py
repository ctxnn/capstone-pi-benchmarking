"""Assembles the whole cane and runs it until asked to stop."""

from __future__ import annotations

import logging
import signal
import threading
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from ..fusion import HapticCommand
from .config import CaneConfig
from .hardware import HardwareSet, build_real_hardware
from .haptics import HapticDriver
from .imu import Calibration, JoltDetector, MotionMonitor, calibrate_gyro
from .runtime import CaneRuntime, LockedFusion
from .sensors import ImuPoller, SensorHub, ToFPoller, UltrasonicPoller, _Poller
from .sim import SimClock, SimVisionWorker, build_simulated_hardware, demo_world
from .vision import VisionWorker

LOGGER = logging.getLogger("yolo_pi.cane")


class LogHapticSink:
    """Prints pattern changes instead of driving motors (``--no-motors`` bench mode)."""

    def __init__(self) -> None:
        self._last: Optional[tuple] = None

    def emit(self, command: HapticCommand) -> None:
        key = (command.level, command.sectors, command.intensity, command.pulse_period_ms)
        if key != self._last:
            self._last = key
            LOGGER.info("haptic -> %s", command.as_dict())

    def burst(self, intensity: float, duration_ms: float) -> None:
        LOGGER.info("haptic burst -> intensity %.2f for %.0f ms", intensity, duration_ms)


@dataclass
class CaneSystem:
    config: CaneConfig
    hardware: HardwareSet
    fusion: LockedFusion
    haptics: Optional[HapticDriver]
    hub: SensorHub
    vision: Optional[Any]
    runtime: CaneRuntime
    calibration_ok: Optional[bool] = None

    def start(self) -> None:
        if self.haptics is not None:
            if self.config.haptics.startup_sweep:
                self.haptics.sweep()  # left (1 tap), centre (2), right (3): learn the code each start-up
            self.haptics.start()
        self.hub.start()
        if self.vision is not None:
            self.vision.start()

    def run(self, stop: threading.Event) -> None:
        self.runtime.run(stop)

    def close(self) -> None:
        if self.vision is not None:
            self.vision.stop()
        self.hub.stop()
        if self.haptics is not None:
            self.haptics.close()  # silences and parks every motor
        if self.vision is not None and self.vision.is_alive():
            self.vision.join(timeout=3.0)
        self.hardware.close()


def build_system(
    config: CaneConfig,
    *,
    simulate: bool = False,
    with_vision: bool = True,
    with_motors: bool = True,
    hardware: Optional[HardwareSet] = None,
    sim_clock: Optional[Callable[[], float]] = None,
    vision_factory: Optional[Callable[[LockedFusion], Any]] = None,
) -> CaneSystem:
    fusion = LockedFusion.from_config(config)
    clock = sim_clock or SimClock()
    if hardware is None:
        if simulate:
            hardware = build_simulated_hardware(config, demo_world, clock)
        else:
            hardware = build_real_hardware(config, with_motors=with_motors)
    for message in hardware.errors:
        LOGGER.warning("degraded: %s", message)

    haptics: Optional[HapticDriver] = None
    sink: Any
    if hardware.haptic is not None and with_motors:
        haptics = HapticDriver(hardware.haptic, config.haptics)
        sink = haptics
    else:
        sink = LogHapticSink()

    pollers: List[_Poller] = []
    tof_poller = None
    if hardware.tof:
        sectors = {spec.sensor_id: spec.sector for spec in config.tof.sensors}
        tof_poller = ToFPoller(hardware.tof, sectors, fusion, config.tof)  # type: ignore[arg-type]
        pollers.append(tof_poller)
    us_poller = None
    if hardware.ultrasonic:
        us_poller = UltrasonicPoller(hardware.ultrasonic, fusion, config.ultrasonic)  # type: ignore[arg-type]
        pollers.append(us_poller)

    imu_poller = None
    calibration_ok: Optional[bool] = None
    if hardware.imu is not None:
        try:
            calibration = calibrate_gyro(hardware.imu, config.imu)
        except Exception as exc:  # a flaky IMU at boot must not stop the cane starting
            LOGGER.warning("gyro calibration failed (%s); continuing with zero bias", exc)
            calibration = Calibration((0.0, 0.0, 0.0), False, 0)
        calibration_ok = calibration.calibrated
        if calibration.calibrated:
            LOGGER.info("gyro bias (deg/s) %s after %d attempt(s)",
                        tuple(round(v, 3) for v in calibration.bias_dps), calibration.attempts)
        else:
            LOGGER.warning("gyro not calibrated (cane was moving); using zero bias")
        monitor = MotionMonitor(config.imu, calibration.bias_dps)
        jolt = JoltDetector(config.imu, calibration.bias_dps) if config.imu.jolt_enabled else None
        imu_poller = ImuPoller(
            hardware.imu, monitor, config.imu.poll_interval_ms, jolt=jolt, imu_sink=fusion
        )
        pollers.append(imu_poller)

    hub = SensorHub(pollers, tof=tof_poller, ultrasonic=us_poller, imu=imu_poller)

    vision: Optional[Any] = None
    if with_vision and config.vision.enabled:
        if vision_factory is not None:
            vision = vision_factory(fusion)
        elif simulate:
            vision = SimVisionWorker(fusion, demo_world, clock)
        else:
            vision = VisionWorker(config.vision, fusion, is_still=lambda: hub.is_still)

    runtime = CaneRuntime(
        config,
        fusion,
        sink,
        health=hub.health,
        imu_events=hub.pop_jolt,
        vision_stats=(vision.stats if vision is not None else (lambda: {})),
        hardware_ok=lambda: not hub.crashed(),
    )
    return CaneSystem(config, hardware, fusion, haptics, hub, vision, runtime, calibration_ok)


def run_until_signal(system: CaneSystem, duration_s: float = 0.0) -> int:
    """Start the cane and run until SIGINT/SIGTERM (or ``duration_s``), parking the motors on exit."""

    stop = threading.Event()

    def _request_stop(signum: int, _frame: Any) -> None:
        LOGGER.info("received signal %d; shutting down", signum)
        stop.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    exit_code = 0
    timer: Optional[threading.Timer] = None
    try:
        system.start()
        if duration_s > 0:
            timer = threading.Timer(duration_s, stop.set)
            timer.daemon = True
            timer.start()
        LOGGER.info("cane running: %s", system.config.describe())
        system.run(stop)
    except Exception:
        LOGGER.exception("cane runtime stopped unexpectedly")
        exit_code = 1
    finally:
        if timer is not None:
            timer.cancel()
        system.close()
    return exit_code
