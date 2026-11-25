"""Haptic output: one DRV2605L driving three ERM motors in parallel.

``command_for`` decides what to say (intensity, urgency, direction) and
``patterns`` turns that into a rhythm. This module plays the rhythm on the one
driver on a background thread, keeps it safe (a watchdog silences the motors if
commands stop, and every shutdown path parks them) and recovers from I2C errors.
``burst`` buzzes once for the IMU's sudden-tilt pulse.

The DRV2605L runs in real-time-playback (RTP) mode with open-loop ERM drive, so
the software sets an amplitude directly and no auto-calibration step is needed.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional, Protocol, Tuple

from ..fusion import HapticCommand, HazardLevel
from ..types import Sector
from .bus import RegisterDevice
from .config import HapticsConfig
from .patterns import Cycle, amplitude_at, build_cycle, cycle_length, one_shot

LOGGER = logging.getLogger("yolo_pi.cane.haptics")

REG_STATUS = 0x00
REG_MODE = 0x01
REG_RTP = 0x02
REG_OD_CLAMP = 0x17
REG_FEEDBACK = 0x1A
REG_CONTROL3 = 0x1D

MODE_RTP = 0x05
MODE_STANDBY = 0x40
MODE_RESET = 0x80
FEEDBACK_N_ERM_LRA = 0x80  # 0 selects an ERM motor
CONTROL3_ERM_OPEN_LOOP = 0x20
CONTROL3_RTP_UNSIGNED = 0x08  # 0 selects the signed RTP format, full scale = 127
RTP_FULL_SCALE = 127
OD_CLAMP_VOLTS_PER_LSB = 0.02118  # DRV2605L datasheet, open-loop ERM drive
KNOWN_DEVICE_IDS = {3, 4, 5, 6}  # DRV2605, DRV2604, DRV2604L, DRV2605L


def od_clamp_for(voltage: float) -> int:
    """Open-loop drive voltage to the OD_CLAMP register value."""

    return max(0, min(255, int(round(voltage / OD_CLAMP_VOLTS_PER_LSB))))


def amplitude_to_rtp(intensity: float, floor: float) -> int:
    """Map a 0..1 intensity to an RTP value.

    Zero stays zero. Any other intensity is lifted to ``floor`` so the weak
    sensor-fault pattern still turns the motor, while keeping the order
    fault < warning < critical.
    """

    if intensity <= 0.0:
        return 0
    level = floor + (1.0 - floor) * min(intensity, 1.0)
    return max(1, min(RTP_FULL_SCALE, int(round(level * RTP_FULL_SCALE))))


class MotorDriver(Protocol):
    def configure(self) -> None: ...

    def set_rtp(self, value: int) -> None: ...

    def stop(self) -> None: ...


class Drv2605Erm:
    """Minimal DRV2605L driver: ERM, open loop, real-time playback."""

    def __init__(
        self,
        device: RegisterDevice,
        motor_voltage_v: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._device = device
        self._voltage = motor_voltage_v
        self._sleep = sleep
        self.device_id: Optional[int] = None

    def configure(self) -> None:
        device = self._device
        self.device_id = device.read_reg(REG_STATUS) >> 5
        if self.device_id not in KNOWN_DEVICE_IDS:
            LOGGER.warning("unexpected DRV2605 device id %d at %s", self.device_id, hex(device.address))
        device.write_reg(REG_MODE, MODE_RESET)
        self._sleep(0.02)
        device.write_reg(REG_MODE, 0x00)  # out of standby, internal trigger
        device.write_reg(REG_FEEDBACK, device.read_reg(REG_FEEDBACK) & ~FEEDBACK_N_ERM_LRA & 0xFF)
        control3 = device.read_reg(REG_CONTROL3)
        control3 = (control3 | CONTROL3_ERM_OPEN_LOOP) & ~CONTROL3_RTP_UNSIGNED & 0xFF
        device.write_reg(REG_CONTROL3, control3)
        device.write_reg(REG_OD_CLAMP, od_clamp_for(self._voltage))
        device.write_reg(REG_RTP, 0)
        device.write_reg(REG_MODE, MODE_RTP)

    def set_rtp(self, value: int) -> None:
        self._device.write_reg(REG_RTP, max(0, min(RTP_FULL_SCALE, int(value))))

    def stop(self) -> None:
        self._device.write_reg(REG_RTP, 0)

    def standby(self) -> None:
        self._device.write_reg(REG_RTP, 0)
        self._device.write_reg(REG_MODE, MODE_STANDBY)


def play_pattern_once(
    driver: MotorDriver,
    command: HapticCommand,
    config: HapticsConfig,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Play a command's pattern once and stop. Raises if the driver fails (used by the bench tools)."""

    try:
        for amplitude, seconds in one_shot(build_cycle(command, config)):
            driver.set_rtp(amplitude_to_rtp(amplitude, config.amplitude_floor))
            sleep(seconds)
    finally:
        try:
            driver.stop()
        except Exception:
            pass


class HapticDriver:
    """A ``HapticSink`` that plays the current pattern on the single DRV2605L."""

    MAX_ON_S = 0.95  # longest legitimate stretch: 0.6 s fault pulse + 0.3 s jolt burst
    FORCED_REST_S = 0.1

    def __init__(
        self,
        driver: MotorDriver,
        config: HapticsConfig,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._driver = driver
        self._config = config
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cycle: Cycle = ()
        self._cycle_len = 0.0
        self._shape: Optional[Tuple[object, ...]] = None
        self._start = 0.0
        self._last_emit = 0.0
        self._timed_out = False
        self._on_since: Optional[float] = None
        self._burst_until = 0.0
        self._burst_value = 0
        self._level = 0
        self._consecutive_errors = 0
        self._last_reconfigure = 0.0
        self.write_count = 0
        self.error_count = 0

    # -- life cycle ------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="haptics", daemon=True)
        self._thread.start()

    def play(self, command: HapticCommand) -> None:
        """Play one command's pattern once, blocking, then silence. Run before ``start()``."""

        try:
            play_pattern_once(self._driver, command, self._config, self._sleep)
        except Exception as exc:
            LOGGER.error("pattern playback failed: %s", exc)
        self.all_off(attempts=2)

    def sweep(self, intensity: float = 0.7, pause_s: float = 0.5) -> None:
        """Play left, centre and right once, so the user hears the three directions at start-up."""

        for sector in (Sector.LEFT, Sector.CENTER, Sector.RIGHT):
            self.play(HapticCommand(0, HazardLevel.WARNING, (sector,), intensity, None, "startup"))
            self._sleep(pause_s)

    def all_off(self, attempts: int = 4) -> None:
        """Stop the motors, retrying: a DRV2605L keeps its last drive value if the stop write is lost."""

        for attempt in range(attempts):
            try:
                self._driver.stop()
                self._level = 0
                return
            except Exception as exc:
                if attempt == attempts - 1:
                    LOGGER.error("could not stop the motors after %d tries: %s", attempts, exc)
                else:
                    self._sleep(0.02)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        self.all_off()
        standby = getattr(self._driver, "standby", None)
        if standby is not None:
            try:
                standby()
            except Exception:
                pass

    # -- commands --------------------------------------------------------------

    def emit(self, command: HapticCommand) -> None:
        """Set the current pattern. Sending the same pattern again keeps its phase.

        A change of direction restarts the rhythm; a change of tempo or strength
        keeps the position in the cycle, so a hazard drifting closer speeds the
        taps up smoothly instead of restarting them on every reading.
        """

        cycle = build_cycle(command, self._config)
        shape = (
            tuple(sorted(sector.value for sector in command.sectors)),
            getattr(command, "source", "range"),
        ) if cycle else None
        length = cycle_length(cycle)
        now = self._clock()
        with self._lock:
            if not cycle or not self._cycle:
                self._start = now
            elif shape != self._shape:
                # A new direction restarts the code, but never in the middle of a
                # tap: a hazard flickering between sectors would otherwise
                # restart the first tap on every reading and hold the motor on.
                if amplitude_at(self._cycle, now - self._start) <= 0.0:
                    self._start = now
            elif cycle != self._cycle and self._cycle_len > 0:
                fraction = ((now - self._start) % self._cycle_len) / self._cycle_len
                self._start = now - fraction * length
            self._shape, self._cycle, self._cycle_len = shape, cycle, length
            self._last_emit = now
            self._timed_out = False
        self._wake.set()

    def burst(self, intensity: float, duration_ms: float) -> None:
        """Buzz solidly for ``duration_ms`` on top of whatever pattern is playing (the IMU's pulse)."""

        value = amplitude_to_rtp(intensity, self._config.amplitude_floor)
        with self._lock:
            self._burst_value = value
            self._burst_until = self._clock() + duration_ms / 1000.0
        self._wake.set()

    # -- pattern thread --------------------------------------------------------

    def target(self, now: float) -> int:
        """The RTP value that should be on the motors at time ``now``."""

        with self._lock:
            cycle, start = self._cycle, self._start
            idle_s = now - self._last_emit
            bursting = now < self._burst_until
            burst_value = self._burst_value
        if cycle and idle_s > self._config.command_timeout_s:
            # The runtime has stopped sending commands. A stale pattern must
            # not keep buzzing, so fall silent until it recovers.
            if not self._timed_out:
                LOGGER.error("no haptic command for %.1f s; motors stopped", idle_s)
                self._timed_out = True
            cycle = ()
        value = amplitude_to_rtp(amplitude_at(cycle, now - start), self._config.amplitude_floor)
        return max(value, burst_value) if bursting else value

    def _run(self) -> None:
        while not self._stop.is_set():
            now = self._clock()
            desired = self.target(now)
            # Backstop: no legitimate pattern holds the motor on this long, so
            # whatever the cause, force a rest and let the pattern carry on.
            if desired > 0:
                if self._on_since is None:
                    self._on_since = now
                elif now - self._on_since > self.MAX_ON_S:
                    desired = 0
                    if now - self._on_since > self.MAX_ON_S + self.FORCED_REST_S:
                        self._on_since = None
            else:
                self._on_since = None
            self._apply(desired)
            self._wake.wait(0.004)
            self._wake.clear()

    def _apply(self, desired: int) -> None:
        if desired == self._level:
            return
        try:
            self._driver.set_rtp(desired)
        except Exception as exc:
            self.error_count += 1
            self._consecutive_errors += 1
            self._level = -1  # unknown: write again on the next pass
            if self._consecutive_errors in (1, 10, 100):
                LOGGER.error("haptic write failed (%d in a row): %s", self._consecutive_errors, exc)
            self._try_reconfigure()
            return
        self._level = desired
        self._consecutive_errors = 0
        self.write_count += 1

    def _try_reconfigure(self) -> None:
        now = self._clock()
        if self._consecutive_errors < 3 or now - self._last_reconfigure < 1.0:
            return
        self._last_reconfigure = now
        try:
            self._driver.configure()
            LOGGER.warning("re-initialised the DRV2605L")
        except Exception as exc:
            LOGGER.error("could not re-initialise the DRV2605L: %s", exc)

    def health(self) -> Dict[str, object]:
        return {
            "writes": self.write_count,
            "errors": self.error_count,
            "consecutive_errors": self._consecutive_errors,
        }
