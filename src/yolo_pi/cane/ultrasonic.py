"""HC-SR04 ultrasonic ranging on Raspberry Pi 5 GPIO.

Echo timing uses ``lgpio`` edge alerts, whose timestamps come from the kernel,
not from when Python happens to run the callback. A polled Python loop would add
tens of microseconds of jitter, which is several millimetres at 343 m/s; the
alert timestamps remove that.

The ECHO pin is a 5 V signal. It must reach the Pi through a level shifter or a
divider, never directly.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Protocol

from .bus import HardwareUnavailable
from .config import UltrasonicConfig, UltrasonicSensorSpec
from .tof import RangeSample

LOGGER = logging.getLogger("yolo_pi.cane.ultrasonic")

TRIGGER_PULSE_NS = 12_000  # the datasheet asks for at least 10 microseconds


class GpioBackend(Protocol):
    def claim_output(self, pin: int, level: int = 0) -> None: ...

    def write(self, pin: int, level: int) -> None: ...

    def claim_echo(self, pin: int, on_edge: Callable[[int, int], None]) -> None:
        """Call ``on_edge(level, timestamp_ns)`` on every edge of ``pin``."""

    def close(self) -> None: ...


class LgpioBackend:  # pragma: no cover - needs a Pi
    """GPIO through ``lgpio``, the library Raspberry Pi 5 uses for the RP1 header."""

    def __init__(self, chip: Optional[int] = None) -> None:
        try:
            import lgpio
        except ImportError as exc:
            raise HardwareUnavailable(
                "lgpio is not installed (sudo apt install python3-lgpio, or pip install lgpio)"
            ) from exc
        self._lgpio = lgpio
        self._callbacks: Dict[int, Any] = {}
        self._pins = []
        self._handle, self.chip = self._open_chip(chip)

    def _open_chip(self, chip: Optional[int]) -> Any:
        lgpio = self._lgpio
        # The Pi 5 header is "pinctrl-rp1". Depending on the kernel it appears
        # as gpiochip4 or gpiochip0, so look for the label instead of guessing.
        candidates = [chip] if chip is not None else [4, 0]
        fallback = None
        for number in candidates:
            try:
                handle = lgpio.gpiochip_open(number)
            except Exception:
                continue
            if chip is not None:
                return handle, number
            try:
                label = str(lgpio.gpio_get_chip_info(handle)).lower()
            except Exception:
                label = ""
            if "rp1" in label:
                if fallback is not None:
                    lgpio.gpiochip_close(fallback[0])
                return handle, number
            if fallback is None:
                fallback = (handle, number)
            else:
                lgpio.gpiochip_close(handle)
        if fallback is not None:
            return fallback
        raise HardwareUnavailable(
            "could not open a GPIO chip (is the user in the 'gpio' group?)"
        )

    def claim_output(self, pin: int, level: int = 0) -> None:
        result = self._lgpio.gpio_claim_output(self._handle, pin, level)
        if result < 0:
            raise HardwareUnavailable(f"could not claim GPIO{pin} as output (code {result})")
        self._pins.append(pin)

    def write(self, pin: int, level: int) -> None:
        self._lgpio.gpio_write(self._handle, pin, level)

    def claim_echo(self, pin: int, on_edge: Callable[[int, int], None]) -> None:
        lgpio = self._lgpio
        result = lgpio.gpio_claim_alert(self._handle, pin, lgpio.BOTH_EDGES, lgpio.SET_PULL_DOWN)
        if result < 0:
            raise HardwareUnavailable(f"could not claim GPIO{pin} for echo alerts (code {result})")
        self._pins.append(pin)

        def _callback(chip: int, gpio: int, level: int, timestamp: int) -> None:
            if level in (0, 1):  # level 2 is a watchdog tick, not an edge
                on_edge(level, timestamp)

        self._callbacks[pin] = lgpio.callback(self._handle, pin, lgpio.BOTH_EDGES, _callback)

    def close(self) -> None:
        for callback in self._callbacks.values():
            try:
                callback.cancel()
            except Exception:
                pass
        self._callbacks.clear()
        for pin in self._pins:
            try:
                self._lgpio.gpio_free(self._handle, pin)
            except Exception:
                pass
        self._pins.clear()
        try:
            self._lgpio.gpiochip_close(self._handle)
        except Exception:
            pass


def echo_to_distance_m(echo_ns: int, speed_of_sound_m_s: float) -> float:
    """Round-trip echo time to one-way distance: d = c * t / 2."""

    return echo_ns * 1e-9 * speed_of_sound_m_s / 2.0


class Hcsr04:
    """One HC-SR04: ``measure()`` fires it and waits for the echo."""

    def __init__(
        self,
        spec: UltrasonicSensorSpec,
        gpio: GpioBackend,
        config: UltrasonicConfig,
    ) -> None:
        self.spec = spec
        self._gpio = gpio
        self._config = config
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._rise_ns: Optional[int] = None
        self._fall_ns: Optional[int] = None
        self._started = False
        self.echo_count = 0
        self.no_echo_streak = 0

    def start(self) -> None:
        self._gpio.claim_output(self.spec.trig_gpio, 0)
        self._gpio.claim_echo(self.spec.echo_gpio, self._on_edge)
        self._started = True

    def _on_edge(self, level: int, timestamp_ns: int) -> None:
        with self._lock:
            if level == 1:
                self._rise_ns = timestamp_ns
                self._fall_ns = None
            elif level == 0 and self._rise_ns is not None and self._fall_ns is None:
                self._fall_ns = timestamp_ns
                self._done.set()

    def _fire(self) -> None:
        self._gpio.write(self.spec.trig_gpio, 1)
        end = time.perf_counter_ns() + TRIGGER_PULSE_NS
        while time.perf_counter_ns() < end:
            pass
        self._gpio.write(self.spec.trig_gpio, 0)

    def measure(self) -> RangeSample:
        if not self._started:
            raise RuntimeError("HC-SR04 measured before start()")
        config = self._config
        with self._lock:
            self._rise_ns = None
            self._fall_ns = None
            self._done.clear()
        self._fire()
        window_s = 2.0 * config.max_range_m / config.speed_of_sound_m_s + 0.012
        completed = self._done.wait(window_s)
        with self._lock:
            rise, fall = self._rise_ns, self._fall_ns
        if not completed or rise is None or fall is None or fall <= rise:
            # No echo inside the window: nothing within max range. This also
            # happens for soft or steeply angled targets, which the ToF sensors
            # and the camera have to cover (see the report's limitations).
            self.no_echo_streak += 1
            return RangeSample(config.max_range_m, True, True, None)
        distance = echo_to_distance_m(fall - rise, config.speed_of_sound_m_s)
        if distance >= config.max_range_m:
            self.no_echo_streak += 1
            return RangeSample(config.max_range_m, True, True, distance)
        self.echo_count += 1
        self.no_echo_streak = 0
        return RangeSample(max(distance, config.min_range_m), True, False, distance)

    def close(self) -> None:
        self._started = False
