"""HC-SR04 ultrasonic pair with strict time-division multiplexed pinging.

The cane carries two center-facing HC-SR04 sensors (TRIG/ECHO on GPIO23/24 and
GPIO27/22). Fired together they hear each other's bursts and invent obstacles,
so this driver alternates them and enforces an acoustic guard interval between
the end of one ping and the trigger of the next.

Distance uses the report's temperature-compensated speed of sound,
``v = 331.3 + 0.606 * T`` and ``d = v * dt / 2``. A listening timeout means the
sensor is healthy and nothing came back inside 25 ms (~4.2 m), which is
``valid=True, no_target=True``. A stuck-high echo line or a GPIO failure is a
real fault and is reported as ``valid=False``.

The GPIO backend is a protocol so the drivers, the guard timing, and the fault
paths all run on Linux x86_64 against a fake backend; ``LgpioBackend`` and
``GpioZeroBackend`` are the Raspberry Pi implementations.

Wiring note: an HC-SR04 drives ECHO at 5 V, so each ECHO line needs a level
shifter (or a 1k/2k divider) before it reaches an RP1 pin, which is not 5 V
tolerant.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from ..types import Sector
from .config import UltrasonicSensorSpec

SPEED_OF_SOUND_BASE_MPS = 331.3
SPEED_OF_SOUND_TEMP_COEFF = 0.606
TRIGGER_PULSE_S = 10e-6
DEFAULT_EDGE_START_TIMEOUT_S = 0.005


class GPIOException(RuntimeError):
    """Raised when a GPIO pin cannot be claimed, written, or read."""


class EchoStuckHighError(GPIOException):
    """Raised when the echo line rises but never falls inside the window."""


class GPIOBackend(Protocol):
    """The pin operations the ultrasonic driver needs from a GPIO provider."""

    def configure_output(self, pin: int) -> None:
        ...

    def configure_input(self, pin: int) -> None:
        ...

    def write(self, pin: int, value: bool) -> None:
        ...

    def read(self, pin: int) -> bool:
        ...

    def wait_for_edge(self, pin: int, level: bool, timeout_s: float) -> bool:
        ...

    def pulse_echo_s(self, pin: int, timeout_s: float) -> Optional[float]:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class UltrasonicMeasurement:
    """One range reading from one ultrasonic sensor."""

    sensor_id: str
    sector: Sector
    timestamp_ns: int
    distance_m: Optional[float]
    echo_s: Optional[float]
    valid: bool
    no_target: bool

    @property
    def clear(self) -> bool:
        return self.valid and self.no_target

    def as_dict(self) -> Dict[str, object]:
        return {
            "sensor_id": self.sensor_id,
            "sector": self.sector.value,
            "timestamp_ns": self.timestamp_ns,
            "distance_m": self.distance_m,
            "echo_s": self.echo_s,
            "valid": self.valid,
            "no_target": self.no_target,
        }


def speed_of_sound_mps(ambient_temp_c: float) -> float:
    """Report Eq. 2: ``v = 331.3 + 0.606 * T`` in m/s."""

    return SPEED_OF_SOUND_BASE_MPS + SPEED_OF_SOUND_TEMP_COEFF * float(ambient_temp_c)


class HCSR04PairDriver:
    """Two (or more) HC-SR04 sensors pinged in strict alternation."""

    def __init__(
        self,
        gpio: GPIOBackend,
        specs: Sequence[UltrasonicSensorSpec],
        guard_interval_ms: float = 35.0,
        timeout_ms: float = 25.0,
        ambient_temp_c: float = 22.0,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not specs:
            raise ValueError("at least one ultrasonic sensor is required")
        if guard_interval_ms <= 0:
            raise ValueError("guard_interval_ms must be positive")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if not -40.0 <= ambient_temp_c <= 85.0:
            raise ValueError("ambient_temp_c must be in [-40, 85]")
        seen: Dict[str, int] = {}
        for spec in specs:
            if spec.sensor_id in seen:
                raise ValueError(f"duplicate ultrasonic sensor_id {spec.sensor_id!r}")
            seen[spec.sensor_id] = spec.trigger_gpio
        self._gpio = gpio
        self.specs = tuple(specs)
        self.guard_interval_ms = float(guard_interval_ms)
        self.timeout_ms = float(timeout_ms)
        self.ambient_temp_c = float(ambient_temp_c)
        self.speed_of_sound_mps = speed_of_sound_mps(ambient_temp_c)
        self._clock = clock
        self._sleeper = sleeper
        self._guard_ns = int(round(self.guard_interval_ms * 1_000_000))
        self._timeout_ns = int(round(self.timeout_ms * 1_000_000))
        self._next_index = 0
        self._last_ping_end_ns: Optional[int] = None
        for spec in self.specs:
            self._gpio.configure_output(spec.trigger_gpio)
            self._gpio.write(spec.trigger_gpio, False)
            self._gpio.configure_input(spec.echo_gpio)

    @property
    def last_ping_end_ns(self) -> Optional[int]:
        return self._last_ping_end_ns

    def read_next(self, now_ns: Optional[int] = None) -> UltrasonicMeasurement:
        """Ping the next sensor in rotation, enforcing the acoustic guard."""

        index = self._next_index
        self._next_index = (index + 1) % len(self.specs)
        return self._ping(index, now_ns)

    def read_all_sequential(self, now_ns: Optional[int] = None) -> List[UltrasonicMeasurement]:
        """Ping every sensor in order, waiting out the guard between each."""

        measurements: List[UltrasonicMeasurement] = []
        first = True
        for index in range(len(self.specs)):
            measurements.append(self._ping(index, now_ns if first else None))
            first = False
        self._next_index = 0
        return measurements

    def close(self) -> None:
        self._gpio.close()

    # -- internals -------------------------------------------------------

    def _ping(self, index: int, now_ns: Optional[int]) -> UltrasonicMeasurement:
        spec = self.specs[index]
        start_ns = now_ns if now_ns is not None else self._clock()
        if self._enforce_guard(start_ns) and now_ns is None:
            start_ns = self._clock()

        try:
            self._pulse_trigger(spec)
            echo_s = self._gpio.pulse_echo_s(spec.echo_gpio, self.timeout_ms / 1000.0)
        except (OSError, GPIOException):
            # A stuck echo line or a dead pin is a real fault, never "clear".
            self._last_ping_end_ns = start_ns + self._timeout_ns
            return self._measurement(spec, start_ns, None, None, valid=False, no_target=False)

        # The listening window closes when the echo returns, or after the full
        # timeout when nothing comes back; the guard interval starts there.
        self._last_ping_end_ns = start_ns + (
            int(round(echo_s * 1e9)) if echo_s is not None else self._timeout_ns
        )
        if echo_s is None:
            return self._measurement(spec, start_ns, None, None, valid=True, no_target=True)

        distance_m = self.speed_of_sound_mps * echo_s / 2.0
        if distance_m > spec.max_range_m:
            return self._measurement(spec, start_ns, None, echo_s, valid=True, no_target=True)
        # Echoes below ``min_range_m`` stay distances: an object inside the
        # rated near field must never be reported as open space.
        return self._measurement(spec, start_ns, distance_m, echo_s, valid=True, no_target=False)

    @staticmethod
    def _measurement(
        spec: UltrasonicSensorSpec,
        timestamp_ns: int,
        distance_m: Optional[float],
        echo_s: Optional[float],
        *,
        valid: bool,
        no_target: bool,
    ) -> UltrasonicMeasurement:
        return UltrasonicMeasurement(
            sensor_id=spec.sensor_id,
            sector=spec.sector,
            timestamp_ns=timestamp_ns,
            distance_m=distance_m,
            echo_s=echo_s,
            valid=valid,
            no_target=no_target,
        )

    def _enforce_guard(self, start_ns: int) -> int:
        if self._last_ping_end_ns is None:
            return 0
        elapsed_ns = max(0, start_ns - self._last_ping_end_ns)
        remaining_ns = self._guard_ns - elapsed_ns
        if remaining_ns <= 0:
            return 0
        self._sleeper(remaining_ns / 1e9)
        return remaining_ns

    def _pulse_trigger(self, spec: UltrasonicSensorSpec) -> None:
        # A real HC-SR04 needs a >=10 us high pulse; busy-wait so the pulse is
        # not stretched by the scheduler granularity of time.sleep().
        self._gpio.write(spec.trigger_gpio, True)
        deadline = time.perf_counter() + TRIGGER_PULSE_S
        while time.perf_counter() < deadline:
            pass
        self._gpio.write(spec.trigger_gpio, False)


class LgpioBackend:
    """``lgpio`` backend; defaults to the Pi 5 RP1 chip and falls back to chip 0."""

    def __init__(self, chip: int = 4) -> None:
        if chip < 0:
            raise ValueError("gpiochip number must not be negative")
        try:
            import lgpio
        except ImportError as exc:  # pragma: no cover - exercised on a Pi
            raise RuntimeError(
                "lgpio is required for Raspberry Pi 5 GPIO access; "
                "install it with 'sudo apt install python3-lgpio' or 'pip install lgpio'"
            ) from exc
        self._lgpio = lgpio
        self.chip = chip
        self._handle: Optional[int] = None
        self._open_chip: Optional[int] = None
        self._claimed: List[int] = []

    @property
    def open_chip(self) -> Optional[int]:
        """Which gpiochip actually opened (4 on the Pi 5 RP1, else 0)."""

        return self._open_chip

    def _handle_or_open(self) -> int:
        if self._handle is not None:
            return self._handle
        failures = []
        for candidate in (self.chip, 0):
            try:
                handle = self._lgpio.gpiochip_open(candidate)
            except Exception as exc:  # noqa: BLE001 - lgpio raises plain exceptions
                failures.append(f"gpiochip{candidate}: {exc}")
                continue
            self._handle = handle
            self._open_chip = candidate
            return handle
        raise RuntimeError("cannot open a GPIO chip (" + "; ".join(failures) + ")")

    def configure_output(self, pin: int) -> None:
        handle = self._handle_or_open()
        try:
            self._lgpio.gpio_claim_output(handle, pin, 0)
        except Exception as exc:  # noqa: BLE001
            raise GPIOException(f"cannot claim GPIO{pin} as an output: {exc}") from exc
        self._claimed.append(pin)

    def configure_input(self, pin: int) -> None:
        handle = self._handle_or_open()
        try:
            self._lgpio.gpio_claim_input(handle, pin)
        except Exception as exc:  # noqa: BLE001
            raise GPIOException(f"cannot claim GPIO{pin} as an input: {exc}") from exc
        self._claimed.append(pin)

    def write(self, pin: int, value: bool) -> None:
        self._lgpio.gpio_write(self._handle_or_open(), pin, 1 if value else 0)

    def read(self, pin: int) -> bool:
        return bool(self._lgpio.gpio_read(self._handle_or_open(), pin))

    def wait_for_edge(self, pin: int, level: bool, timeout_s: float) -> bool:
        # Busy-wait: an echo pulse is tens of microseconds wide and sleeping
        # here would add scheduler jitter to every distance measurement.
        deadline = time.perf_counter() + timeout_s
        while self.read(pin) != level:
            if time.perf_counter() >= deadline:
                return False
        return True

    def pulse_echo_s(self, pin: int, timeout_s: float) -> Optional[float]:
        """Return the high pulse width, ``None`` if no echo, raise if stuck high."""

        if not self.wait_for_edge(pin, True, min(DEFAULT_EDGE_START_TIMEOUT_S, timeout_s)):
            return None
        rise = time.perf_counter()
        if not self.wait_for_edge(pin, False, timeout_s):
            raise EchoStuckHighError(f"GPIO{pin} echo stayed high beyond {timeout_s * 1000:.1f} ms")
        return time.perf_counter() - rise

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        for pin in self._claimed:
            try:
                self._lgpio.gpio_free(handle, pin)
            except Exception:  # noqa: BLE001 - best effort during shutdown
                pass
        self._claimed.clear()
        try:
            self._lgpio.gpiochip_close(handle)
        finally:
            self._handle = None
            self._open_chip = None


class GpioZeroBackend:
    """``gpiozero`` fallback for boards where ``lgpio`` is not installed."""

    def __init__(self) -> None:
        try:
            import gpiozero
        except ImportError as exc:  # pragma: no cover - exercised on a Pi
            raise RuntimeError(
                "gpiozero is required as a GPIO fallback; "
                "install it with 'sudo apt install python3-gpiozero' or 'pip install gpiozero'"
            ) from exc
        self._gpiozero = gpiozero
        self._outputs: Dict[int, object] = {}
        self._inputs: Dict[int, object] = {}

    def configure_output(self, pin: int) -> None:
        device = self._gpiozero.DigitalOutputDevice(pin)
        device.off()
        self._outputs[pin] = device

    def configure_input(self, pin: int) -> None:
        self._inputs[pin] = self._gpiozero.DigitalInputDevice(pin, pull_up=False)

    def write(self, pin: int, value: bool) -> None:
        device = self._outputs[pin]
        if value:
            device.on()
        else:
            device.off()

    def read(self, pin: int) -> bool:
        return bool(self._inputs[pin].value)

    def wait_for_edge(self, pin: int, level: bool, timeout_s: float) -> bool:
        device = self._inputs[pin]
        waiter = device.wait_for_active if level else device.wait_for_inactive
        return bool(waiter(timeout=timeout_s))

    def pulse_echo_s(self, pin: int, timeout_s: float) -> Optional[float]:
        if not self.wait_for_edge(pin, True, min(DEFAULT_EDGE_START_TIMEOUT_S, timeout_s)):
            return None
        rise = time.perf_counter()
        if not self.wait_for_edge(pin, False, timeout_s):
            raise EchoStuckHighError(f"GPIO{pin} echo stayed high beyond {timeout_s * 1000:.1f} ms")
        return time.perf_counter() - rise

    def close(self) -> None:
        for device in list(self._outputs.values()) + list(self._inputs.values()):
            device.close()
        self._outputs.clear()
        self._inputs.clear()


def create_gpio_backend(chip: int = 4) -> GPIOBackend:
    """Return ``lgpio`` on a Pi, else ``gpiozero``, with an actionable error."""

    try:
        return LgpioBackend(chip)
    except RuntimeError as lgpio_error:
        try:
            return GpioZeroBackend()
        except RuntimeError as gpiozero_error:
            raise RuntimeError(
                "no GPIO backend available: install 'lgpio' (preferred) or "
                f"'gpiozero' on the Raspberry Pi ({lgpio_error}; {gpiozero_error})"
            ) from gpiozero_error
