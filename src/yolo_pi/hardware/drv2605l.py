"""DRV2605L directional haptics: three 3 V ERM motors, one per warning sector.

Each motor sits behind its own TCA9548A channel (4 = left, 5 = center,
6 = right) and answers at 0x5A, so every write is a channel-selected
transaction. The controller drives them in real-time playback (RTP) mode by
default, turning a continuous intensity plus pulse period and duty cycle into a
deterministic, non-blocking on/off pattern per sector.

Two safety rules matter more than the vibration pattern:

* redundant RTP writes are skipped, because the 30 Hz loop already spends most
  of its I2C budget on the ToF sensors;
* :meth:`tick_watchdog` zeroes every motor when frames stop arriving, and
  :meth:`stop_all` zeroes each motor independently so one failed channel can
  never leave the other two buzzing.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..types import Sector
from .config import HapticMotorSpec
from .i2c import TCA9548AMux

LOGGER = logging.getLogger(__name__)

STATUS = 0x00
MODE = 0x01
RTP_INPUT = 0x02
LIBRARY_SEL = 0x03
WAVESEQ1 = 0x04
WAVESEQ2 = 0x05
GO = 0x0C
RATED_VOLTAGE = 0x16
OD_CLAMP = 0x17
FEEDBACK_CTRL = 0x1A
CONTROL3 = 0x1D

MODE_STANDBY_BIT = 0x40
MODE_INTERNAL_TRIGGER = 0x00
MODE_RTP = 0x05
LIBRARY_ERM_A = 0x01
#: CONTROL3 bit 3: unsigned RTP input (0 = signed, 1 = unsigned).
CONTROL3_RTP_UNSIGNED_DATA = 1 << 3
#: CONTROL3 bit 5: ERM open loop (cleared here so the ERM runs closed loop).
CONTROL3_ERM_OPEN_LOOP = 1 << 5
#: FEEDBACK_CTRL bit 7: 0 selects ERM, 1 selects LRA.
FEEDBACK_CTRL_LRA_BIT = 1 << 7

STATUS_OVER_CURRENT = 1 << 0
STATUS_OVER_TEMPERATURE = 1 << 1

MAX_RTP_VALUE = 127
HAPTIC_SECTORS = (Sector.LEFT, Sector.CENTER, Sector.RIGHT)
SUPPORTED_MODES = ("rtp", "library")


class DRV2605LError(RuntimeError):
    """Raised when a DRV2605L is missing, misconfigured, or unusable."""


@dataclass
class _MotorState:
    spec: HapticMotorSpec
    rtp: int = 0
    phase_origin_ns: Optional[int] = None
    waveform_active: bool = False
    standby: bool = False


class DRV2605LController:
    """Directional haptic sink for the left, center, and right ERM motors."""

    def __init__(
        self,
        mux: TCA9548AMux,
        specs: Sequence[HapticMotorSpec],
        mode: str = "rtp",
        watchdog_timeout_ms: float = 500.0,
    ) -> None:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {SUPPORTED_MODES}")
        if watchdog_timeout_ms <= 0:
            raise ValueError("watchdog_timeout_ms must be positive")
        specs = tuple(specs)
        if not specs:
            raise ValueError("at least one haptic motor spec is required")
        sectors = [spec.sector for spec in specs]
        if sorted(sector.value for sector in sectors) != sorted(
            sector.value for sector in HAPTIC_SECTORS
        ):
            raise ValueError("haptic specs must cover exactly the left, center, and right sectors")
        if len({spec.mux_channel for spec in specs}) != len(specs):
            raise ValueError("each haptic motor needs its own multiplexer channel")
        self._mux = mux
        self.specs = specs
        self.mode = mode
        self.watchdog_timeout_ms = float(watchdog_timeout_ms)
        self._watchdog_ns = int(round(self.watchdog_timeout_ms * 1_000_000))
        self._motors: Dict[Sector, _MotorState] = {
            spec.sector: _MotorState(spec=spec) for spec in specs
        }
        self._initialized = False
        self._closed = False
        self._last_update_ns: Optional[int] = None
        self._watchdog_tripped = False
        self._last_errors: Dict[Sector, str] = {}
        self._failed_sectors: Set[Sector] = set()
        self._lock = threading.RLock()

    # -- lifecycle -------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def watchdog_tripped(self) -> bool:
        return self._watchdog_tripped

    @property
    def last_errors(self) -> Dict[Sector, str]:
        return dict(self._last_errors)

    @property
    def failed_sectors(self) -> List[Sector]:
        return [sector for sector in HAPTIC_SECTORS if sector in self._failed_sectors]

    def initialize(self) -> None:
        """Configure every chip as a closed-loop 3 V ERM driver.

        A single dead channel must not cost the user the other two warning
        directions, so the working motors are configured and the failed ones are
        reported through :attr:`failed_sectors`. Only losing all three raises.
        """

        failures: List[str] = []
        for spec in self.specs:
            try:
                self._initialize_motor(spec)
            except (OSError, DRV2605LError) as exc:
                failures.append(f"{spec.motor_id}: {exc}")
                self._last_errors[spec.sector] = str(exc)
                self._failed_sectors.add(spec.sector)
                LOGGER.warning("%s: haptic channel unavailable: %s", spec.motor_id, exc)
            else:
                self._failed_sectors.discard(spec.sector)
        if len(self._failed_sectors) == len(self.specs):
            raise DRV2605LError("no haptic motor could be initialized: " + "; ".join(failures))
        self._initialized = True

    def _initialize_motor(self, spec: HapticMotorSpec) -> None:
        # Out of standby with the real-time playback register cleared first, so
        # a warm reboot can never resume the previous frame's vibration.
        self._write(spec, MODE, MODE_INTERNAL_TRIGGER)
        self._write(spec, RTP_INPUT, 0x00)
        self._write(spec, LIBRARY_SEL, LIBRARY_ERM_A)
        self._write(spec, WAVESEQ1, 0x00)
        self._write(spec, WAVESEQ2, 0x00)

        # ERM closed loop: clear the LRA select bit and the ERM open-loop bit.
        self._write(spec, FEEDBACK_CTRL, self._read(spec, FEEDBACK_CTRL) & ~FEEDBACK_CTRL_LRA_BIT)
        control3 = self._read(spec, CONTROL3)
        control3 &= ~CONTROL3_ERM_OPEN_LOOP
        control3 |= CONTROL3_RTP_UNSIGNED_DATA
        self._write(spec, CONTROL3, control3)

        self._write(spec, RATED_VOLTAGE, spec.rated_voltage)
        self._write(spec, OD_CLAMP, spec.od_clamp)

        mode_value = MODE_RTP if self.mode == "rtp" else MODE_INTERNAL_TRIGGER
        self._write(spec, MODE, mode_value)
        self._write(spec, RTP_INPUT, 0x00)

        state = self._motors[spec.sector]
        state.rtp = 0
        state.standby = False
        state.waveform_active = False
        state.phase_origin_ns = None

        status = self._read(spec, STATUS)
        if status & STATUS_OVER_CURRENT:
            LOGGER.warning("%s: DRV2605L reports over-current at startup", spec.motor_id)
        if status & STATUS_OVER_TEMPERATURE:
            LOGGER.warning("%s: DRV2605L reports over-temperature at startup", spec.motor_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.stop_all()
            self._closed = True

    # -- actuation -------------------------------------------------------

    def apply_sector_states(self, sector_commands: Mapping[Any, Any], now_ns: int) -> None:
        """Apply one control frame's per-sector haptic commands.

        ``sector_commands`` maps :class:`Sector` (or its string value) to any
        object exposing ``intensity`` (0..1), ``pulse_period_ms``,
        ``duty_cycle``, and ``waveform_id``. Fields are read duck-typed so the
        fusion layer owns the command type without a circular import.
        """

        with self._lock:
            self._require_initialized()
            commands = self._normalize(sector_commands)
            self._last_update_ns = int(now_ns)
            self._watchdog_tripped = False
            self._last_errors.clear()

            for sector in HAPTIC_SECTORS:
                state = self._motors[sector]
                command = commands.get(sector)
                intensity = _clamp_unit(_field(command, "intensity", 0.0))
                if intensity <= 0.0:
                    state.phase_origin_ns = None
                    self._drive(sector, 0, waveform_id=None)
                    continue
                period_ms = _optional_float(_field(command, "pulse_period_ms", None))
                duty_cycle = _optional_float(_field(command, "duty_cycle", 1.0))
                waveform_id = _field(command, "waveform_id", None)
                if period_ms and period_ms > 0.0 and duty_cycle is not None and duty_cycle < 1.0:
                    if state.phase_origin_ns is None:
                        state.phase_origin_ns = int(now_ns)
                    period_ns = int(round(period_ms * 1_000_000))
                    on_ns = int(round(_clamp_unit(duty_cycle) * period_ns))
                    elapsed_ns = max(0, int(now_ns) - state.phase_origin_ns)
                    is_on = (elapsed_ns % period_ns) < on_ns
                else:
                    state.phase_origin_ns = None
                    is_on = True
                target = _intensity_to_rtp(intensity) if is_on else 0
                self._drive(sector, target, waveform_id if is_on else None)

    def tick_watchdog(self, now_ns: int) -> bool:
        """Zero every motor when no frame has arrived inside the timeout."""

        with self._lock:
            if self._last_update_ns is None:
                return False
            silence_ns = int(now_ns) - self._last_update_ns
            if silence_ns <= self._watchdog_ns:
                return False
            LOGGER.warning(
                "haptic watchdog: no frame for %.1f ms, stopping all motors",
                silence_ns / 1_000_000.0,
            )
            errors = self._stop_motors()
            if errors:
                # A channel that failed to stop keeps the watchdog armed so the
                # next tick retries it instead of trusting a stale intensity.
                return True
            self._watchdog_tripped = True
            self._last_update_ns = None
            return True

    def stop_all(self) -> None:
        """Zero all three motors, isolating each channel's failure."""

        with self._lock:
            self._stop_motors()
            self._last_update_ns = None
            self._watchdog_tripped = False

    def _stop_motors(self) -> Dict[Sector, str]:
        errors: Dict[Sector, str] = {}
        self._last_errors.clear()
        for sector in HAPTIC_SECTORS:
            state = self._motors[sector]
            state.phase_origin_ns = None
            state.waveform_active = False
            try:
                if self.mode == "library":
                    self._write(state.spec, GO, 0x00)
                    self._write(state.spec, MODE, MODE_STANDBY_BIT)
                    state.standby = True
                self._write(state.spec, RTP_INPUT, 0x00)
            except (OSError, ValueError, DRV2605LError) as exc:
                errors[sector] = str(exc)
                self._last_errors[sector] = str(exc)
                # The cached value is no longer trustworthy; force the next
                # frame to rewrite the register once the channel recovers.
                state.rtp = -1
                continue
            state.rtp = 0
        return errors

    def _drive(self, sector: Sector, rtp_value: int, waveform_id: Optional[int]) -> None:
        state = self._motors[sector]
        if state.spec.sector in self._failed_sectors:
            return
        try:
            if self.mode == "rtp":
                if state.rtp != rtp_value:
                    self._write(state.spec, RTP_INPUT, rtp_value)
                    state.rtp = rtp_value
                return
            if rtp_value > 0:
                if state.standby:
                    self._write(state.spec, MODE, MODE_INTERNAL_TRIGGER)
                    state.standby = False
                if not state.waveform_active:
                    self._write(
                        state.spec,
                        WAVESEQ1,
                        int(waveform_id) if waveform_id is not None else LIBRARY_ERM_A,
                    )
                    self._write(state.spec, GO, 0x01)
                    state.waveform_active = True
            elif state.waveform_active:
                self._write(state.spec, GO, 0x00)
                state.waveform_active = False
        except (OSError, ValueError, DRV2605LError) as exc:
            self._last_errors[sector] = str(exc)
            state.rtp = -1

    # -- fusion sink compatibility ---------------------------------------

    def emit_frame(self, frame: Any) -> None:
        """Apply one ``DirectionalHapticSink`` frame (duck-typed, no imports).

        ``fusion_v2.DirectionalHapticFrame`` carries the per-sector commands in
        ``motors``; plain mappings and attribute bags are accepted as well.
        """

        timestamp_ns, commands = self._unpack_frame(frame)
        self.apply_sector_states(commands, timestamp_ns)

    def stop(self) -> None:
        self.stop_all()

    def emit(self, command: Any) -> None:
        """Adapt a single ``HapticCommand``-shaped object to the sector API."""

        timestamp_ns = _field(command, "timestamp_ns", None)
        if timestamp_ns is None:
            timestamp_ns = time.monotonic_ns()
        sectors = _field(command, "sectors", None) or ()
        commands = {sector: command for sector in self._normalize_sectors(sectors)}
        self.apply_sector_states(commands, int(timestamp_ns))

    # -- internals -------------------------------------------------------

    def _unpack_frame(self, frame: Any) -> Tuple[int, Dict[Sector, Any]]:
        timestamp_ns = _field(frame, "timestamp_ns", None)
        if timestamp_ns is None:
            timestamp_ns = _field(frame, "now_ns", None)
        commands: Any = None
        for attribute in ("motors", "sector_commands", "commands", "sector_haptics", "sectors"):
            candidate = _field(frame, attribute, None)
            if candidate is not None:
                commands = candidate
                break
        if commands is None and isinstance(frame, Mapping):
            commands = frame
        if timestamp_ns is None:
            timestamp_ns = time.monotonic_ns()
        return int(timestamp_ns), self._normalize(commands)

    def _normalize(self, commands: Any) -> Dict[Sector, Any]:
        if commands is None:
            return {}
        if isinstance(commands, Mapping):
            return {
                _coerce_sector(key): value
                for key, value in commands.items()
                if key is not None and _is_sector_key(key)
            }
        if isinstance(commands, Iterable) and not isinstance(commands, (str, bytes)):
            normalized: Dict[Sector, Any] = {}
            for command in commands:
                sector = _field(command, "sector", None)
                if sector is None:
                    continue
                normalized[_coerce_sector(sector)] = command
            return normalized
        raise TypeError("sector commands must be a mapping or an iterable of commands")

    def _normalize_sectors(self, sectors: Iterable[Any]) -> List[Sector]:
        return [_coerce_sector(sector) for sector in sectors if _is_sector_key(sector)]

    def _require_initialized(self) -> None:
        if self._closed:
            raise DRV2605LError("haptic controller is closed")
        if not self._initialized:
            raise DRV2605LError("initialize() must be called before applying haptic frames")

    def _read(self, spec: HapticMotorSpec, register: int) -> int:
        return self._mux.read_byte(spec.mux_channel, spec.i2c_address, register)

    def _write(self, spec: HapticMotorSpec, register: int, value: int) -> None:
        self._mux.write_byte(spec.mux_channel, spec.i2c_address, register, value)


def _field(source: Any, name: str, default: Any) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _clamp_unit(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return min(1.0, max(0.0, number))


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _intensity_to_rtp(intensity: float) -> int:
    return int(round(_clamp_unit(intensity) * MAX_RTP_VALUE))


def _is_sector_key(key: Any) -> bool:
    try:
        _coerce_sector(key)
    except (ValueError, TypeError):
        return False
    return True


def _coerce_sector(value: Any) -> Sector:
    if isinstance(value, Sector):
        return value
    if isinstance(value, str):
        for sector in Sector:
            if sector.value == value.strip().lower():
                return sector
        raise ValueError(f"unknown sector {value!r}")
    raise TypeError(f"sector must be a Sector or string, not {type(value).__name__}")
