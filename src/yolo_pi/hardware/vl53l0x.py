"""Pure-Python VL53L0X time-of-flight driver for the cane's three sectors.

The driver talks to the sensor through :class:`TCA9548AMux` only, so the three
sensors that share address 0x29 stay isolated on channels 0, 1, and 2. Range
decoding follows the report's safety contract:

* a valid ranging (register status 11) up to 2000 mm is a real distance;
* an out-of-range value (beyond 2000 mm, including the 8190/8191 sentinels) or
  a phase/sigma/signal failure means the sensor is healthy and sees open space,
  so the reading is ``valid=True, no_target=True``;
* an I2C error, a status indicating a VCSEL/hardware failure, or a ranging
  timeout is a real fault: ``valid=False, no_target=False``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..types import Sector
from .config import ToFSensorSpec
from .i2c import TCA9548AMux

# Register map (ST VL53L0X API / Pololu Arduino library).
SYSRANGE_START = 0x00
SYSTEM_SEQUENCE_CONFIG = 0x01
SYSTEM_INTERRUPT_CONFIG_GPIO = 0x0A
SYSTEM_INTERRUPT_CLEAR = 0x0B
RESULT_INTERRUPT_STATUS = 0x13
RESULT_RANGE_STATUS = 0x14
MSRC_CONFIG_CONTROL = 0x60
MSRC_CONFIG_TIMEOUT_MACROP = 0x46
FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT = 0x44
PRE_RANGE_CONFIG_VCSEL_PERIOD = 0x50
PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI = 0x51
FINAL_RANGE_CONFIG_VCSEL_PERIOD = 0x70
FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI = 0x71
VHV_CONFIG_PAD_SCL_SDA__EXTSUP_HV = 0x89
IDENTIFICATION_MODEL_ID = 0xC0
GPIO_HV_MUX_ACTIVE_HIGH = 0x84
OSC_CALIBRATE_VAL = 0xF8
SYSTEM_INTERMEASUREMENT_PERIOD = 0x04

MUX_PAGE_CONTROL = 0xFF
STOP_VARIABLE_REGISTER = 0x91

MODEL_ID_VL53L0X = 0xEE
RANGE_VALID_STATUS = 11
#: ST PAL "NONE" statuses: no target inside the ranging window.
NO_TARGET_STATUSES = frozenset({0, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15})
#: ST PAL hardware failures: VCSEL continuity/watchdog or missing VHV value.
HARDWARE_FAULT_STATUSES = frozenset({1, 2, 3})

MIN_VALID_MM = 30
MAX_VALID_MM = 2000
SENTINEL_MM = (8190, 8191)

# Pololu's VL53L0X::setMeasurementTimingBudget overheads, in microseconds.
_START_OVERHEAD_US = 1910
_END_OVERHEAD_US = 960
_MSRC_OVERHEAD_US = 660
_TCC_OVERHEAD_US = 590
_DSS_OVERHEAD_US = 690
_PRE_RANGE_OVERHEAD_US = 660
_FINAL_RANGE_OVERHEAD_US = 550

#: ST ``DefaultTuningSettings`` (``VL53L0X_load_tuning_settings``).
DEFAULT_TUNING_SETTINGS = (
    (0xFF, 0x01), (0x00, 0x00), (0xFF, 0x00), (0x09, 0x00), (0x10, 0x00), (0x11, 0x00),
    (0x24, 0x01), (0x25, 0xFF), (0x75, 0x00), (0xFF, 0x01), (0x4E, 0x2C), (0x48, 0x00),
    (0x30, 0x20), (0xFF, 0x00), (0x30, 0x09), (0x54, 0x00), (0x31, 0x04), (0x32, 0x03),
    (0x40, 0x83), (0x46, 0x25), (0x60, 0x00), (0x27, 0x00), (0x50, 0x06), (0x51, 0x00),
    (0x52, 0x96), (0x56, 0x08), (0x57, 0x30), (0x61, 0x00), (0x62, 0x00), (0x64, 0x00),
    (0x65, 0x00), (0x66, 0xA0), (0xFF, 0x01), (0x22, 0x32), (0x47, 0x14), (0x49, 0xFF),
    (0x4A, 0x00), (0xFF, 0x00), (0x7A, 0x0A), (0x7B, 0x00), (0x78, 0x21), (0xFF, 0x01),
    (0x23, 0x34), (0x42, 0x00), (0x44, 0xFF), (0x45, 0x26), (0x46, 0x05), (0x40, 0x40),
    (0x0E, 0x06), (0x20, 0x1A), (0x43, 0x40), (0xFF, 0x00), (0x34, 0x03), (0x35, 0x44),
    (0xFF, 0x01), (0x31, 0x04), (0x4B, 0x09), (0x4C, 0x05), (0x4D, 0x04), (0xFF, 0x00),
    (0x44, 0x00), (0x45, 0x20), (0x47, 0x08), (0x48, 0x28), (0x67, 0x00), (0x70, 0x04),
    (0x71, 0x01), (0x72, 0xFE), (0x76, 0x00), (0x77, 0x00), (0xFF, 0x01), (0x0D, 0x01),
    (0xFF, 0x00), (0x80, 0x01), (0x01, 0xF8), (0xFF, 0x01), (0x8E, 0x01), (0x00, 0x01),
    (0xFF, 0x00), (0x80, 0x00),
)


class VL53L0XError(RuntimeError):
    """Raised when a VL53L0X is missing, misidentified, or unusable."""


class VL53L0XTimeout(VL53L0XError):
    """Raised when a ranging or calibration step does not complete in time."""


@dataclass(frozen=True)
class ToFMeasurement:
    """One decoded time-of-flight reading for one warning sector."""

    sensor_id: str
    sector: Sector
    timestamp_ns: int
    distance_m: Optional[float]
    raw_mm: Optional[int]
    range_status: Optional[int]
    valid: bool
    no_target: bool

    @property
    def clear(self) -> bool:
        """True when a healthy sensor reports open space beyond its range."""

        return self.valid and self.no_target

    def as_dict(self) -> Dict[str, object]:
        return {
            "sensor_id": self.sensor_id,
            "sector": self.sector.value,
            "timestamp_ns": self.timestamp_ns,
            "distance_m": self.distance_m,
            "raw_mm": self.raw_mm,
            "range_status": self.range_status,
            "valid": self.valid,
            "no_target": self.no_target,
        }


def decode_range_status(status: Optional[int], mm: Optional[int]) -> Tuple[bool, bool, Optional[float]]:
    """Map a raw status byte and millimetre count onto ``(valid, no_target, distance_m)``."""

    if status is None or mm is None:
        return False, False, None
    if status == RANGE_VALID_STATUS:
        if mm > MAX_VALID_MM:
            return True, True, None
        # Counts below MIN_VALID_MM sit inside the rated near field. The echo is
        # still real, so the distance is kept: reporting "clear" would tell a
        # blind user the path is open while an object is centimetres away.
        return True, False, mm / 1000.0
    if status in HARDWARE_FAULT_STATUSES:
        return False, False, None
    if status in NO_TARGET_STATUSES or mm >= MAX_VALID_MM or mm in SENTINEL_MM:
        return True, True, None
    return False, False, None


class VL53L0XDriver:
    """One VL53L0X reached through a single TCA9548A channel."""

    def __init__(
        self,
        mux: TCA9548AMux,
        spec: ToFSensorSpec,
        *,
        ready_timeout_s: Optional[float] = None,
    ) -> None:
        self._mux = mux
        self.spec = spec
        self.sensor_id = spec.sensor_id
        self.sector = spec.sector
        self.address = spec.i2c_address
        self.channel = spec.mux_channel
        # A ranging result needs at least one full measurement window, plus
        # margin for the I2C round trip and the control loop's scheduling.
        self._ready_timeout_s = (
            ready_timeout_s
            if ready_timeout_s is not None
            else max(0.25, spec.timing_budget_ms * 2.0 / 1000.0 + 0.05)
        )
        self._stop_variable = 0x00
        self._measurement_timing_budget_us = 0
        self._continuous = False
        self._initialized = False
        self._model_id: Optional[int] = None

    # -- lifecycle -------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def model_id(self) -> Optional[int]:
        return self._model_id

    def initialize(self) -> None:
        """Run the ST data/static init sequence plus VHV and phase calibration."""

        self._initialized = False
        self._continuous = False
        try:
            self._model_id = self._read(IDENTIFICATION_MODEL_ID)
            if self._model_id != MODEL_ID_VL53L0X:
                raise VL53L0XError(
                    f"{self.sensor_id}: unexpected model id 0x{self._model_id:02X} at "
                    f"channel {self.channel} (expected 0x{MODEL_ID_VL53L0X:02X})"
                )

            # VL53L0X_DataInit(): 2.8 V I/O mode and standard-mode I2C.
            self._write(
                VHV_CONFIG_PAD_SCL_SDA__EXTSUP_HV,
                self._read(VHV_CONFIG_PAD_SCL_SDA__EXTSUP_HV) | 0x01,
            )
            self._write(0x88, 0x00)
            self._enter_stop_variable_page()
            self._stop_variable = self._read(STOP_VARIABLE_REGISTER)
            self._exit_stop_variable_page()

            # VL53L0X_StaticInit(): ST's default tuning first, because that
            # table writes 0x60 itself and would undo the limit-check settings.
            for register, value in DEFAULT_TUNING_SETTINGS:
                self._write(register, value)

            # Disable the MSRC and pre-range signal-rate limit checks, then set
            # the final-range signal rate limit to 0.25 MCPS (Q9.7 fixed point).
            self._write(MSRC_CONFIG_CONTROL, self._read(MSRC_CONFIG_CONTROL) | 0x12)
            self._write_reg16(FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT, int(0.25 * (1 << 7)))
            self._write(SYSTEM_SEQUENCE_CONFIG, 0xFF)

            # "Set interrupt config to new sample ready" and active-low GPIO.
            self._write(SYSTEM_INTERRUPT_CONFIG_GPIO, 0x04)
            self._write(GPIO_HV_MUX_ACTIVE_HIGH, self._read(GPIO_HV_MUX_ACTIVE_HIGH) & ~0x10)
            self._write(SYSTEM_INTERRUPT_CLEAR, 0x01)

            # Disable MSRC and TCC, apply the report's timing budget, then run
            # the VHV and phase reference calibrations.
            self._write(SYSTEM_SEQUENCE_CONFIG, 0xE8)
            self.set_measurement_timing_budget_ms(self.spec.timing_budget_ms)
            self._write(SYSTEM_SEQUENCE_CONFIG, 0x01)
            self._perform_single_ref_calibration(0x40)
            self._write(SYSTEM_SEQUENCE_CONFIG, 0x02)
            self._perform_single_ref_calibration(0x00)
            self._write(SYSTEM_SEQUENCE_CONFIG, 0xE8)
        except OSError as exc:
            raise VL53L0XError(
                f"{self.sensor_id}: I2C failure during initialization on channel {self.channel}"
            ) from exc
        self._initialized = True

    def close(self) -> None:
        if self._initialized:
            try:
                self.stop_continuous()
            except (OSError, ValueError):
                pass
        self._continuous = False
        self._initialized = False

    # -- timing budget ---------------------------------------------------

    @property
    def measurement_timing_budget_ms(self) -> float:
        return self._measurement_timing_budget_us / 1000.0

    def set_measurement_timing_budget_ms(self, budget_ms: float) -> None:
        self.set_measurement_timing_budget_us(int(round(float(budget_ms) * 1000.0)))

    def set_measurement_timing_budget_us(self, budget_us: int) -> None:
        """Apply ``VL53L0X_set_measurement_timing_budget`` to the final range step."""

        if budget_us <= 0:
            raise VL53L0XError("measurement timing budget must be positive")
        try:
            sequence_config = self._read(SYSTEM_SEQUENCE_CONFIG)
            enables = {
                "tcc": (sequence_config >> 4) & 0x1,
                "dss": (sequence_config >> 3) & 0x1,
                "msrc": (sequence_config >> 2) & 0x1,
                "pre_range": (sequence_config >> 6) & 0x1,
                "final_range": (sequence_config >> 7) & 0x1,
            }
            pre_range_vcsel = _decode_vcsel_period(self._read(PRE_RANGE_CONFIG_VCSEL_PERIOD))
            final_range_vcsel = _decode_vcsel_period(self._read(FINAL_RANGE_CONFIG_VCSEL_PERIOD))
            msrc_dss_tcc_us = _mclks_to_us(
                self._read(MSRC_CONFIG_TIMEOUT_MACROP) + 1, pre_range_vcsel
            )
            pre_range_mclks = _decode_timeout(self._read_reg16(PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI))
            pre_range_us = _mclks_to_us(pre_range_mclks, pre_range_vcsel)
            used_us = _START_OVERHEAD_US + _END_OVERHEAD_US
            if enables["tcc"]:
                used_us += msrc_dss_tcc_us + _TCC_OVERHEAD_US
            if enables["dss"]:
                used_us += 2 * (msrc_dss_tcc_us + _DSS_OVERHEAD_US)
            elif enables["msrc"]:
                used_us += msrc_dss_tcc_us + _MSRC_OVERHEAD_US
            if enables["pre_range"]:
                used_us += pre_range_us + _PRE_RANGE_OVERHEAD_US
            if not enables["final_range"]:
                raise VL53L0XError(
                    f"{self.sensor_id}: final range step is disabled, cannot set a timing budget"
                )
            used_us += _FINAL_RANGE_OVERHEAD_US
            if used_us > budget_us:
                raise VL53L0XError(
                    f"{self.sensor_id}: timing budget {budget_us} us is too small; "
                    f"the enabled steps already need {used_us} us"
                )
            final_range_timeout_mclks = _us_to_mclks(budget_us - used_us, final_range_vcsel)
            if enables["pre_range"]:
                final_range_timeout_mclks += pre_range_mclks
            self._write_reg16(
                FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI, _encode_timeout(final_range_timeout_mclks)
            )
        except OSError as exc:
            raise VL53L0XError(
                f"{self.sensor_id}: I2C failure while setting the timing budget"
            ) from exc
        self._measurement_timing_budget_us = int(budget_us)

    # -- ranging ---------------------------------------------------------

    def start_continuous(self, period_ms: Optional[float] = None) -> None:
        """Start back-to-back (or timed) continuous ranging."""

        self._require_initialized()
        self._enter_stop_variable_page()
        self._write(STOP_VARIABLE_REGISTER, self._stop_variable)
        self._exit_stop_variable_page()
        if period_ms:
            osc_calibrate = self._read_reg16(OSC_CALIBRATE_VAL)
            period = int(round(period_ms * osc_calibrate)) if osc_calibrate else int(round(period_ms))
            self._write(SYSTEM_INTERMEASUREMENT_PERIOD, (period >> 24) & 0xFF)
            self._write(SYSTEM_INTERMEASUREMENT_PERIOD + 1, (period >> 16) & 0xFF)
            self._write(SYSTEM_INTERMEASUREMENT_PERIOD + 2, (period >> 8) & 0xFF)
            self._write(SYSTEM_INTERMEASUREMENT_PERIOD + 3, period & 0xFF)
            self._write(SYSRANGE_START, 0x04)
        else:
            self._write(SYSRANGE_START, 0x02)
        self._continuous = True

    def stop_continuous(self) -> None:
        self._write(SYSRANGE_START, 0x00)
        self._continuous = False

    def read_measurement(self, now_ns: Optional[int] = None) -> ToFMeasurement:
        """Return the next ranging result, degrading to a fault instead of raising."""

        timestamp_ns = now_ns if now_ns is not None else time.monotonic_ns()
        if not self._initialized:
            raise VL53L0XError(f"{self.sensor_id}: initialize() must be called before ranging")
        try:
            if not self._continuous:
                self._start_single_measurement()
            self._wait_for_result()
            block = self._mux.read_block(self.channel, self.address, RESULT_RANGE_STATUS, 12)
            self._write(SYSTEM_INTERRUPT_CLEAR, 0x01)
        except (OSError, VL53L0XTimeout):
            return self._fault(timestamp_ns)

        raw_status = (block[0] >> 3) & 0x0F
        raw_mm = (block[10] << 8) | block[11]
        valid, no_target, distance_m = decode_range_status(raw_status, raw_mm)
        return ToFMeasurement(
            sensor_id=self.sensor_id,
            sector=self.sector,
            timestamp_ns=timestamp_ns,
            distance_m=distance_m,
            raw_mm=raw_mm,
            range_status=raw_status,
            valid=valid,
            no_target=no_target,
        )

    def _fault(self, timestamp_ns: int) -> ToFMeasurement:
        return ToFMeasurement(
            sensor_id=self.sensor_id,
            sector=self.sector,
            timestamp_ns=timestamp_ns,
            distance_m=None,
            raw_mm=None,
            range_status=None,
            valid=False,
            no_target=False,
        )

    def _start_single_measurement(self) -> None:
        # The stop-variable sequence is required before every single shot; the
        # start bit clears itself in hardware, so completion is polled through
        # RESULT_INTERRUPT_STATUS rather than by reading SYSRANGE_START back.
        self._enter_stop_variable_page()
        self._write(STOP_VARIABLE_REGISTER, self._stop_variable)
        self._exit_stop_variable_page()
        self._write(SYSRANGE_START, 0x01)

    def _wait_for_result(self) -> None:
        deadline = time.monotonic() + self._ready_timeout_s
        while True:
            status = self._read(RESULT_INTERRUPT_STATUS)
            if status & 0x07:
                return
            if time.monotonic() >= deadline:
                raise VL53L0XTimeout(f"{self.sensor_id}: ranging did not complete in time")
            time.sleep(0.001)

    def _perform_single_ref_calibration(self, vhv_init_byte: int) -> None:
        self._write(SYSRANGE_START, 0x01 | vhv_init_byte)
        self._wait_for_result()
        self._write(SYSTEM_INTERRUPT_CLEAR, 0x01)
        self._write(SYSRANGE_START, 0x00)

    def _enter_stop_variable_page(self) -> None:
        self._write(0x80, 0x01)
        self._write(MUX_PAGE_CONTROL, 0x01)
        self._write(0x00, 0x00)

    def _exit_stop_variable_page(self) -> None:
        self._write(0x00, 0x01)
        self._write(MUX_PAGE_CONTROL, 0x00)
        self._write(0x80, 0x00)

    # -- register helpers ------------------------------------------------

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise VL53L0XError(f"{self.sensor_id}: initialize() must be called first")

    def _read(self, register: int) -> int:
        return self._mux.read_byte(self.channel, self.address, register)

    def _write(self, register: int, value: int) -> None:
        self._mux.write_byte(self.channel, self.address, register, value)

    def _read_reg16(self, register: int) -> int:
        block = self._mux.read_block(self.channel, self.address, register, 2)
        return (block[0] << 8) | block[1]

    def _write_reg16(self, register: int, value: int) -> None:
        self._write(register, (value >> 8) & 0xFF)
        self._write(register + 1, value & 0xFF)


def _decode_vcsel_period(register_value: int) -> int:
    return (register_value + 1) << 1


def _decode_timeout(register_value: int) -> int:
    return ((register_value & 0x00FF) << ((register_value & 0xFF00) >> 8)) + 1


def _encode_timeout(timeout_mclks: int) -> int:
    if timeout_mclks <= 0:
        return 0
    ls_byte = timeout_mclks - 1
    ms_byte = 0
    while ls_byte & 0xFFFFFF00:
        ls_byte >>= 1
        ms_byte += 1
    return (ms_byte << 8) | (ls_byte & 0xFF)


def _macro_period_ns(vcsel_period_pclks: int) -> int:
    return ((2304 * vcsel_period_pclks * 1655) + 500) // 1000


def _mclks_to_us(timeout_mclks: int, vcsel_period_pclks: int) -> int:
    macro_period_ns = _macro_period_ns(vcsel_period_pclks)
    return ((timeout_mclks * macro_period_ns) + 500) // 1000


def _us_to_mclks(timeout_us: int, vcsel_period_pclks: int) -> int:
    macro_period_ns = _macro_period_ns(vcsel_period_pclks)
    if macro_period_ns <= 0:
        raise VL53L0XError("invalid VCSEL period: macro period is zero")
    return ((timeout_us * 1000) + (macro_period_ns // 2)) // macro_period_ns


class VL53L0XArray:
    """The three cane ToF sensors with per-sensor fault isolation.

    One unplugged or shorted sensor must never blind the other two: a failing
    channel produces a fault measurement for that sector only, and the sensor
    can be re-initialized later without disturbing its neighbours.
    """

    def __init__(self, mux: TCA9548AMux, specs: Sequence[ToFSensorSpec]) -> None:
        if not specs:
            raise ValueError("VL53L0XArray needs at least one sensor spec")
        self._mux = mux
        self.specs = tuple(specs)
        self._drivers: Dict[str, VL53L0XDriver] = {
            spec.sensor_id: VL53L0XDriver(mux, spec) for spec in self.specs
        }
        self._faults: Dict[str, str] = {}
        self._quarantined: Set[str] = set()

    @property
    def faulted_sensor_ids(self) -> List[str]:
        """Sensors whose latest poll failed or that never initialized."""

        return [spec.sensor_id for spec in self.specs if spec.sensor_id in self._faults]

    @property
    def quarantined_sensor_ids(self) -> List[str]:
        """Sensors taken off the bus until :meth:`initialize_sensor` succeeds."""

        return [spec.sensor_id for spec in self.specs if spec.sensor_id in self._quarantined]

    @property
    def ready_sensor_ids(self) -> List[str]:
        return [spec.sensor_id for spec in self.specs if spec.sensor_id not in self._faults]

    def fault_reason(self, sensor_id: str) -> Optional[str]:
        return self._faults.get(sensor_id)

    def initialize(self) -> List[str]:
        """Initialize every sensor; return the ids that could not be brought up."""

        for spec in self.specs:
            driver = self._drivers[spec.sensor_id]
            try:
                driver.initialize()
            except (VL53L0XError, OSError) as exc:
                self._faults[spec.sensor_id] = str(exc)
                self._quarantined.add(spec.sensor_id)
            else:
                self._faults.pop(spec.sensor_id, None)
                self._quarantined.discard(spec.sensor_id)
        return self.quarantined_sensor_ids

    def initialize_sensor(self, sensor_id: str) -> None:
        """Re-initialize one sensor and clear its fault on success."""

        spec = self._spec(sensor_id)
        self._drivers[spec.sensor_id].initialize()
        self._faults.pop(spec.sensor_id, None)
        self._quarantined.discard(spec.sensor_id)

    def read_all(self, now_ns: Optional[int] = None) -> List[ToFMeasurement]:
        timestamp_ns = now_ns if now_ns is not None else time.monotonic_ns()
        measurements: List[ToFMeasurement] = []
        for spec in self.specs:
            driver = self._drivers[spec.sensor_id]
            if spec.sensor_id in self._quarantined:
                measurements.append(self._fault_measurement(spec, timestamp_ns))
                continue
            try:
                measurement = driver.read_measurement(timestamp_ns)
            except (VL53L0XError, OSError) as exc:
                self._faults[spec.sensor_id] = str(exc)
                self._quarantined.add(spec.sensor_id)
                measurements.append(self._fault_measurement(spec, timestamp_ns))
                continue
            if measurement.valid:
                self._faults.pop(spec.sensor_id, None)
            else:
                # A single bad range (bus glitch, ranging timeout) is reported
                # for this sector but the sensor keeps being polled, so a
                # transient fault heals without an explicit re-initialization.
                self._faults[spec.sensor_id] = (
                    "invalid range measurement (I2C fault or ranging timeout)"
                )
            measurements.append(measurement)
        return measurements

    def close(self) -> None:
        for driver in self._drivers.values():
            driver.close()

    @staticmethod
    def _fault_measurement(spec: ToFSensorSpec, timestamp_ns: int) -> ToFMeasurement:
        return ToFMeasurement(
            sensor_id=spec.sensor_id,
            sector=spec.sector,
            timestamp_ns=timestamp_ns,
            distance_m=None,
            raw_mm=None,
            range_status=None,
            valid=False,
            no_target=False,
        )

    def _spec(self, sensor_id: str) -> ToFSensorSpec:
        for spec in self.specs:
            if spec.sensor_id == sensor_id:
                return spec
        raise KeyError(f"unknown time-of-flight sensor {sensor_id!r}")
