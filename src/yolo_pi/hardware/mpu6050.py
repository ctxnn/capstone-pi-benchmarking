"""Pure-Python MPU-6050 six-axis IMU driver.

The report places the IMU behind TCA9548A channel 7 (or on the main bus when
``mux_channel`` is ``None``), with an optional interrupt on GPIO17. The driver
runs the device at ~100 Hz with a 44 Hz digital low-pass filter, because the
cane's handle transmits motor and cane-tip vibration straight into the MEMS
element and unfiltered gyro noise dominates step detection.

Ranges follow the report: +-250 deg/s (131.0 LSB per deg/s) and +-2 g
(16384.0 LSB per g), with the datasheet temperature formula
``raw / 340.0 + 36.53``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .config import IMUSpec
from .i2c import TCA9548AMux

LOGGER = logging.getLogger(__name__)

WHO_AM_I = 0x75
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
PWR_MGMT_1 = 0x6B

PWR_MGMT_1_SLEEP = 0x40
PWR_MGMT_1_CLKSEL_X_GYRO = 0x01
DLPF_CONFIG_44_HZ = 0x03
SAMPLE_RATE_DIVIDER_BASE_HZ = 1000.0

GYRO_FS_250_LSB_PER_DPS = 131.0
ACCEL_FS_2G_LSB_PER_G = 16384.0
STANDARD_GRAVITY_MPS2 = 9.80665
DEGREES_TO_RADIANS = math.pi / 180.0

#: MPU-6050 clones report 0x68; some MPU-6500/9250 boards answer 0x70/0x71.
SUPPORTED_WHO_AM_I = frozenset({0x68, 0x69, 0x70, 0x71})
BURST_LENGTH = 14


class MPU6050Error(RuntimeError):
    """Raised when the IMU is missing, misidentified, or returns bad data."""


@dataclass(frozen=True)
class RawIMUSample:
    """One bias-free SI-unit sample, shaped to feed the fusion IMU contract."""

    timestamp_ns: int
    accel_mps2: Tuple[float, float, float]
    gyro_rad_s: Tuple[float, float, float]
    temperature_c: float
    accel_raw: Tuple[int, int, int]
    gyro_raw: Tuple[int, int, int]
    valid: bool = True

    def as_imu_fields(self) -> Dict[str, object]:
        """Keyword arguments for ``yolo_pi.imu.IMUSample``.

        Kept as a plain dict so the hardware layer never imports the processing
        layer and the fusion pipeline can wrap a sample without conversion code.
        """

        ax, ay, az = self.accel_mps2
        gx, gy, gz = self.gyro_rad_s
        return {
            "timestamp_ns": self.timestamp_ns,
            "ax_mps2": ax,
            "ay_mps2": ay,
            "az_mps2": az,
            "gx_rps": gx,
            "gy_rps": gy,
            "gz_rps": gz,
            "temp_c": self.temperature_c,
            "valid": self.valid,
        }


@dataclass(frozen=True)
class IMUCalibration:
    """Startup bias estimates measured with the cane held still."""

    accel_bias_mps2: Tuple[float, float, float]
    gyro_bias_rad_s: Tuple[float, float, float]
    samples: int
    temperature_c: float

    def as_tuple(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return (self.accel_bias_mps2, self.gyro_bias_rad_s)


class MPU6050Driver:
    """One MPU-6050 reached through a multiplexer channel or the main bus."""

    def __init__(self, mux: TCA9548AMux, spec: IMUSpec, *, settle_s: float = 0.03) -> None:
        self._mux = mux
        self.spec = spec
        self.sensor_id = spec.sensor_id
        self.address = spec.i2c_address
        self.channel = spec.mux_channel
        self._settle_s = max(0.0, float(settle_s))
        self._initialized = False
        self._who_am_i: Optional[int] = None
        self._calibration: Optional[IMUCalibration] = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def who_am_i(self) -> Optional[int]:
        return self._who_am_i

    @property
    def calibration(self) -> Optional[IMUCalibration]:
        return self._calibration

    @property
    def sample_rate_divider(self) -> int:
        divider = int(round(SAMPLE_RATE_DIVIDER_BASE_HZ / self.spec.sample_rate_hz)) - 1
        return max(0, min(255, divider))

    def initialize(self) -> None:
        """Verify identity, wake the device, and configure ranges and filtering."""

        try:
            self._who_am_i = self._read(WHO_AM_I)
            if self._who_am_i not in SUPPORTED_WHO_AM_I:
                raise MPU6050Error(
                    f"{self.sensor_id}: unexpected WHO_AM_I 0x{self._who_am_i:02X} "
                    f"(expected one of {sorted(hex(value) for value in SUPPORTED_WHO_AM_I)})"
                )
            if self._who_am_i != 0x68:
                LOGGER.warning(
                    "%s: WHO_AM_I reports 0x%02X rather than the MPU-6050's 0x68; "
                    "treating it as a register-compatible clone",
                    self.sensor_id,
                    self._who_am_i,
                )

            # Wake on the X-gyro PLL: the internal 8 MHz oscillator drifts with
            # temperature and would bias every integrated heading.
            self._write(PWR_MGMT_1, PWR_MGMT_1_CLKSEL_X_GYRO)
            if self._read(PWR_MGMT_1) & PWR_MGMT_1_SLEEP:
                raise MPU6050Error(f"{self.sensor_id}: device stayed asleep after wake-up")
            if self._settle_s:
                time.sleep(self._settle_s)

            self._write(SMPLRT_DIV, self.sample_rate_divider)
            self._write(CONFIG, DLPF_CONFIG_44_HZ)
            self._write(GYRO_CONFIG, 0x00)
            self._write(ACCEL_CONFIG, 0x00)
        except OSError as exc:
            raise MPU6050Error(
                f"{self.sensor_id}: I2C failure during initialization "
                f"(address 0x{self.address:02X}, channel {self.channel})"
            ) from exc
        self._initialized = True

    def read_raw(self, now_ns: Optional[int] = None) -> RawIMUSample:
        """Burst-read all 14 bytes from ACCEL_XOUT_H and convert to SI units."""

        self._require_initialized()
        timestamp_ns = now_ns if now_ns is not None else time.monotonic_ns()
        try:
            block = self._mux.read_block(self.channel, self.address, ACCEL_XOUT_H, BURST_LENGTH)
        except OSError as exc:
            raise MPU6050Error(
                f"{self.sensor_id}: I2C failure reading the burst sample"
            ) from exc
        if len(block) != BURST_LENGTH:
            raise MPU6050Error(
                f"{self.sensor_id}: short burst read ({len(block)} of {BURST_LENGTH} bytes)"
            )
        values = [_to_signed_int16(block[index], block[index + 1]) for index in range(0, 14, 2)]
        accel_raw = (values[0], values[1], values[2])
        temperature_raw = values[3]
        gyro_raw = (values[4], values[5], values[6])
        return RawIMUSample(
            timestamp_ns=timestamp_ns,
            accel_mps2=tuple(
                raw / ACCEL_FS_2G_LSB_PER_G * STANDARD_GRAVITY_MPS2 for raw in accel_raw
            ),
            gyro_rad_s=tuple(
                raw / GYRO_FS_250_LSB_PER_DPS * DEGREES_TO_RADIANS for raw in gyro_raw
            ),
            temperature_c=temperature_raw / 340.0 + 36.53,
            accel_raw=accel_raw,
            gyro_raw=gyro_raw,
        )

    def calibrate_startup(self, num_samples: int = 80, delay_s: float = 0.005) -> IMUCalibration:
        """Average still samples to estimate accelerometer and gyro biases."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if delay_s < 0:
            raise ValueError("delay_s must not be negative")
        accel_sums = [0.0, 0.0, 0.0]
        gyro_sums = [0.0, 0.0, 0.0]
        temperature = 0.0
        for index in range(num_samples):
            sample = self.read_raw()
            for axis in range(3):
                accel_sums[axis] += sample.accel_mps2[axis]
                gyro_sums[axis] += sample.gyro_rad_s[axis]
            temperature += sample.temperature_c
            if delay_s and index + 1 < num_samples:
                time.sleep(delay_s)
        accel_mean = [total / num_samples for total in accel_sums]
        gyro_bias = [total / num_samples for total in gyro_sums]
        # A still accelerometer measures gravity plus bias. Removing one g along
        # the measured gravity direction leaves the residual bias, which stays
        # valid for any handle tilt at startup.
        norm = math.sqrt(sum(value * value for value in accel_mean))
        if norm > 0.0:
            gravity = [value * STANDARD_GRAVITY_MPS2 / norm for value in accel_mean]
        else:
            gravity = [0.0, 0.0, STANDARD_GRAVITY_MPS2]
        accel_bias = [measured - reference for measured, reference in zip(accel_mean, gravity)]
        self._calibration = IMUCalibration(
            accel_bias_mps2=(accel_bias[0], accel_bias[1], accel_bias[2]),
            gyro_bias_rad_s=(gyro_bias[0], gyro_bias[1], gyro_bias[2]),
            samples=num_samples,
            temperature_c=temperature / num_samples,
        )
        return self._calibration

    def close(self) -> None:
        """Best-effort sleep entry so the IMU stops drawing current."""

        if self._initialized:
            try:
                self._write(PWR_MGMT_1, PWR_MGMT_1_SLEEP)
            except OSError:
                pass
        self._initialized = False

    # -- register helpers ------------------------------------------------

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise MPU6050Error(f"{self.sensor_id}: initialize() must be called first")

    def _read(self, register: int) -> int:
        return self._mux.read_byte(self.channel, self.address, register)

    def _write(self, register: int, value: int) -> None:
        self._mux.write_byte(self.channel, self.address, register, value)


def _to_signed_int16(high: int, low: int) -> int:
    value = ((high & 0xFF) << 8) | (low & 0xFF)
    return value - 0x10000 if value & 0x8000 else value
