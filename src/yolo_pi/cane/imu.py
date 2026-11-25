"""MPU-6050 motion sensing: gyro-bias calibration, a stillness flag, and the jolt trigger.

The IMU never reports obstacles. It does two things:

* ``JoltDetector``: vibrate all three motors once when the cane's tilt changes
  suddenly (knocked, dropped, snagged). This is the only thing the IMU ever
  vibrates for.
* ``MotionMonitor``: tells the runtime whether the cane is resting, so the
  detector can drop to a low frame rate and save battery. It produces no
  vibration. The range sensors keep running at full rate either way.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Sequence, Tuple

from .bus import RegisterDevice
from .config import ImuConfig

LOGGER = logging.getLogger("yolo_pi.cane.imu")

REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B
REG_PWR_MGMT_1 = 0x6B
REG_WHO_AM_I = 0x75

GYRO_LSB_PER_DPS = 65.5  # +/-500 deg/s full scale
ACCEL_LSB_PER_G = 8192.0  # +/-4 g full scale
KNOWN_WHO_AM_I = {0x68, 0x69, 0x70, 0x71, 0x72}


@dataclass(frozen=True)
class ImuSample:
    accel_g: Tuple[float, float, float]
    gyro_dps: Tuple[float, float, float]
    temperature_c: float

    def is_plausible(self) -> bool:
        """Finite and inside what the chip can physically report (+/-4 g, +/-500 deg/s)."""

        return all(math.isfinite(v) for v in (*self.accel_g, *self.gyro_dps)) and (
            all(abs(v) <= 5.0 for v in self.accel_g) and all(abs(v) <= 600.0 for v in self.gyro_dps)
        )


def _signed16(high: int, low: int) -> int:
    value = (high << 8) | low
    return value - 0x10000 if value & 0x8000 else value


def decode_sample(raw: bytes) -> ImuSample:
    """Decode the 14-byte burst at register 0x3B into physical units."""

    if len(raw) != 14:
        raise ValueError("MPU-6050 burst must be 14 bytes")
    accel = tuple(_signed16(raw[i], raw[i + 1]) / ACCEL_LSB_PER_G for i in (0, 2, 4))
    temperature = _signed16(raw[6], raw[7]) / 340.0 + 36.53
    gyro = tuple(_signed16(raw[i], raw[i + 1]) / GYRO_LSB_PER_DPS for i in (8, 10, 12))
    return ImuSample(accel, gyro, temperature)  # type: ignore[arg-type]


class Mpu6050:
    def __init__(self, device: RegisterDevice, sleep: Callable[[float], None] = time.sleep) -> None:
        self._device = device
        self._sleep = sleep
        self.who_am_i: Optional[int] = None

    def configure(self) -> None:
        self.who_am_i = self._device.read_reg(REG_WHO_AM_I)
        if self.who_am_i not in KNOWN_WHO_AM_I:
            LOGGER.warning("unexpected MPU-6050 WHO_AM_I 0x%02x; continuing", self.who_am_i)
        self._device.write_reg(REG_PWR_MGMT_1, 0x80)  # device reset
        self._sleep(0.1)
        self._device.write_reg(REG_PWR_MGMT_1, 0x01)  # wake, gyro-X PLL clock
        self._sleep(0.05)
        self._device.write_reg(REG_SMPLRT_DIV, 9)  # 1 kHz / 10 = 100 Hz
        self._device.write_reg(REG_CONFIG, 0x03)  # ~42 Hz low-pass filter
        self._device.write_reg(REG_GYRO_CONFIG, 0x08)
        self._device.write_reg(REG_ACCEL_CONFIG, 0x08)

    def read(self) -> ImuSample:
        return decode_sample(self._device.read_regs(REG_ACCEL_XOUT_H, 14))


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


@dataclass(frozen=True)
class Calibration:
    bias_dps: Tuple[float, float, float]
    calibrated: bool
    attempts: int


def calibrate_gyro(
    imu: Mpu6050,
    config: ImuConfig,
    sleep: Callable[[float], None] = time.sleep,
) -> Calibration:
    """Average the gyro while the cane is still to find its zero-rate offset.

    A bias measured while the cane is being waved about would be wrong, so a run
    whose mean rate is too high is rejected and retried. If every attempt is
    rejected the bias stays at zero and the runtime reports ``calibrated=False``
    instead of trusting a bad number.
    """

    for attempt in range(1, config.calibration_attempts + 1):
        samples: List[ImuSample] = [imu.read() for _ in range(config.calibration_samples)]
        mean = tuple(
            sum(sample.gyro_dps[axis] for sample in samples) / len(samples) for axis in range(3)
        )
        spread = max(
            math.sqrt(
                sum((sample.gyro_dps[axis] - mean[axis]) ** 2 for sample in samples) / len(samples)
            )
            for axis in range(3)
        )
        if _norm(mean) <= config.calibration_max_gyro_dps and spread <= config.calibration_max_gyro_dps:
            return Calibration(mean, True, attempt)  # type: ignore[arg-type]
        LOGGER.warning(
            "gyro calibration attempt %d rejected (mean %.1f dps, spread %.1f dps): cane is moving",
            attempt, _norm(mean), spread,
        )
        sleep(0.5)
    return Calibration((0.0, 0.0, 0.0), False, config.calibration_attempts)


class MotionMonitor:
    """Reports ``is_still`` once the cane has been motionless for ``still_after_s``."""

    def __init__(
        self,
        config: ImuConfig,
        bias_dps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._config = config
        self._bias = bias_dps
        self._still_since_ns: Optional[int] = None
        self.is_still = False
        self.last_gyro_dps = 0.0
        self.last_accel_error_g = 0.0

    def update(self, sample: ImuSample, now_ns: int) -> bool:
        if not sample.is_plausible():
            self._still_since_ns = None
            self.is_still = False  # a corrupt reading counts as "moving", never as "resting"
            return False
        corrected = tuple(sample.gyro_dps[i] - self._bias[i] for i in range(3))
        self.last_gyro_dps = _norm(corrected)
        self.last_accel_error_g = abs(_norm(sample.accel_g) - 1.0)
        quiet = (
            self.last_gyro_dps <= self._config.still_gyro_dps
            and self.last_accel_error_g <= self._config.still_accel_g
        )
        if not quiet:
            self._still_since_ns = None
            self.is_still = False
            return False
        if self._still_since_ns is None:
            self._still_since_ns = now_ns
        self.is_still = (now_ns - self._still_since_ns) >= int(self._config.still_after_s * 1e9)
        return self.is_still


@dataclass(frozen=True)
class JoltEvent:
    timestamp_ns: int
    angle_deg: float


class JoltDetector:
    """Detects a sudden change in the cane's tilt, whichever way the chip is mounted.

    It tracks the direction of gravity in the sensor's own frame, rotating it
    with the gyro and gently pulling it back to the accelerometer's direction
    (a complementary filter). Because it only watches that direction, it needs
    no assumption about which MPU-6050 axis points where on the cane, and it
    ignores turning about the vertical (which does not change tilt).

    A jolt is the gravity direction moving by ``jolt_angle_deg`` or more within
    ``jolt_window_s``; slow tilting, such as swinging the cane while walking,
    does not qualify. After a jolt it stays quiet for ``jolt_cooldown_s``.
    Accelerometer samples far from 1 g (impacts, free fall) are not trusted to
    correct the estimate, so a knock does not fake a rotation.
    """

    MAX_STEP_S = 0.2

    def __init__(
        self,
        config: ImuConfig,
        bias_dps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._config = config
        self._bias = bias_dps
        self._gravity: Optional[Tuple[float, float, float]] = None
        self._last_ns: Optional[int] = None
        self._history: Deque[Tuple[int, Tuple[float, float, float]]] = deque()
        self._quiet_until_ns = 0
        self.last_angle_deg = 0.0

    @staticmethod
    def _unit(vector: Sequence[float]) -> Optional[Tuple[float, float, float]]:
        length = _norm(vector)
        if length < 1e-9:
            return None
        return (vector[0] / length, vector[1] / length, vector[2] / length)

    def update(self, sample: ImuSample, now_ns: int) -> Optional[JoltEvent]:
        if not sample.is_plausible():
            return None  # one corrupt reading must never poison the running estimate
        measured = self._unit(sample.accel_g)
        magnitude = _norm(sample.accel_g)
        trusted = measured is not None and abs(magnitude - 1.0) <= self._config.jolt_accel_tolerance_g
        if self._gravity is None:
            if trusted:
                self._gravity, self._last_ns = measured, now_ns
                self._history.append((now_ns, measured))
            return None

        dt = (now_ns - (self._last_ns or now_ns)) / 1e9
        self._last_ns = now_ns
        if dt <= 0.0:
            return None
        dt = min(dt, self.MAX_STEP_S)

        wx, wy, wz = (math.radians(sample.gyro_dps[i] - self._bias[i]) for i in range(3))
        gx, gy, gz = self._gravity
        # In the body frame a fixed world vector turns by -(omega x g) * dt.
        predicted = (
            gx - (wy * gz - wz * gy) * dt,
            gy - (wz * gx - wx * gz) * dt,
            gz - (wx * gy - wy * gx) * dt,
        )
        if trusted and measured is not None:
            alpha = self._config.gravity_filter_alpha
            blended = tuple(alpha * predicted[i] + (1.0 - alpha) * measured[i] for i in range(3))
        else:
            blended = predicted
        unit = self._unit(blended)
        if unit is None or not all(math.isfinite(v) for v in unit):
            self._gravity, self._history = None, deque()  # start over from the next good sample
            return None
        self._gravity = unit

        window_ns = int(self._config.jolt_window_s * 1e9)
        self._history.append((now_ns, unit))
        while self._history and now_ns - self._history[0][0] > window_ns:
            self._history.popleft()
        oldest = self._history[0][1]
        dot = max(-1.0, min(1.0, sum(unit[i] * oldest[i] for i in range(3))))
        self.last_angle_deg = math.degrees(math.acos(dot))
        if self.last_angle_deg >= self._config.jolt_angle_deg and now_ns >= self._quiet_until_ns:
            self._quiet_until_ns = now_ns + int(self._config.jolt_cooldown_s * 1e9)
            self._history.clear()
            self._history.append((now_ns, unit))
            return JoltEvent(now_ns, self.last_angle_deg)
        return None
