"""Six-axis MPU-6050 processing for the SmartCane.

The module depends on the standard library and :mod:`math` only, so identical
code runs on the laptop replay harness and on the Pi.

Sensor frame convention (right-handed, right angles measured about it)::

    +x  forward, along the cane shaft toward the tip
    +y  to the cane's left
    +z  up

``pitch_deg`` is positive when the shaft tip is pitched downward relative to
horizontal, ``roll_deg`` is positive when the shaft is tilted toward the cane's
right, and yaw is positive counter-clockwise seen from above. At rest a level
sensor reads ``(0, 0, +9.80665)`` m/s^2 for the gravity reaction.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

GRAVITY_MPS2 = 9.80665
RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0

_MAX_INTEGRATION_STEP_S = 0.5


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _accel_pitch_deg(ax: float, ay: float, az: float) -> float:
    """Pitch from the gravity reaction, positive when the tip points down."""

    return math.degrees(math.atan2(-ax, math.hypot(ay, az)))


def _accel_roll_deg(ax: float, ay: float, az: float) -> float:
    """Roll from the gravity reaction, positive when tilted to the right."""

    return math.degrees(math.atan2(ay, math.hypot(ax, az)))


@dataclass(frozen=True)
class IMUSample:
    """One raw 6-axis sample expressed in SI units."""

    timestamp_ns: int
    ax_mps2: float
    ay_mps2: float
    az_mps2: float
    gx_rps: float
    gy_rps: float
    gz_rps: float
    temp_c: Optional[float] = None
    valid: bool = True

    @property
    def accel_norm_mps2(self) -> float:
        return math.sqrt(
            self.ax_mps2 * self.ax_mps2
            + self.ay_mps2 * self.ay_mps2
            + self.az_mps2 * self.az_mps2
        )

    @property
    def gyro_norm_rps(self) -> float:
        return math.sqrt(
            self.gx_rps * self.gx_rps
            + self.gy_rps * self.gy_rps
            + self.gz_rps * self.gz_rps
        )

    def is_finite(self) -> bool:
        values = (
            self.ax_mps2,
            self.ay_mps2,
            self.az_mps2,
            self.gx_rps,
            self.gy_rps,
            self.gz_rps,
        )
        return all(math.isfinite(value) for value in values)

    def usable(self) -> bool:
        return self.valid and self.is_finite()


@dataclass(frozen=True)
class IMUConfig:
    """Tuning for the attitude filter and the cane-specific detectors."""

    nominal_pitch_deg: float = 12.0
    complementary_alpha: float = 0.96
    accel_trust_band_mps2: float = 2.0
    sweep_tau_s: float = 0.6
    ground_tilt_warn_deg: float = 25.0
    lifted_pitch_deg: float = -35.0
    extreme_roll_deg: float = 55.0
    sensor_height_m: float = 0.75
    stationary_gyro_rps: float = 0.08
    stationary_accel_std_mps2: float = 0.20
    stationary_dwell_s: float = 1.5
    freefall_g: float = 0.35
    impact_g: float = 3.2
    stale_after_ms: float = 250.0
    camera_hfov_deg: float = 90.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.complementary_alpha <= 1.0:
            raise ValueError("complementary_alpha must be in [0, 1]")
        if self.accel_trust_band_mps2 <= 0.0:
            raise ValueError("accel_trust_band_mps2 must be positive")
        if self.sweep_tau_s <= 0.0:
            raise ValueError("sweep_tau_s must be positive")
        if self.sensor_height_m <= 0.0:
            raise ValueError("sensor_height_m must be positive")
        if self.stationary_dwell_s <= 0.0:
            raise ValueError("stationary_dwell_s must be positive")
        if self.stale_after_ms <= 0.0:
            raise ValueError("stale_after_ms must be positive")
        if not 0.0 < self.camera_hfov_deg < 180.0:
            raise ValueError("camera_hfov_deg must be in (0, 180)")


@dataclass(frozen=True)
class IMUState:
    """Fused attitude and cane-state snapshot consumed by the fusion engine.

    ``relative_pitch_deg`` is measured against the calibrated baseline, so a
    cane carried at its nominal 12 degree forward tilt reads zero and only
    deviations (lifted tip, floor strike, slope) move it.
    """

    timestamp_ns: int
    fresh: bool
    calibrated: bool
    pitch_deg: float
    roll_deg: float
    relative_pitch_deg: float
    yaw_rate_rps: float
    sweep_yaw_deg: float
    is_stationary: bool = False
    is_lifted_or_tilted: bool = False
    is_pitched_down: bool = False
    expected_ground_distance_m: Optional[float] = None
    drop_detected: bool = False
    gyro_bias_rps: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_dict(self) -> dict:
        return {
            "timestamp_ns": self.timestamp_ns,
            "fresh": self.fresh,
            "calibrated": self.calibrated,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "relative_pitch_deg": self.relative_pitch_deg,
            "yaw_rate_rps": self.yaw_rate_rps,
            "sweep_yaw_deg": self.sweep_yaw_deg,
            "is_stationary": self.is_stationary,
            "is_lifted_or_tilted": self.is_lifted_or_tilted,
            "is_pitched_down": self.is_pitched_down,
            "expected_ground_distance_m": self.expected_ground_distance_m,
            "drop_detected": self.drop_detected,
            "gyro_bias_rps": list(self.gyro_bias_rps),
        }


class IMUProcessor:
    """Adaptive complementary attitude filter with cane-specific detectors.

    The filter integrates the bias-corrected gyro and corrects toward the
    accelerometer only while the specific-force magnitude stays inside
    ``accel_trust_band_mps2`` of one g, so footsteps, floor strikes and shock
    loads cannot drag the pitch estimate around.
    """

    def __init__(self, config: Optional[IMUConfig] = None) -> None:
        self.config = config or IMUConfig()
        self._bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._baseline_pitch_deg = 0.0
        self._baseline_roll_deg = 0.0
        self._calibrated = False
        self._initialised = False
        self._pitch_deg = 0.0
        self._roll_deg = 0.0
        self._sweep_yaw_deg = 0.0
        self._yaw_rate_rps = 0.0
        self._last_ts_ns: Optional[int] = None
        self._last_state: Optional[IMUState] = None
        self._window: Deque[Tuple[int, float, float, float, float, float, float]] = deque()
        self._freefall_since_ns: Optional[int] = None
        self._drop_detected = False

    @property
    def baseline_pitch_deg(self) -> float:
        return self._baseline_pitch_deg

    @property
    def baseline_roll_deg(self) -> float:
        return self._baseline_roll_deg

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def gyro_bias_rps(self) -> Tuple[float, float, float]:
        return self._bias

    def calibrate(self, samples: Sequence[IMUSample]) -> Tuple[float, float, float]:
        """Learn gyro bias and the resting pitch/roll from startup samples."""

        usable = [sample for sample in samples if sample.usable()]
        if not usable:
            return (0.0, 0.0, 0.0)
        count = float(len(usable))
        bx = sum(sample.gx_rps for sample in usable) / count
        by = sum(sample.gy_rps for sample in usable) / count
        bz = sum(sample.gz_rps for sample in usable) / count
        ax = sum(sample.ax_mps2 for sample in usable) / count
        ay = sum(sample.ay_mps2 for sample in usable) / count
        az = sum(sample.az_mps2 for sample in usable) / count
        if not (math.isfinite(ax) and math.isfinite(ay) and math.isfinite(az)):
            return (bx, by, bz)
        self.set_bias(
            bx,
            by,
            bz,
            baseline_pitch_deg=_accel_pitch_deg(ax, ay, az),
            baseline_roll_deg=_accel_roll_deg(ax, ay, az),
        )
        return (bx, by, bz)

    def set_bias(
        self,
        bx: float,
        by: float,
        bz: float,
        baseline_pitch_deg: float = 0.0,
        baseline_roll_deg: float = 0.0,
    ) -> None:
        """Install a known bias and resting attitude, e.g. from a rig fixture."""

        self._bias = (float(bx), float(by), float(bz))
        self._baseline_pitch_deg = float(baseline_pitch_deg)
        self._baseline_roll_deg = float(baseline_roll_deg)
        self._pitch_deg = self._baseline_pitch_deg
        self._roll_deg = self._baseline_roll_deg
        self._initialised = True
        self._calibrated = True

    def reset(self) -> None:
        """Forget all integrator, window, and detector history."""

        self._bias = (0.0, 0.0, 0.0)
        self._baseline_pitch_deg = 0.0
        self._baseline_roll_deg = 0.0
        self._calibrated = False
        self._initialised = False
        self._pitch_deg = 0.0
        self._roll_deg = 0.0
        self._sweep_yaw_deg = 0.0
        self._yaw_rate_rps = 0.0
        self._last_ts_ns = None
        self._last_state = None
        self._window.clear()
        self._freefall_since_ns = None
        self._drop_detected = False

    def update(self, sample: IMUSample) -> IMUState:
        """Fold one sample into the filter and return the resulting state."""

        timestamp_ns = int(sample.timestamp_ns)
        if not sample.usable():
            return self.state(timestamp_ns)
        if self._last_ts_ns is not None and timestamp_ns <= self._last_ts_ns:
            return self.state(self._last_ts_ns)

        dt_s = 0.0 if self._last_ts_ns is None else (timestamp_ns - self._last_ts_ns) / 1e9
        gx = sample.gx_rps - self._bias[0]
        gy = sample.gy_rps - self._bias[1]
        gz = sample.gz_rps - self._bias[2]
        accel_norm = sample.accel_norm_mps2
        accel_pitch = _accel_pitch_deg(sample.ax_mps2, sample.ay_mps2, sample.az_mps2)
        accel_roll = _accel_roll_deg(sample.ax_mps2, sample.ay_mps2, sample.az_mps2)

        if not self._initialised or dt_s <= 0.0 or dt_s > _MAX_INTEGRATION_STEP_S:
            self._pitch_deg = accel_pitch
            self._roll_deg = accel_roll
            self._initialised = True
        else:
            gyro_pitch = self._pitch_deg + gy * dt_s * RAD_TO_DEG
            gyro_roll = self._roll_deg + gx * dt_s * RAD_TO_DEG
            accel_weight = (1.0 - self.config.complementary_alpha) * self._accel_trust(
                accel_norm
            )
            self._pitch_deg = (1.0 - accel_weight) * gyro_pitch + accel_weight * accel_pitch
            self._roll_deg = (1.0 - accel_weight) * gyro_roll + accel_weight * accel_roll

        self._yaw_rate_rps = gz
        if dt_s > 0.0:
            self._integrate_sweep(gz, dt_s)
        self._push_window(timestamp_ns, gx, gy, gz, sample)
        stationary = self._is_stationary(timestamp_ns)
        self._update_drop_detector(timestamp_ns, accel_norm)

        state = self._build_state(timestamp_ns, stationary)
        self._last_state = state
        self._last_ts_ns = timestamp_ns
        return state

    def state(self, now_ns: int) -> IMUState:
        """Return the current state, or a neutral fallback when unusable."""

        if self._last_state is None or self._last_ts_ns is None:
            return self._neutral_state(int(now_ns))
        age_ns = int(now_ns) - self._last_ts_ns
        stale_ns = int(self.config.stale_after_ms * 1e6)
        if age_ns > stale_ns:
            return self._neutral_state(int(now_ns), elapsed_s=age_ns / 1e9)
        return self._last_state

    def project_horizontal_distance(
        self,
        raw_distance_m: float,
        state: IMUState,
        sensor_pitch_offset_deg: float = 0.0,
    ) -> float:
        """Project a shaft-aligned range onto the walking plane."""

        try:
            raw = float(raw_distance_m)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(raw) or raw <= 0.0:
            return 0.0
        relative = _clamp(
            state.relative_pitch_deg + float(sensor_pitch_offset_deg), -30.0, 30.0
        )
        return raw * math.cos(relative * DEG_TO_RAD)

    def is_likely_ground_echo(
        self,
        distance_m: float,
        state: IMUState,
        min_safety_floor_m: float = 0.35,
        sensor_pitch_offset_deg: float = 0.0,
    ) -> bool:
        """Report whether a range return matches the floor under the shaft.

        The 0.35 m safety floor keeps a genuinely close obstacle from ever being
        mistaken for the ground, regardless of how far the cane is tilted.
        """

        if not state.fresh:
            return False
        offset = float(sensor_pitch_offset_deg)
        effective_rel = state.relative_pitch_deg + offset
        effective_pitch = state.pitch_deg + offset
        expected: Optional[float] = None
        if abs(offset) <= 1e-6 and state.is_pitched_down:
            expected = state.expected_ground_distance_m
        elif (state.is_pitched_down or effective_rel >= 10.0) and 15.0 <= effective_pitch < 89.0:
            sine = math.sin(effective_pitch * DEG_TO_RAD)
            if sine > 1e-6:
                expected = self.config.sensor_height_m / sine
        if expected is None or expected <= 0.0:
            return False
        try:
            distance = float(distance_m)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(distance) or distance < min_safety_floor_m:
            return False
        return abs(distance - expected) / expected <= 0.18

    def compensate_vision_center_x(
        self,
        center_x: float,
        frame_width: int,
        age_s: float,
        state: IMUState,
    ) -> float:
        """Shift a stale image centre to where the object sits now.

        A cane yawing left at ``w`` rad/s moves a static object right in the
        frame by ``w * age`` radians, which maps to ``w * age * width / hfov``
        pixels.
        """

        if frame_width <= 0:
            raise ValueError("frame_width must be positive")
        center = float(center_x)
        if not math.isfinite(center):
            center = 0.0
        if not state.fresh:
            return _clamp(center, 0.0, float(frame_width))
        age = max(float(age_s), 0.0)
        if not math.isfinite(age):
            age = 0.0
        radians_per_pixel = (self.config.camera_hfov_deg * DEG_TO_RAD) / float(frame_width)
        if radians_per_pixel <= 0.0:
            return _clamp(center, 0.0, float(frame_width))
        pixels_per_radian = 1.0 / radians_per_pixel
        shift = state.yaw_rate_rps * age * pixels_per_radian
        return _clamp(center + shift, 0.0, float(frame_width))

    def _accel_trust(self, accel_norm_mps2: float) -> float:
        band = self.config.accel_trust_band_mps2
        error = abs(accel_norm_mps2 - GRAVITY_MPS2)
        if error <= band:
            return 1.0
        return max(0.0, 1.0 - (error - band) / band)

    def _integrate_sweep(self, yaw_rate_rps: float, dt_s: float) -> None:
        tau = self.config.sweep_tau_s
        decay = math.exp(-dt_s / tau)
        self._sweep_yaw_deg = (
            self._sweep_yaw_deg * decay
            + yaw_rate_rps * RAD_TO_DEG * tau * (1.0 - decay)
        )

    def _push_window(
        self,
        timestamp_ns: int,
        gx: float,
        gy: float,
        gz: float,
        sample: IMUSample,
    ) -> None:
        self._window.append(
            (
                timestamp_ns,
                gx,
                gy,
                gz,
                sample.ax_mps2,
                sample.ay_mps2,
                sample.az_mps2,
            )
        )
        dwell_ns = int(self.config.stationary_dwell_s * 1e9)
        while self._window and timestamp_ns - self._window[0][0] > dwell_ns:
            self._window.popleft()

    def _is_stationary(self, timestamp_ns: int) -> bool:
        if len(self._window) < 3:
            return False
        span_ns = timestamp_ns - self._window[0][0]
        if span_ns < int(self.config.stationary_dwell_s * 0.9 * 1e9):
            return False
        peak_gyro = 0.0
        norms: List[float] = []
        for _, gx, gy, gz, ax, ay, az in self._window:
            peak_gyro = max(peak_gyro, math.sqrt(gx * gx + gy * gy + gz * gz))
            norms.append(math.sqrt(ax * ax + ay * ay + az * az))
        if peak_gyro > self.config.stationary_gyro_rps:
            return False
        mean = sum(norms) / float(len(norms))
        variance = sum((norm - mean) ** 2 for norm in norms) / float(len(norms))
        return math.sqrt(variance) <= self.config.stationary_accel_std_mps2

    def _update_drop_detector(self, timestamp_ns: int, accel_norm_mps2: float) -> None:
        freefall_limit = self.config.freefall_g * GRAVITY_MPS2
        impact_limit = self.config.impact_g * GRAVITY_MPS2
        if accel_norm_mps2 < freefall_limit:
            if self._freefall_since_ns is None:
                self._freefall_since_ns = timestamp_ns
            return
        if self._freefall_since_ns is None:
            return
        if accel_norm_mps2 >= impact_limit:
            self._drop_detected = True
            self._freefall_since_ns = None
            return
        if timestamp_ns - self._freefall_since_ns > int(1.5e9):
            self._freefall_since_ns = None

    def _build_state(self, timestamp_ns: int, stationary: bool) -> IMUState:
        relative_pitch = self._pitch_deg - self._baseline_pitch_deg
        relative_roll = self._roll_deg - self._baseline_roll_deg
        lifted = relative_pitch <= self.config.lifted_pitch_deg
        rolled = abs(relative_roll) >= self.config.extreme_roll_deg
        pitched_down = relative_pitch > self.config.ground_tilt_warn_deg
        expected_ground: Optional[float] = None
        if pitched_down and 0.0 < self._pitch_deg < 90.0:
            sine = math.sin(self._pitch_deg * DEG_TO_RAD)
            if sine > 1e-6:
                expected_ground = self.config.sensor_height_m / sine
        return IMUState(
            timestamp_ns=timestamp_ns,
            fresh=True,
            calibrated=self._calibrated,
            pitch_deg=self._pitch_deg,
            roll_deg=self._roll_deg,
            relative_pitch_deg=relative_pitch,
            yaw_rate_rps=self._yaw_rate_rps,
            sweep_yaw_deg=self._sweep_yaw_deg,
            is_stationary=stationary,
            is_lifted_or_tilted=lifted or rolled,
            is_pitched_down=pitched_down,
            expected_ground_distance_m=expected_ground,
            drop_detected=self._drop_detected,
            gyro_bias_rps=self._bias,
        )

    def _neutral_state(self, now_ns: int, elapsed_s: float = 0.0) -> IMUState:
        sweep = 0.0
        if elapsed_s > 0.0:
            sweep = self._sweep_yaw_deg * math.exp(-elapsed_s / self.config.sweep_tau_s)
        return IMUState(
            timestamp_ns=now_ns,
            fresh=False,
            calibrated=self._calibrated,
            pitch_deg=0.0,
            roll_deg=0.0,
            relative_pitch_deg=0.0,
            yaw_rate_rps=0.0,
            sweep_yaw_deg=sweep,
            is_stationary=False,
            is_lifted_or_tilted=False,
            is_pitched_down=False,
            expected_ground_distance_m=None,
            drop_detected=self._drop_detected,
            gyro_bias_rps=self._bias,
        )
