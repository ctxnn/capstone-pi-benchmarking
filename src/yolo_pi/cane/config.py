"""Validated hardware and runtime configuration for the whole cane.

Every wiring fact (mux channels, GPIO pins, addresses) lives in one JSON file,
``configs/cane-hardware.json``, so the harness can be changed without touching
code. Unknown keys are rejected: a misspelt pin name must fail loudly at
start-up, not silently fall back to a default on a cane someone is holding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from ..fusion import FusionConfig
from ..types import Sector


class ConfigError(ValueError):
    """Raised when the hardware configuration is incomplete or inconsistent."""


def _as_int(value: Any, name: str) -> int:
    """Accept 112, "0x70" or "112" so addresses can be written the datasheet way."""

    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError as exc:
            raise ConfigError(f"{name} is not an integer: {value!r}") from exc
    raise ConfigError(f"{name} must be an integer, got {value!r}")


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    return float(value)


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {value!r}")
    return value


def _as_sector(value: Any, name: str) -> Sector:
    try:
        sector = Sector(str(value))
    except ValueError as exc:
        raise ConfigError(f"{name} must be left, center or right, got {value!r}") from exc
    if sector == Sector.UNKNOWN:
        raise ConfigError(f"{name} must be left, center or right, not unknown")
    return sector


def _check_keys(section: str, data: Mapping[str, Any], allowed: Tuple[str, ...]) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(
            f"unknown key(s) in '{section}': {', '.join(unknown)} "
            f"(allowed: {', '.join(allowed)})"
        )


def _mux_channel(value: Any, name: str) -> int:
    channel = _as_int(value, name)
    if not 0 <= channel <= 7:
        raise ConfigError(f"{name} must be a TCA9548A channel from 0 to 7, got {channel}")
    return channel


@dataclass(frozen=True)
class I2CConfig:
    mux_address: int = 0x70
    tof_address: int = 0x29
    haptic_address: int = 0x5A
    imu_address: int = 0x68


@dataclass(frozen=True)
class TofSensorSpec:
    sensor_id: str
    mux_channel: int
    sector: Sector


@dataclass(frozen=True)
class TofConfig:
    sensors: Tuple[TofSensorSpec, ...] = (
        TofSensorSpec("tof-left", 0, Sector.LEFT),
        TofSensorSpec("tof-center", 1, Sector.CENTER),
        TofSensorSpec("tof-right", 2, Sector.RIGHT),
    )
    timing_budget_us: int = 33000
    # A VL53L0X that sees nothing reports a large "out of range" code. That is a
    # valid *clear* reading, not a fault, so it is mapped to this distance.
    clear_distance_m: float = 2.0
    out_of_range_mm: int = 8000
    io_timeout_s: float = 0.25
    poll_interval_ms: float = 5.0
    reinit_after_errors: int = 5
    reinit_backoff_s: float = 1.0


ROLE_FORWARD = "forward"
ROLE_GROUND = "ground"
ROLE_OVERHEAD = "overhead"
ULTRASONIC_ROLES = (ROLE_FORWARD, ROLE_GROUND, ROLE_OVERHEAD)


@dataclass(frozen=True)
class UltrasonicSensorSpec:
    """One HC-SR04.

    ``forward``: the sensor tilted to look along the walking path (the cane
    itself leans, so this is the one that points "straight ahead" in the world).
    ``ground``: the second center sensor angled slightly toward the ground side
    (detects low obstacles/curbs in the center corridor and gets extra downward
    pitch offset in IMU ground-echo gating).
    ``overhead``: optional upward/head-height role that warns on the centre
    motor without mixing with ground-level ranges.
    """

    sensor_id: str
    trig_gpio: int
    echo_gpio: int
    sector: Sector
    role: str = ROLE_FORWARD
    pitch_offset_deg: float = 0.0


@dataclass(frozen=True)
class UltrasonicConfig:
    sensors: Tuple[UltrasonicSensorSpec, ...] = (
        UltrasonicSensorSpec("us-1", 23, 24, Sector.CENTER, ROLE_FORWARD, 0.0),
        UltrasonicSensorSpec("us-2", 27, 22, Sector.CENTER, ROLE_GROUND, 8.0),
    )
    max_range_m: float = 4.0
    min_range_m: float = 0.02
    # Sensors are fired one at a time so one sensor never hears another's burst.
    spacing_ms: float = 60.0
    speed_of_sound_m_s: float = 343.0
    gpio_chip: Optional[int] = None
    reinit_after_errors: int = 5
    reinit_backoff_s: float = 1.0


@dataclass(frozen=True)
class HapticsConfig:
    """One DRV2605L drives all three ERM motors in parallel.

    The motors cannot be switched individually, so direction is carried by the
    rhythm, not by which motor buzzes: a group of ``taps_left`` / ``taps_center``
    / ``taps_right`` short taps, repeated. The closer the hazard, the faster
    the taps and the shorter the pause between groups (tempo runs from the
    ``*_ms`` values at the warning distance to the ``*_fast_ms`` values at
    ``urgent_distance_m`` or closer).
    """

    # None: the driver sits on the main I2C bus. A number: on that TCA9548A channel.
    driver_mux_channel: Optional[int] = None
    motor_voltage_v: float = 3.0
    # Fusion asks for an intensity from 0 to 1. A small ERM motor does not spin
    # below a start voltage, so non-zero intensities are lifted to this floor.
    amplitude_floor: float = 0.30
    taps_left: int = 1
    taps_center: int = 2
    taps_right: int = 3
    tap_on_ms: float = 90.0
    tap_on_fast_ms: float = 60.0
    tap_off_ms: float = 110.0
    tap_off_fast_ms: float = 60.0
    group_gap_ms: float = 600.0
    group_gap_fast_ms: float = 150.0
    sector_gap_ms: float = 250.0  # between the groups when two sides are alerting at once
    long_pulse_ms: float = 250.0  # overhead (two long pulses) and sensor fault (one, weak)
    urgent_distance_m: float = 0.15
    command_timeout_s: float = 1.0
    startup_sweep: bool = True

    def taps_for(self, sector: Sector) -> int:
        return {Sector.LEFT: self.taps_left, Sector.CENTER: self.taps_center,
                Sector.RIGHT: self.taps_right}.get(sector, 0)


@dataclass(frozen=True)
class ImuConfig:
    enabled: bool = True
    mux_channel: Optional[int] = None
    calibration_samples: int = 200
    calibration_max_gyro_dps: float = 8.0
    calibration_attempts: int = 5
    still_gyro_dps: float = 3.0
    still_accel_g: float = 0.08
    still_after_s: float = 3.0
    poll_interval_ms: float = 20.0
    # Sudden-tilt pulse: the only thing the IMU ever vibrates for.
    jolt_enabled: bool = True
    jolt_angle_deg: float = 25.0
    jolt_window_s: float = 0.25
    jolt_cooldown_s: float = 1.0
    jolt_pulse_ms: float = 300.0
    jolt_intensity: float = 0.8
    gravity_filter_alpha: float = 0.98
    jolt_accel_tolerance_g: float = 0.25


# Typical real-world heights (metres) of the 15 trained classes. They only scale
# the camera's distance estimate, so rough values are fine, but they are
# estimates and should be tuned from a measured walk-up test.
DEFAULT_CLASS_HEIGHTS_M: Tuple[Tuple[str, float], ...] = (
    ("person", 1.7), ("vehicle", 1.5), ("bicycle_motorcycle", 1.1), ("pole", 2.5),
    ("tree", 3.0), ("stairs", 0.6), ("curb", 0.15), ("barrier", 1.0), ("cone", 0.7),
    ("dog", 0.5), ("signboard", 1.5), ("door", 2.0), ("slope", 0.5),
    ("overhead_obstacle", 0.6), ("generic_obstacle", 0.8),
)


@dataclass(frozen=True)
class PriorityConfig:
    """Who decides at which distance, and how the far / overhead alerts feel.

    Up to ``fusion.warning_distance_m`` the ToF and forward ultrasonic sensors
    decide. From there out to ``far_max_m`` the camera decides. The handoff
    point is deliberately the warning distance so there is no gap between them.
    """

    far_max_m: float = 4.0
    vision_min_confidence: float = 0.5
    confirm_frames: int = 2
    far_intensity: float = 0.45
    overhead_warning_m: float = 1.0
    overhead_critical_m: float = 0.5
    vertical_fov_deg: float = 48.8  # IMX219 full-frame mode; calibrate for the real lens
    default_height_m: float = 0.8
    edge_margin_px: int = 4
    class_heights_m: Tuple[Tuple[str, float], ...] = DEFAULT_CLASS_HEIGHTS_M
    blind_sector_max_distance_m: float = 2.0
    blind_sector_critical_confidence: float = 0.60
    blind_sector_critical_frames: int = 3

    def height_for(self, class_name: str) -> float:
        for name, height in self.class_heights_m:
            if name == class_name:
                return height
        return self.default_height_m


@dataclass(frozen=True)
class VisionConfig:
    enabled: bool = True
    model: str = "artifacts/models/cane-v1-fast/ov-fp32-416/best_openvino_model"
    imgsz: int = 0  # 0 reads the size from the exported model's metadata
    confidence: float = 0.35
    threads: int = 3
    width: int = 640
    height: int = 480
    camera_fps: float = 30.0
    camera_num: int = 0
    center_fraction: float = 1 / 3
    idle_when_still: bool = True
    idle_interval_s: float = 1.0
    restart_backoff_s: float = 5.0


@dataclass(frozen=True)
class RuntimeConfig:
    loop_hz: float = 50.0
    release_hold_ms: float = 250.0
    status_interval_s: float = 5.0
    event_log: Optional[str] = "logs/cane-events.jsonl"


@dataclass(frozen=True)
class CaneConfig:
    i2c: I2CConfig = I2CConfig()
    tof: TofConfig = TofConfig()
    ultrasonic: UltrasonicConfig = UltrasonicConfig()
    haptics: HapticsConfig = HapticsConfig()
    imu: ImuConfig = ImuConfig()
    vision: VisionConfig = VisionConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    fusion: FusionConfig = FusionConfig()
    priority: PriorityConfig = PriorityConfig()

    def validate(self) -> None:
        ids = [spec.sensor_id for spec in self.tof.sensors]
        ids += [spec.sensor_id for spec in self.ultrasonic.sensors]
        if len(set(ids)) != len(ids):
            raise ConfigError("sensor ids must be unique across ToF and ultrasonic sensors")
        if not ids:
            raise ConfigError("at least one range sensor must be configured")

        tof_channels = [spec.mux_channel for spec in self.tof.sensors]
        if len(set(tof_channels)) != len(tof_channels):
            raise ConfigError(
                "ToF sensors share address 0x29, so each needs its own mux channel"
            )
        if self.haptics.driver_mux_channel is not None and not 0 <= self.haptics.driver_mux_channel <= 7:
            raise ConfigError("haptics.driver_mux_channel must be null or a channel from 0 to 7")
        hp = self.haptics
        taps = (hp.taps_left, hp.taps_center, hp.taps_right)
        if any(not 1 <= count <= 6 for count in taps):
            raise ConfigError("haptics tap counts must be between 1 and 6")
        if len(set(taps)) != 3:
            raise ConfigError(
                "haptics.taps_left / taps_center / taps_right must all differ, "
                "or the user could not tell the directions apart"
            )
        for name in ("tap_on_ms", "tap_on_fast_ms", "tap_off_ms", "tap_off_fast_ms",
                     "group_gap_ms", "group_gap_fast_ms", "sector_gap_ms", "long_pulse_ms"):
            if getattr(hp, name) < 20.0:
                raise ConfigError(f"haptics.{name} must be at least 20 ms")
        if hp.group_gap_fast_ms < 1.8 * hp.tap_off_fast_ms:
            raise ConfigError(
                "haptics.group_gap_fast_ms must be well above tap_off_fast_ms "
                "(at least 1.8x), or tap groups blur into one buzz and direction is lost"
            )
        if hp.group_gap_ms < hp.group_gap_fast_ms or hp.tap_on_ms < hp.tap_on_fast_ms \
                or hp.tap_off_ms < hp.tap_off_fast_ms:
            raise ConfigError("the 'fast' haptic timings must not be slower than the normal ones")
        if hp.urgent_distance_m <= 0 or hp.urgent_distance_m >= self.fusion.warning_distance_m:
            raise ConfigError("haptics.urgent_distance_m must be in (0, fusion.warning_distance_m)")

        pins = []
        for spec in self.ultrasonic.sensors:
            pins += [spec.trig_gpio, spec.echo_gpio]
        if len(set(pins)) != len(pins):
            raise ConfigError("ultrasonic TRIG and ECHO GPIO numbers must all be different")
        if any(pin in (2, 3) for pin in pins):
            raise ConfigError("GPIO2 and GPIO3 are the I2C bus and cannot be used for ranging")

        if not 0.0 <= self.haptics.amplitude_floor < 1.0:
            raise ConfigError("haptics.amplitude_floor must be in [0, 1)")
        if not 0.5 <= self.haptics.motor_voltage_v <= 5.0:
            raise ConfigError("haptics.motor_voltage_v must be between 0.5 and 5.0")
        if self.haptics.command_timeout_s <= 0:
            raise ConfigError("haptics.command_timeout_s must be positive")
        if self.runtime.loop_hz <= 0:
            raise ConfigError("runtime.loop_hz must be positive")
        if self.runtime.loop_hz * self.haptics.command_timeout_s < 5.0:
            raise ConfigError(
                "runtime.loop_hz must give at least five commands per "
                "haptics.command_timeout_s, or motors would drop out between commands"
            )
        if self.tof.clear_distance_m <= self.fusion.warning_distance_m:
            raise ConfigError(
                "tof.clear_distance_m must exceed fusion.warning_distance_m, "
                "or an empty path would read as a hazard"
            )
        if self.ultrasonic.max_range_m <= self.fusion.warning_distance_m:
            raise ConfigError(
                "ultrasonic.max_range_m must exceed fusion.warning_distance_m"
            )
        if self.vision.threads <= 0:
            raise ConfigError("vision.threads must be positive")
        if not self.tof.sensors and not any(
            spec.role in (ROLE_FORWARD, ROLE_GROUND) for spec in self.ultrasonic.sensors
        ):
            raise ConfigError(
                "at least one forward-looking range sensor (ToF or a 'forward'/'ground' ultrasonic) is required"
            )
        if sum(1 for spec in self.ultrasonic.sensors if spec.role == ROLE_OVERHEAD) > 1:
            raise ConfigError("only one ultrasonic sensor can have the 'overhead' role")
        pr = self.priority
        if pr.far_max_m <= self.fusion.warning_distance_m:
            raise ConfigError(
                "priority.far_max_m must exceed fusion.warning_distance_m, "
                "or the camera would have no distance band to cover"
            )
        if not 0.0 < pr.overhead_critical_m <= pr.overhead_warning_m:
            raise ConfigError("priority overhead distances must satisfy 0 < critical <= warning")
        if not 0.0 <= pr.vision_min_confidence <= 1.0:
            raise ConfigError("priority.vision_min_confidence must be in [0, 1]")
        if pr.confirm_frames < 1:
            raise ConfigError("priority.confirm_frames must be at least 1")
        if not 0.0 < pr.far_intensity <= 1.0:
            raise ConfigError("priority.far_intensity must be in (0, 1]")
        if not 10.0 <= pr.vertical_fov_deg <= 120.0:
            raise ConfigError("priority.vertical_fov_deg must be between 10 and 120")
        if pr.default_height_m <= 0 or any(height <= 0 for _, height in pr.class_heights_m):
            raise ConfigError("priority class heights must be positive")
        if pr.blind_sector_max_distance_m <= 0.0:
            raise ConfigError("priority.blind_sector_max_distance_m must be positive")
        if not 0.0 <= pr.blind_sector_critical_confidence <= 1.0:
            raise ConfigError("priority.blind_sector_critical_confidence must be in [0, 1]")
        if pr.blind_sector_critical_frames < 1:
            raise ConfigError("priority.blind_sector_critical_frames must be at least 1")
        im = self.imu
        if not 5.0 <= im.jolt_angle_deg <= 120.0:
            raise ConfigError("imu.jolt_angle_deg must be between 5 and 120")
        if not 0.05 <= im.jolt_window_s <= 2.0:
            raise ConfigError("imu.jolt_window_s must be between 0.05 and 2 seconds")
        if im.jolt_cooldown_s < 0 or im.jolt_pulse_ms <= 0:
            raise ConfigError("imu.jolt_cooldown_s must be >= 0 and imu.jolt_pulse_ms positive")
        if not 0.0 < im.jolt_intensity <= 1.0:
            raise ConfigError("imu.jolt_intensity must be in (0, 1]")
        if not 0.0 < im.gravity_filter_alpha < 1.0:
            raise ConfigError("imu.gravity_filter_alpha must be strictly between 0 and 1")

    def describe(self) -> Dict[str, Any]:
        """A small, loggable summary of the wiring this config describes."""

        return {
            "mux_address": hex(self.i2c.mux_address),
            "tof": {s.sensor_id: {"mux_channel": s.mux_channel, "sector": s.sector.value}
                    for s in self.tof.sensors},
            "ultrasonic": {
                s.sensor_id: {"trig": s.trig_gpio, "echo": s.echo_gpio,
                    "sector": s.sector.value, "role": s.role}
                for s in self.ultrasonic.sensors
            },
            "haptics": {
                "driver": hex(self.i2c.haptic_address),
                "mux_channel": self.haptics.driver_mux_channel,
                "taps": {"left": self.haptics.taps_left, "center": self.haptics.taps_center,
                         "right": self.haptics.taps_right},
            },
            "imu": {"enabled": self.imu.enabled, "address": hex(self.i2c.imu_address),
                    "mux_channel": self.imu.mux_channel},
            "vision": {"enabled": self.vision.enabled, "model": self.vision.model},
        }


def _section(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = root.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be an object")
    return value


def _parse_i2c(data: Mapping[str, Any]) -> I2CConfig:
    _check_keys("i2c", data, ("mux_address", "tof_address", "haptic_address", "imu_address"))
    base = I2CConfig()
    return I2CConfig(
        mux_address=_as_int(data.get("mux_address", base.mux_address), "i2c.mux_address"),
        tof_address=_as_int(data.get("tof_address", base.tof_address), "i2c.tof_address"),
        haptic_address=_as_int(data.get("haptic_address", base.haptic_address), "i2c.haptic_address"),
        imu_address=_as_int(data.get("imu_address", base.imu_address), "i2c.imu_address"),
    )


def _parse_tof(data: Mapping[str, Any]) -> TofConfig:
    allowed = (
        "sensors", "timing_budget_us", "clear_distance_m", "out_of_range_mm",
        "io_timeout_s", "poll_interval_ms", "reinit_after_errors", "reinit_backoff_s",
    )
    _check_keys("tof", data, allowed)
    base = TofConfig()
    sensors = base.sensors
    if "sensors" in data:
        parsed = []
        for index, item in enumerate(data["sensors"]):
            _check_keys(f"tof.sensors[{index}]", item, ("id", "mux_channel", "sector"))
            parsed.append(
                TofSensorSpec(
                    str(item["id"]),
                    _mux_channel(item["mux_channel"], f"tof.sensors[{index}].mux_channel"),
                    _as_sector(item["sector"], f"tof.sensors[{index}].sector"),
                )
            )
        sensors = tuple(parsed)
    return TofConfig(
        sensors=sensors,
        timing_budget_us=_as_int(data.get("timing_budget_us", base.timing_budget_us), "tof.timing_budget_us"),
        clear_distance_m=_as_float(data.get("clear_distance_m", base.clear_distance_m), "tof.clear_distance_m"),
        out_of_range_mm=_as_int(data.get("out_of_range_mm", base.out_of_range_mm), "tof.out_of_range_mm"),
        io_timeout_s=_as_float(data.get("io_timeout_s", base.io_timeout_s), "tof.io_timeout_s"),
        poll_interval_ms=_as_float(data.get("poll_interval_ms", base.poll_interval_ms), "tof.poll_interval_ms"),
        reinit_after_errors=_as_int(data.get("reinit_after_errors", base.reinit_after_errors), "tof.reinit_after_errors"),
        reinit_backoff_s=_as_float(data.get("reinit_backoff_s", base.reinit_backoff_s), "tof.reinit_backoff_s"),
    )


def _parse_ultrasonic(data: Mapping[str, Any]) -> UltrasonicConfig:
    allowed = (
        "sensors", "max_range_m", "min_range_m", "spacing_ms", "speed_of_sound_m_s",
        "gpio_chip", "reinit_after_errors", "reinit_backoff_s",
    )
    _check_keys("ultrasonic", data, allowed)
    base = UltrasonicConfig()
    sensors = base.sensors
    if "sensors" in data:
        parsed = []
        for index, item in enumerate(data["sensors"]):
            _check_keys(
                f"ultrasonic.sensors[{index}]", item,
                ("id", "trig_gpio", "echo_gpio", "sector", "role", "pitch_offset_deg"),
            )
            role = str(item.get("role", ROLE_FORWARD))
            if role not in ULTRASONIC_ROLES:
                raise ConfigError(
                    f"ultrasonic.sensors[{index}].role must be one of {ULTRASONIC_ROLES}, got {role!r}"
                )
            default_offset = 8.0 if role == ROLE_GROUND else 0.0
            pitch_offset = _as_float(
                item.get("pitch_offset_deg", default_offset),
                f"ultrasonic.sensors[{index}].pitch_offset_deg",
            )
            parsed.append(
                UltrasonicSensorSpec(
                    str(item["id"]),
                    _as_int(item["trig_gpio"], f"ultrasonic.sensors[{index}].trig_gpio"),
                    _as_int(item["echo_gpio"], f"ultrasonic.sensors[{index}].echo_gpio"),
                    _as_sector(item["sector"], f"ultrasonic.sensors[{index}].sector"),
                    role,
                    pitch_offset,
                )
            )
        sensors = tuple(parsed)
    chip = data.get("gpio_chip", base.gpio_chip)
    return UltrasonicConfig(
        sensors=sensors,
        max_range_m=_as_float(data.get("max_range_m", base.max_range_m), "ultrasonic.max_range_m"),
        min_range_m=_as_float(data.get("min_range_m", base.min_range_m), "ultrasonic.min_range_m"),
        spacing_ms=_as_float(data.get("spacing_ms", base.spacing_ms), "ultrasonic.spacing_ms"),
        speed_of_sound_m_s=_as_float(data.get("speed_of_sound_m_s", base.speed_of_sound_m_s), "ultrasonic.speed_of_sound_m_s"),
        gpio_chip=None if chip is None else _as_int(chip, "ultrasonic.gpio_chip"),
        reinit_after_errors=_as_int(data.get("reinit_after_errors", base.reinit_after_errors), "ultrasonic.reinit_after_errors"),
        reinit_backoff_s=_as_float(data.get("reinit_backoff_s", base.reinit_backoff_s), "ultrasonic.reinit_backoff_s"),
    )


def _parse_haptics(data: Mapping[str, Any]) -> HapticsConfig:
    allowed = (
        "driver_mux_channel", "motor_voltage_v", "amplitude_floor", "taps_left", "taps_center",
        "taps_right", "tap_on_ms", "tap_on_fast_ms", "tap_off_ms", "tap_off_fast_ms",
        "group_gap_ms", "group_gap_fast_ms", "sector_gap_ms", "long_pulse_ms",
        "urgent_distance_m", "command_timeout_s", "startup_sweep",
    )
    _check_keys("haptics", data, allowed)
    base = HapticsConfig()
    channel = data.get("driver_mux_channel", base.driver_mux_channel)
    floats = (
        "motor_voltage_v", "amplitude_floor", "tap_on_ms", "tap_on_fast_ms", "tap_off_ms",
        "tap_off_fast_ms", "group_gap_ms", "group_gap_fast_ms", "sector_gap_ms", "long_pulse_ms",
        "urgent_distance_m", "command_timeout_s",
    )
    values: Dict[str, Any] = {
        name: _as_float(data.get(name, getattr(base, name)), f"haptics.{name}") for name in floats
    }
    for name in ("taps_left", "taps_center", "taps_right"):
        values[name] = _as_int(data.get(name, getattr(base, name)), f"haptics.{name}")
    return HapticsConfig(
        driver_mux_channel=None if channel is None else _mux_channel(channel, "haptics.driver_mux_channel"),
        startup_sweep=_as_bool(data.get("startup_sweep", base.startup_sweep), "haptics.startup_sweep"),
        **values,
    )


def _parse_imu(data: Mapping[str, Any]) -> ImuConfig:
    allowed = (
        "enabled", "mux_channel", "calibration_samples", "calibration_max_gyro_dps",
        "calibration_attempts", "still_gyro_dps", "still_accel_g", "still_after_s",
        "poll_interval_ms", "jolt_enabled", "jolt_angle_deg", "jolt_window_s",
        "jolt_cooldown_s", "jolt_pulse_ms", "jolt_intensity", "gravity_filter_alpha",
        "jolt_accel_tolerance_g",
    )
    _check_keys("imu", data, allowed)
    base = ImuConfig()
    channel = data.get("mux_channel", base.mux_channel)
    return ImuConfig(
        enabled=_as_bool(data.get("enabled", base.enabled), "imu.enabled"),
        mux_channel=None if channel is None else _mux_channel(channel, "imu.mux_channel"),
        calibration_samples=_as_int(data.get("calibration_samples", base.calibration_samples), "imu.calibration_samples"),
        calibration_max_gyro_dps=_as_float(data.get("calibration_max_gyro_dps", base.calibration_max_gyro_dps), "imu.calibration_max_gyro_dps"),
        calibration_attempts=_as_int(data.get("calibration_attempts", base.calibration_attempts), "imu.calibration_attempts"),
        still_gyro_dps=_as_float(data.get("still_gyro_dps", base.still_gyro_dps), "imu.still_gyro_dps"),
        still_accel_g=_as_float(data.get("still_accel_g", base.still_accel_g), "imu.still_accel_g"),
        still_after_s=_as_float(data.get("still_after_s", base.still_after_s), "imu.still_after_s"),
        poll_interval_ms=_as_float(data.get("poll_interval_ms", base.poll_interval_ms), "imu.poll_interval_ms"),
        jolt_enabled=_as_bool(data.get("jolt_enabled", base.jolt_enabled), "imu.jolt_enabled"),
        jolt_angle_deg=_as_float(data.get("jolt_angle_deg", base.jolt_angle_deg), "imu.jolt_angle_deg"),
        jolt_window_s=_as_float(data.get("jolt_window_s", base.jolt_window_s), "imu.jolt_window_s"),
        jolt_cooldown_s=_as_float(data.get("jolt_cooldown_s", base.jolt_cooldown_s), "imu.jolt_cooldown_s"),
        jolt_pulse_ms=_as_float(data.get("jolt_pulse_ms", base.jolt_pulse_ms), "imu.jolt_pulse_ms"),
        jolt_intensity=_as_float(data.get("jolt_intensity", base.jolt_intensity), "imu.jolt_intensity"),
        gravity_filter_alpha=_as_float(data.get("gravity_filter_alpha", base.gravity_filter_alpha), "imu.gravity_filter_alpha"),
        jolt_accel_tolerance_g=_as_float(data.get("jolt_accel_tolerance_g", base.jolt_accel_tolerance_g), "imu.jolt_accel_tolerance_g"),
    )


def _parse_vision(data: Mapping[str, Any]) -> VisionConfig:
    allowed = (
        "enabled", "model", "imgsz", "confidence", "threads", "width", "height",
        "camera_fps", "camera_num", "center_fraction", "idle_when_still",
        "idle_interval_s", "restart_backoff_s",
    )
    _check_keys("vision", data, allowed)
    base = VisionConfig()
    return VisionConfig(
        enabled=_as_bool(data.get("enabled", base.enabled), "vision.enabled"),
        model=str(data.get("model", base.model)),
        imgsz=_as_int(data.get("imgsz", base.imgsz), "vision.imgsz"),
        confidence=_as_float(data.get("confidence", base.confidence), "vision.confidence"),
        threads=_as_int(data.get("threads", base.threads), "vision.threads"),
        width=_as_int(data.get("width", base.width), "vision.width"),
        height=_as_int(data.get("height", base.height), "vision.height"),
        camera_fps=_as_float(data.get("camera_fps", base.camera_fps), "vision.camera_fps"),
        camera_num=_as_int(data.get("camera_num", base.camera_num), "vision.camera_num"),
        center_fraction=_as_float(data.get("center_fraction", base.center_fraction), "vision.center_fraction"),
        idle_when_still=_as_bool(data.get("idle_when_still", base.idle_when_still), "vision.idle_when_still"),
        idle_interval_s=_as_float(data.get("idle_interval_s", base.idle_interval_s), "vision.idle_interval_s"),
        restart_backoff_s=_as_float(data.get("restart_backoff_s", base.restart_backoff_s), "vision.restart_backoff_s"),
    )


def _parse_runtime(data: Mapping[str, Any]) -> RuntimeConfig:
    allowed = ("loop_hz", "release_hold_ms", "status_interval_s", "event_log")
    _check_keys("runtime", data, allowed)
    base = RuntimeConfig()
    log = data.get("event_log", base.event_log)
    return RuntimeConfig(
        loop_hz=_as_float(data.get("loop_hz", base.loop_hz), "runtime.loop_hz"),
        release_hold_ms=_as_float(data.get("release_hold_ms", base.release_hold_ms), "runtime.release_hold_ms"),
        status_interval_s=_as_float(data.get("status_interval_s", base.status_interval_s), "runtime.status_interval_s"),
        event_log=None if log is None else str(log),
    )


def _parse_priority(data: Mapping[str, Any]) -> PriorityConfig:
    allowed = (
        "far_max_m", "vision_min_confidence", "confirm_frames", "far_intensity",
        "overhead_warning_m", "overhead_critical_m", "vertical_fov_deg",
        "default_height_m", "edge_margin_px", "class_heights_m",
        "blind_sector_max_distance_m", "blind_sector_critical_confidence",
        "blind_sector_critical_frames",
    )
    _check_keys("priority", data, allowed)
    base = PriorityConfig()
    heights = dict(base.class_heights_m)
    extra = data.get("class_heights_m", {})
    if not isinstance(extra, Mapping):
        raise ConfigError("priority.class_heights_m must be an object of class name to metres")
    for name, value in extra.items():
        heights[str(name)] = _as_float(value, f"priority.class_heights_m.{name}")
    return PriorityConfig(
        far_max_m=_as_float(data.get("far_max_m", base.far_max_m), "priority.far_max_m"),
        vision_min_confidence=_as_float(data.get("vision_min_confidence", base.vision_min_confidence), "priority.vision_min_confidence"),
        confirm_frames=_as_int(data.get("confirm_frames", base.confirm_frames), "priority.confirm_frames"),
        far_intensity=_as_float(data.get("far_intensity", base.far_intensity), "priority.far_intensity"),
        overhead_warning_m=_as_float(data.get("overhead_warning_m", base.overhead_warning_m), "priority.overhead_warning_m"),
        overhead_critical_m=_as_float(data.get("overhead_critical_m", base.overhead_critical_m), "priority.overhead_critical_m"),
        vertical_fov_deg=_as_float(data.get("vertical_fov_deg", base.vertical_fov_deg), "priority.vertical_fov_deg"),
        default_height_m=_as_float(data.get("default_height_m", base.default_height_m), "priority.default_height_m"),
        edge_margin_px=_as_int(data.get("edge_margin_px", base.edge_margin_px), "priority.edge_margin_px"),
        class_heights_m=tuple(sorted(heights.items())),
        blind_sector_max_distance_m=_as_float(
            data.get("blind_sector_max_distance_m", base.blind_sector_max_distance_m),
            "priority.blind_sector_max_distance_m",
        ),
        blind_sector_critical_confidence=_as_float(
            data.get("blind_sector_critical_confidence", base.blind_sector_critical_confidence),
            "priority.blind_sector_critical_confidence",
        ),
        blind_sector_critical_frames=_as_int(
            data.get("blind_sector_critical_frames", base.blind_sector_critical_frames),
            "priority.blind_sector_critical_frames",
        ),
    )


def _parse_fusion(data: Mapping[str, Any]) -> FusionConfig:
    allowed = (
        "warning_distance_m", "critical_distance_m", "sensor_stale_after_ms",
        "vision_stale_after_ms", "vision_confidence",
    )
    _check_keys("fusion", data, allowed)
    base = FusionConfig()
    try:
        return FusionConfig(
            warning_distance_m=_as_float(data.get("warning_distance_m", base.warning_distance_m), "fusion.warning_distance_m"),
            critical_distance_m=_as_float(data.get("critical_distance_m", base.critical_distance_m), "fusion.critical_distance_m"),
            sensor_stale_after_ms=_as_float(data.get("sensor_stale_after_ms", base.sensor_stale_after_ms), "fusion.sensor_stale_after_ms"),
            vision_stale_after_ms=_as_float(data.get("vision_stale_after_ms", base.vision_stale_after_ms), "fusion.vision_stale_after_ms"),
            vision_confidence=_as_float(data.get("vision_confidence", base.vision_confidence), "fusion.vision_confidence"),
        )
    except ValueError as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"invalid fusion settings: {exc}") from exc


def config_from_dict(payload: Mapping[str, Any]) -> CaneConfig:
    """Build and validate a :class:`CaneConfig` from a parsed JSON object."""

    if not isinstance(payload, Mapping):
        raise ConfigError("the configuration root must be a JSON object")
    allowed = ("comment", "i2c", "tof", "ultrasonic", "haptics", "imu", "vision", "runtime", "fusion", "priority")
    _check_keys("configuration", payload, allowed)
    config = CaneConfig(
        i2c=_parse_i2c(_section(payload, "i2c")),
        tof=_parse_tof(_section(payload, "tof")),
        ultrasonic=_parse_ultrasonic(_section(payload, "ultrasonic")),
        haptics=_parse_haptics(_section(payload, "haptics")),
        imu=_parse_imu(_section(payload, "imu")),
        vision=_parse_vision(_section(payload, "vision")),
        runtime=_parse_runtime(_section(payload, "runtime")),
        fusion=_parse_fusion(_section(payload, "fusion")),
        priority=_parse_priority(_section(payload, "priority")),
    )
    config.validate()
    return config


def load_config(path: Optional[Union[str, Path]] = None) -> CaneConfig:
    """Load a configuration file, or return the built-in defaults when no path is given."""

    if path is None:
        config = CaneConfig()
        config.validate()
        return config
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    return config_from_dict(payload)
