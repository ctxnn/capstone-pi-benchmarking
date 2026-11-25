"""Validated hardware wiring contracts for the Raspberry Pi 5 smart cane.

Every value here mirrors the physical bill of materials in the capstone
report: three VL53L0X time-of-flight sensors and three DRV2605L haptic drivers
behind one TCA9548A multiplexer, two HC-SR04 ultrasonic sensors on RP1 GPIO,
one MPU-6050 inertial sensor, and an Arducam IMX219 camera. Loading is strict:
unknown JSON keys, duplicate bus addresses, and incomplete left/center/right
coverage are rejected before any driver touches the I2C bus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..types import Sector

I2C_MIN_ADDRESS = 0x08
I2C_MAX_ADDRESS = 0x77
MUX_MIN_CHANNEL = 0
MUX_MAX_CHANNEL = 7
WARNING_SECTORS = (Sector.LEFT, Sector.CENTER, Sector.RIGHT)


class HardwareConfigError(ValueError):
    """Raised when a hardware description is incomplete, ambiguous, or unsafe."""


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HardwareConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HardwareConfigError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise HardwareConfigError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _require_float(
    value: Any, name: str, minimum: float, maximum: float, *, inclusive_max: bool = True
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HardwareConfigError(f"{name} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise HardwareConfigError(f"{name} must be finite")
    upper_ok = number <= maximum if inclusive_max else number < maximum
    if number < minimum or not upper_ok:
        bound = "]" if inclusive_max else ")"
        raise HardwareConfigError(f"{name} must be in [{minimum}, {maximum}{bound}")
    return number


def _require_sector(value: Any, name: str = "sector") -> Sector:
    try:
        sector = value if isinstance(value, Sector) else Sector(value)
    except ValueError as exc:
        raise HardwareConfigError(f"{name} must be left, center, or right") from exc
    if sector not in WARNING_SECTORS:
        raise HardwareConfigError(f"{name} must be left, center, or right")
    return sector


def _require_channel(value: Any, name: str, *, optional: bool = False) -> Optional[int]:
    if value is None and optional:
        return None
    return _require_int(value, name, MUX_MIN_CHANNEL, MUX_MAX_CHANNEL)


def _require_bus_address(value: Any, name: str) -> int:
    return _require_int(value, name, I2C_MIN_ADDRESS, I2C_MAX_ADDRESS)


@dataclass(frozen=True)
class ToFSensorSpec:
    """One VL53L0X on a dedicated TCA9548A channel."""

    sensor_id: str
    sector: Sector
    mux_channel: int
    i2c_address: int = 0x29
    timing_budget_ms: float = 33.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_id", _require_text(self.sensor_id, "tof sensor_id"))
        object.__setattr__(self, "sector", _require_sector(self.sector, f"{self.sensor_id} sector"))
        _require_channel(self.mux_channel, f"{self.sensor_id} mux_channel")
        _require_bus_address(self.i2c_address, f"{self.sensor_id} i2c_address")
        _require_float(self.timing_budget_ms, f"{self.sensor_id} timing_budget_ms", 20.0, 200.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "sector": self.sector.value,
            "mux_channel": self.mux_channel,
            "i2c_address": self.i2c_address,
            "timing_budget_ms": self.timing_budget_ms,
        }


@dataclass(frozen=True)
class UltrasonicSensorSpec:
    """One HC-SR04 ping/echo pair on RP1 GPIO."""

    sensor_id: str
    sector: Sector
    trigger_gpio: int
    echo_gpio: int
    min_range_m: float = 0.04
    max_range_m: float = 3.50
    pitch_offset_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_id", _require_text(self.sensor_id, "ultrasonic sensor_id"))
        object.__setattr__(self, "sector", _require_sector(self.sector, f"{self.sensor_id} sector"))
        _require_int(self.trigger_gpio, f"{self.sensor_id} trigger_gpio", 0, 53)
        _require_int(self.echo_gpio, f"{self.sensor_id} echo_gpio", 0, 53)
        if self.trigger_gpio == self.echo_gpio:
            raise HardwareConfigError(f"{self.sensor_id} trigger and echo pins must differ")
        _require_float(self.min_range_m, f"{self.sensor_id} min_range_m", 0.01, 1.0)
        _require_float(self.max_range_m, f"{self.sensor_id} max_range_m", 0.5, 6.0)
        if self.min_range_m >= self.max_range_m:
            raise HardwareConfigError(f"{self.sensor_id} min_range_m must be below max_range_m")
        _require_float(self.pitch_offset_deg, f"{self.sensor_id} pitch_offset_deg", -45.0, 45.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "sector": self.sector.value,
            "trigger_gpio": self.trigger_gpio,
            "echo_gpio": self.echo_gpio,
            "min_range_m": self.min_range_m,
            "max_range_m": self.max_range_m,
            "pitch_offset_deg": self.pitch_offset_deg,
        }


@dataclass(frozen=True)
class IMUSpec:
    """One MPU-6050, optionally behind a multiplexer channel."""

    sensor_id: str = "imu"
    mux_channel: Optional[int] = 7
    i2c_address: int = 0x68
    interrupt_gpio: Optional[int] = 17
    sample_rate_hz: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_id", _require_text(self.sensor_id, "imu sensor_id"))
        _require_channel(self.mux_channel, f"{self.sensor_id} mux_channel", optional=True)
        # 0x68 and 0x69 are the two AD0 straps the MPU-6050 supports.
        _require_int(self.i2c_address, f"{self.sensor_id} i2c_address", 0x68, 0x69)
        if self.interrupt_gpio is not None:
            _require_int(self.interrupt_gpio, f"{self.sensor_id} interrupt_gpio", 0, 53)
        _require_float(self.sample_rate_hz, f"{self.sensor_id} sample_rate_hz", 4.0, 1000.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "mux_channel": self.mux_channel,
            "i2c_address": self.i2c_address,
            "interrupt_gpio": self.interrupt_gpio,
            "sample_rate_hz": self.sample_rate_hz,
        }


@dataclass(frozen=True)
class HapticMotorSpec:
    """One DRV2605L driving a 3 V eccentric-rotating-mass motor."""

    motor_id: str
    sector: Sector
    mux_channel: int
    i2c_address: int = 0x5A
    motor_type: str = "erm"
    supply_voltage_v: float = 3.0
    rated_voltage: int = 0x7F
    od_clamp: int = 0x96

    def __post_init__(self) -> None:
        object.__setattr__(self, "motor_id", _require_text(self.motor_id, "haptic motor_id"))
        object.__setattr__(self, "sector", _require_sector(self.sector, f"{self.motor_id} sector"))
        _require_channel(self.mux_channel, f"{self.motor_id} mux_channel")
        _require_bus_address(self.i2c_address, f"{self.motor_id} i2c_address")
        if self.motor_type not in {"erm", "lra"}:
            raise HardwareConfigError(f"{self.motor_id} motor_type must be 'erm' or 'lra'")
        _require_float(self.supply_voltage_v, f"{self.motor_id} supply_voltage_v", 1.8, 5.5)
        _require_int(self.rated_voltage, f"{self.motor_id} rated_voltage", 0x00, 0xFF)
        _require_int(self.od_clamp, f"{self.motor_id} od_clamp", 0x00, 0xFF)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "motor_id": self.motor_id,
            "sector": self.sector.value,
            "mux_channel": self.mux_channel,
            "i2c_address": self.i2c_address,
            "motor_type": self.motor_type,
            "supply_voltage_v": self.supply_voltage_v,
            "rated_voltage": self.rated_voltage,
            "od_clamp": self.od_clamp,
        }


@dataclass(frozen=True)
class CameraSpec:
    """The CSI-2 camera and the model it runs by default."""

    sensor: str = "imx219"
    width: int = 640
    height: int = 480
    pixel_format: str = "RGB888"
    fps: float = 30.0
    model_path: str = "artifacts/models/cane-v1-fast/ov-fp32-416/best_openvino_model"
    num_threads: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor", _require_text(self.sensor, "camera sensor").lower())
        object.__setattr__(self, "pixel_format", _require_text(self.pixel_format, "camera pixel_format"))
        object.__setattr__(self, "model_path", _require_text(self.model_path, "camera model_path"))
        _require_int(self.width, "camera width", 1, 8192)
        _require_int(self.height, "camera height", 1, 8192)
        _require_float(self.fps, "camera fps", 1.0, 120.0)
        _require_int(self.num_threads, "camera num_threads", 1, 64)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sensor": self.sensor,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "fps": self.fps,
            "model_path": self.model_path,
            "num_threads": self.num_threads,
        }


@dataclass(frozen=True)
class FusionTimingSpec:
    """Loop, staleness, and warning thresholds shared with the fusion policy."""

    control_loop_hz: float = 30.0
    tof_poll_interval_ms: float = 33.0
    ultrasonic_guard_interval_ms: float = 35.0
    ultrasonic_timeout_ms: float = 25.0
    ambient_temp_c: float = 22.0
    sensor_stale_after_ms: float = 250.0
    vision_stale_after_ms: float = 500.0
    warning_distance_m: float = 1.0
    critical_distance_m: float = 0.5
    haptic_watchdog_timeout_ms: float = 500.0

    def __post_init__(self) -> None:
        _require_float(self.control_loop_hz, "control_loop_hz", 1.0, 200.0)
        _require_float(self.tof_poll_interval_ms, "tof_poll_interval_ms", 5.0, 500.0)
        _require_float(self.ultrasonic_guard_interval_ms, "ultrasonic_guard_interval_ms", 1.0, 500.0)
        _require_float(self.ultrasonic_timeout_ms, "ultrasonic_timeout_ms", 1.0, 200.0)
        _require_float(self.ambient_temp_c, "ambient_temp_c", -40.0, 85.0)
        _require_float(self.sensor_stale_after_ms, "sensor_stale_after_ms", 1.0, 10_000.0)
        _require_float(self.vision_stale_after_ms, "vision_stale_after_ms", 1.0, 10_000.0)
        _require_float(self.warning_distance_m, "warning_distance_m", 0.05, 10.0)
        _require_float(self.critical_distance_m, "critical_distance_m", 0.01, 10.0)
        if self.critical_distance_m > self.warning_distance_m:
            raise HardwareConfigError("critical_distance_m must not exceed warning_distance_m")
        _require_float(self.haptic_watchdog_timeout_ms, "haptic_watchdog_timeout_ms", 50.0, 5_000.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "control_loop_hz": self.control_loop_hz,
            "tof_poll_interval_ms": self.tof_poll_interval_ms,
            "ultrasonic_guard_interval_ms": self.ultrasonic_guard_interval_ms,
            "ultrasonic_timeout_ms": self.ultrasonic_timeout_ms,
            "ambient_temp_c": self.ambient_temp_c,
            "sensor_stale_after_ms": self.sensor_stale_after_ms,
            "vision_stale_after_ms": self.vision_stale_after_ms,
            "warning_distance_m": self.warning_distance_m,
            "critical_distance_m": self.critical_distance_m,
            "haptic_watchdog_timeout_ms": self.haptic_watchdog_timeout_ms,
        }


@dataclass(frozen=True)
class SmartCaneHardwareConfig:
    """The complete, validated Pi 5 wiring contract."""

    tof_sensors: Tuple[ToFSensorSpec, ...]
    ultrasonic_sensors: Tuple[UltrasonicSensorSpec, ...]
    imu: IMUSpec
    haptic_motors: Tuple[HapticMotorSpec, ...]
    camera: CameraSpec = field(default_factory=CameraSpec)
    timing: FusionTimingSpec = field(default_factory=FusionTimingSpec)
    profile_id: str = "smart-cane-pi5-v1"
    i2c_bus: int = 1
    mux_address: int = 0x70
    gpio_chip: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _require_text(self.profile_id, "profile_id"))
        _require_int(self.i2c_bus, "i2c_bus", 0, 8)
        _require_int(self.mux_address, "mux_address", 0x70, 0x77)
        _require_int(self.gpio_chip, "gpio_chip", 0, 15)
        object.__setattr__(self, "tof_sensors", tuple(self.tof_sensors))
        object.__setattr__(self, "ultrasonic_sensors", tuple(self.ultrasonic_sensors))
        object.__setattr__(self, "haptic_motors", tuple(self.haptic_motors))
        self._validate_unique_ids()
        self._validate_sector_coverage("tof_sensors", self.tof_sensors)
        self._validate_sector_coverage("haptic_motors", self.haptic_motors)
        self._validate_bus_topology()

    def _validate_unique_ids(self) -> None:
        for name, sensors in (
            ("tof_sensors", self.tof_sensors),
            ("ultrasonic_sensors", self.ultrasonic_sensors),
            ("haptic_motors", self.haptic_motors),
        ):
            identifiers = [getattr(sensor, _id_field(name)) for sensor in sensors]
            duplicates = {item for item in identifiers if identifiers.count(item) > 1}
            if duplicates:
                raise HardwareConfigError(f"{name} has duplicate ids: {sorted(duplicates)}")
        if not self.tof_sensors:
            raise HardwareConfigError("at least one time-of-flight sensor is required")
        if not self.ultrasonic_sensors:
            raise HardwareConfigError("at least one ultrasonic sensor is required")
        if not self.haptic_motors:
            raise HardwareConfigError("at least one haptic motor is required")

    @staticmethod
    def _validate_sector_coverage(name: str, sensors: Sequence[Any]) -> None:
        sectors = [sensor.sector for sensor in sensors]
        for sector in WARNING_SECTORS:
            if sector not in sectors:
                raise HardwareConfigError(f"{name} must cover the {sector.value} sector")
        duplicates = {sector for sector in sectors if sectors.count(sector) > 1}
        if duplicates:
            raise HardwareConfigError(
                f"{name} assigns multiple sensors to: {sorted(item.value for item in duplicates)}"
            )

    def _validate_bus_topology(self) -> None:
        # Every multiplexer channel isolates one device type, so two devices may
        # share 0x29/0x5A as long as they sit on different channels.
        claimed: Dict[Tuple[Optional[int], int], str] = {}
        entries: Sequence[Tuple[str, Optional[int], int]] = [
            (sensor.sensor_id, sensor.mux_channel, sensor.i2c_address) for sensor in self.tof_sensors
        ]
        entries = entries + [(self.imu.sensor_id, self.imu.mux_channel, self.imu.i2c_address)]
        entries = entries + [
            (motor.motor_id, motor.mux_channel, motor.i2c_address) for motor in self.haptic_motors
        ]
        for identifier, channel, address in entries:
            key = (channel, address)
            if key in claimed:
                where = "the main bus" if channel is None else f"channel {channel}"
                raise HardwareConfigError(
                    f"{identifier} shares address 0x{address:02X} on {where} with {claimed[key]}"
                )
            claimed[key] = identifier
            if address == self.mux_address:
                raise HardwareConfigError(
                    f"{identifier} cannot use the multiplexer's own address 0x{self.mux_address:02X}"
                )

    def tof_sensor(self, sensor_id: str) -> ToFSensorSpec:
        return self._lookup(self.tof_sensors, "sensor_id", sensor_id)

    def haptic_motor(self, sector: Sector) -> HapticMotorSpec:
        wanted = _require_sector(sector)
        for motor in self.haptic_motors:
            if motor.sector == wanted:
                return motor
        raise HardwareConfigError(f"no haptic motor for sector {wanted.value}")

    @staticmethod
    def _lookup(sensors: Sequence[Any], attribute: str, value: str) -> Any:
        for sensor in sensors:
            if getattr(sensor, attribute) == value:
                return sensor
        raise HardwareConfigError(f"unknown sensor_id {value!r}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "i2c_bus": self.i2c_bus,
            "mux_address": self.mux_address,
            "gpio_chip": self.gpio_chip,
            "tof_sensors": [sensor.as_dict() for sensor in self.tof_sensors],
            "ultrasonic_sensors": [sensor.as_dict() for sensor in self.ultrasonic_sensors],
            "imu": self.imu.as_dict(),
            "haptic_motors": [motor.as_dict() for motor in self.haptic_motors],
            "camera": self.camera.as_dict(),
            "timing": self.timing.as_dict(),
        }


def _id_field(group: str) -> str:
    return "motor_id" if group == "haptic_motors" else "sensor_id"


def _build(spec_type: Any, payload: Any, context: str) -> Any:
    if not isinstance(payload, Mapping):
        raise HardwareConfigError(f"{context} must be a JSON object")
    allowed = {item.name for item in fields(spec_type)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise HardwareConfigError(f"{context} has unknown field(s): {unknown}")
    try:
        return spec_type(**dict(payload))
    except TypeError as exc:
        raise HardwareConfigError(f"{context} is incomplete: {exc}") from exc


def _build_sequence(spec_type: Any, payload: Any, context: str) -> Tuple[Any, ...]:
    if not isinstance(payload, list) or not payload:
        raise HardwareConfigError(f"{context} must be a non-empty JSON array")
    return tuple(
        _build(spec_type, item, f"{context}[{index}]") for index, item in enumerate(payload)
    )


def load_hardware_config(path: Path) -> SmartCaneHardwareConfig:
    """Load and validate a smart-cane hardware profile from JSON."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HardwareConfigError(f"hardware config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HardwareConfigError(f"hardware config is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise HardwareConfigError("hardware config must be a JSON object")

    top_level = {item.name for item in fields(SmartCaneHardwareConfig)}
    unknown = sorted(set(payload) - top_level)
    if unknown:
        raise HardwareConfigError(f"hardware config has unknown field(s): {unknown}")
    missing = sorted(
        name
        for name in ("tof_sensors", "ultrasonic_sensors", "imu", "haptic_motors")
        if name not in payload
    )
    if missing:
        raise HardwareConfigError(f"hardware config is missing required field(s): {missing}")

    return SmartCaneHardwareConfig(
        tof_sensors=_build_sequence(ToFSensorSpec, payload["tof_sensors"], "tof_sensors"),
        ultrasonic_sensors=_build_sequence(
            UltrasonicSensorSpec, payload["ultrasonic_sensors"], "ultrasonic_sensors"
        ),
        imu=_build(IMUSpec, payload["imu"], "imu"),
        haptic_motors=_build_sequence(HapticMotorSpec, payload["haptic_motors"], "haptic_motors"),
        camera=_build(CameraSpec, payload.get("camera", {}), "camera"),
        timing=_build(FusionTimingSpec, payload.get("timing", {}), "timing"),
        profile_id=payload.get("profile_id", "smart-cane-pi5-v1"),
        i2c_bus=payload.get("i2c_bus", 1),
        mux_address=payload.get("mux_address", 0x70),
        gpio_chip=payload.get("gpio_chip", 4),
    )


def default_pi5_config() -> SmartCaneHardwareConfig:
    """Return the report's wiring without reading JSON from disk."""

    return SmartCaneHardwareConfig(
        profile_id="smart-cane-pi5-v1",
        i2c_bus=1,
        mux_address=0x70,
        gpio_chip=4,
        tof_sensors=(
            ToFSensorSpec("tof-left", Sector.LEFT, 0, 0x29, 33.0),
            ToFSensorSpec("tof-center", Sector.CENTER, 1, 0x29, 33.0),
            ToFSensorSpec("tof-right", Sector.RIGHT, 2, 0x29, 33.0),
        ),
        ultrasonic_sensors=(
            UltrasonicSensorSpec("us-center-1", Sector.CENTER, 23, 24, 0.04, 3.50, 0.0),
            UltrasonicSensorSpec("us-center-2", Sector.CENTER, 27, 22, 0.04, 3.50, 8.0),
        ),
        imu=IMUSpec("imu", 7, 0x68, 17, 100.0),
        haptic_motors=(
            HapticMotorSpec("haptic-left", Sector.LEFT, 4, 0x5A, "erm", 3.0, 0x7F, 0x96),
            HapticMotorSpec("haptic-center", Sector.CENTER, 5, 0x5A, "erm", 3.0, 0x7F, 0x96),
            HapticMotorSpec("haptic-right", Sector.RIGHT, 6, 0x5A, "erm", 3.0, 0x7F, 0x96),
        ),
        camera=CameraSpec(),
        timing=FusionTimingSpec(),
    )
