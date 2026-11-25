"""Raspberry Pi 5 hardware drivers for the smart cane.

Everything here is importable on a laptop: the I2C, GPIO, and camera libraries
are imported only when a real backend is constructed, and the tests exercise
the drivers against register-level fakes.
"""

from .config import (
    CameraSpec,
    FusionTimingSpec,
    HapticMotorSpec,
    HardwareConfigError,
    IMUSpec,
    SmartCaneHardwareConfig,
    ToFSensorSpec,
    UltrasonicSensorSpec,
    default_pi5_config,
    load_hardware_config,
)
from .drv2605l import DRV2605LController, DRV2605LError
from .hcsr04 import (
    EchoStuckHighError,
    GPIOBackend,
    GPIOException,
    GpioZeroBackend,
    HCSR04PairDriver,
    LgpioBackend,
    UltrasonicMeasurement,
    create_gpio_backend,
    speed_of_sound_mps,
)
from .i2c import (
    FakeDevice,
    FakeSMBus,
    SMBus2Adapter,
    SMBusProtocol,
    TCA9548AMux,
)
from .mpu6050 import IMUCalibration, MPU6050Driver, MPU6050Error, RawIMUSample
from .vl53l0x import (
    ToFMeasurement,
    VL53L0XArray,
    VL53L0XDriver,
    VL53L0XError,
    VL53L0XTimeout,
    decode_range_status,
)

__all__ = [
    "CameraSpec",
    "DRV2605LController",
    "DRV2605LError",
    "EchoStuckHighError",
    "FakeDevice",
    "FakeSMBus",
    "FusionTimingSpec",
    "GPIOBackend",
    "GPIOException",
    "GpioZeroBackend",
    "HCSR04PairDriver",
    "HapticMotorSpec",
    "HardwareConfigError",
    "IMUCalibration",
    "IMUSpec",
    "LgpioBackend",
    "MPU6050Driver",
    "MPU6050Error",
    "RawIMUSample",
    "SMBus2Adapter",
    "SMBusProtocol",
    "SmartCaneHardwareConfig",
    "TCA9548AMux",
    "ToFMeasurement",
    "ToFSensorSpec",
    "UltrasonicMeasurement",
    "UltrasonicSensorSpec",
    "VL53L0XArray",
    "VL53L0XDriver",
    "VL53L0XError",
    "VL53L0XTimeout",
    "create_gpio_backend",
    "decode_range_status",
    "default_pi5_config",
    "load_hardware_config",
    "speed_of_sound_mps",
]
