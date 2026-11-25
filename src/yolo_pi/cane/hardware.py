"""Builds the real devices from the configuration, tolerating partial failure.

A cane with one dead sensor is still a useful cane, so each device is brought up
on its own and failures are collected instead of aborting. The only fatal
conditions are no I2C bus and no working motor: without those the cane cannot
tell the user anything, and the systemd service restarts it to try again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .bus import BlinkaBus, HardwareUnavailable, I2CBus
from .config import CaneConfig
from .haptics import Drv2605Erm, MotorDriver
from .imu import Mpu6050
from .tof import Vl53l0xDevice, create_vl53l0x
from .ultrasonic import GpioBackend, Hcsr04, LgpioBackend

LOGGER = logging.getLogger("yolo_pi.cane.hardware")


@dataclass
class HardwareSet:
    bus: Optional[I2CBus] = None
    gpio: Optional[GpioBackend] = None
    tof: Dict[str, Vl53l0xDevice] = field(default_factory=dict)
    ultrasonic: Dict[str, Hcsr04] = field(default_factory=dict)
    haptic: Optional[MotorDriver] = None
    imu: Optional[Mpu6050] = None
    errors: List[str] = field(default_factory=list)

    def close(self) -> None:
        for device in list(self.tof.values()) + list(self.ultrasonic.values()):
            try:
                device.close()
            except Exception as exc:  # one stuck device must not stop the rest being released
                LOGGER.warning("closing a device failed: %s", exc)
        if self.gpio is not None:
            try:
                self.gpio.close()
            except Exception:
                pass
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass


def build_real_hardware(
    config: CaneConfig,
    *,
    with_motors: bool = True,
    with_ranging: bool = True,
) -> HardwareSet:
    """Open the I2C bus and GPIO and bring up every configured device."""

    hardware = HardwareSet()
    hardware.bus = BlinkaBus(config.i2c.mux_address)
    bus = hardware.bus

    if with_motors:
        where = config.haptics.driver_mux_channel
        try:
            device = bus.register_device(config.i2c.haptic_address, where)
            driver = Drv2605Erm(device, config.haptics.motor_voltage_v)
            driver.configure()
            hardware.haptic = driver
        except Exception as exc:
            message = (
                f"DRV2605L ({'main bus' if where is None else f'mux channel {where}'}): {exc}"
            )
            hardware.errors.append(message)
            LOGGER.error("haptic driver unavailable: %s", message)
            hardware.close()
            raise HardwareUnavailable(
                "the DRV2605L haptic driver did not answer; the cane cannot alert the user. "
                "Run 'yolo-pi-cane scan' to check where it sits. " + message
            )

    if with_ranging:
        for spec in config.tof.sensors:
            try:
                device = create_vl53l0x(bus, spec, config.i2c, config.tof)
                device.start()
                hardware.tof[spec.sensor_id] = device
            except Exception as exc:
                message = f"{spec.sensor_id} (mux channel {spec.mux_channel}): {exc}"
                hardware.errors.append(message)
                LOGGER.error("ToF unavailable: %s", message)

        if config.ultrasonic.sensors:
            try:
                hardware.gpio = LgpioBackend(config.ultrasonic.gpio_chip)
            except Exception as exc:
                hardware.errors.append(f"ultrasonic GPIO: {exc}")
                LOGGER.error("ultrasonic GPIO unavailable: %s", exc)
            else:
                for spec in config.ultrasonic.sensors:
                    try:
                        sensor = Hcsr04(spec, hardware.gpio, config.ultrasonic)
                        sensor.start()
                        hardware.ultrasonic[spec.sensor_id] = sensor
                    except Exception as exc:
                        message = f"{spec.sensor_id} (GPIO{spec.trig_gpio}/GPIO{spec.echo_gpio}): {exc}"
                        hardware.errors.append(message)
                        LOGGER.error("ultrasonic unavailable: %s", message)

    if config.imu.enabled:
        try:
            device = bus.register_device(config.i2c.imu_address, config.imu.mux_channel)
            imu = Mpu6050(device)
            imu.configure()
            hardware.imu = imu
        except Exception as exc:
            hardware.errors.append(f"imu: {exc}")
            LOGGER.error("IMU unavailable: %s", exc)

    return hardware
