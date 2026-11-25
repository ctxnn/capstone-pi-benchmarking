"""VL53L0X time-of-flight ranging behind the TCA9548A multiplexer.

The three sensors share I2C address 0x29, so each lives on its own mux channel.
They run in the sensor's own continuous mode: every sensor measures on its own
clock, and the poller only collects a result when one is ready. That keeps the
loop non-blocking, so one slow sensor never delays the other two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .bus import I2CBus
from .config import I2CConfig, TofConfig, TofSensorSpec

LOGGER = logging.getLogger("yolo_pi.cane.tof")


@dataclass(frozen=True)
class RangeSample:
    """One interpreted range measurement, ready to become a ``SensorReading``."""

    distance_m: Optional[float]
    valid: bool
    out_of_range: bool = False
    raw: Optional[float] = None


def convert_tof_mm(raw_mm: int, config: TofConfig) -> RangeSample:
    """Interpret a raw VL53L0X reading.

    * A large code (8190/8191/65535) means "no target in range". For a cane that
      is the *good* case, so it becomes a valid reading at ``clear_distance_m``.
      Treating it as invalid would make the fusion engine report a sensor fault
      whenever the path is empty.
    * Zero is what the chip returns on a failed measurement, so it is invalid.
    * Everything else is the distance in metres, capped at ``clear_distance_m``.
    """

    if raw_mm >= config.out_of_range_mm:
        return RangeSample(config.clear_distance_m, True, True, float(raw_mm))
    if raw_mm <= 0:
        return RangeSample(None, False, False, float(raw_mm))
    return RangeSample(min(raw_mm / 1000.0, config.clear_distance_m), True, False, float(raw_mm))


class Vl53l0xDevice:
    """Non-blocking continuous-mode wrapper around one VL53L0X."""

    def __init__(self, sensor_factory: Callable[[], Any], config: TofConfig) -> None:
        self._factory = sensor_factory
        self._config = config
        self._sensor: Any = None

    def start(self) -> None:
        sensor = self._factory()
        try:
            sensor.io_timeout_s = self._config.io_timeout_s
        except AttributeError:
            pass
        try:
            sensor.measurement_timing_budget = self._config.timing_budget_us
        except (ValueError, AttributeError) as exc:
            LOGGER.warning("could not set the ToF timing budget: %s", exc)
        sensor.start_continuous()
        self._sensor = sensor

    def poll(self) -> Optional[RangeSample]:
        """Return a new sample, or ``None`` if the sensor has not finished one yet."""

        if self._sensor is None:
            raise RuntimeError("ToF sensor polled before start()")
        if not self._sensor.data_ready:
            return None
        return convert_tof_mm(int(self._sensor.range), self._config)

    def restart(self) -> None:
        self.close()
        self.start()

    def close(self) -> None:
        sensor, self._sensor = self._sensor, None
        if sensor is not None:
            try:
                sensor.stop_continuous()
            except Exception:
                pass


def create_vl53l0x(
    bus: I2CBus, spec: TofSensorSpec, i2c: I2CConfig, config: TofConfig
) -> Vl53l0xDevice:
    """Build a device whose factory imports the Adafruit driver lazily."""

    def factory() -> Any:  # pragma: no cover - needs a Pi
        try:
            import adafruit_vl53l0x
        except ImportError as exc:
            from .bus import HardwareUnavailable

            raise HardwareUnavailable(
                "adafruit-circuitpython-vl53l0x is not installed"
            ) from exc
        return adafruit_vl53l0x.VL53L0X(
            bus.channel(spec.mux_channel),
            address=i2c.tof_address,
            io_timeout_s=config.io_timeout_s,
        )

    return Vl53l0xDevice(factory, config)
