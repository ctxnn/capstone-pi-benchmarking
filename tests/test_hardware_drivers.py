"""Register-level tests for the Pi 5 hardware drivers.

Everything runs on Linux x86_64: the I2C multiplexer and every sensor is a
register map in :class:`FakeSMBus`, GPIO is a protocol-level fake, and time is
injected wherever the driver has to wait. The tests therefore assert the exact
bus traffic (channel-select bytes, register writes, redundant-write skipping)
and the fault/clear-space distinctions that make the cane safe.
"""

import errno
import sys
import threading
import time
import types
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

from yolo_pi.hardware import (
    DRV2605LController,
    DRV2605LError,
    EchoStuckHighError,
    FakeSMBus,
    GPIOException,
    HCSR04PairDriver,
    LgpioBackend,
    MPU6050Driver,
    MPU6050Error,
    SMBus2Adapter,
    TCA9548AMux,
    VL53L0XArray,
    VL53L0XDriver,
    VL53L0XError,
    create_gpio_backend,
    decode_range_status,
    default_pi5_config,
    load_hardware_config,
    speed_of_sound_mps,
)
from yolo_pi.hardware.config import (
    CameraSpec,
    FusionTimingSpec,
    HapticMotorSpec,
    HardwareConfigError,
    IMUSpec,
    SmartCaneHardwareConfig,
    ToFSensorSpec,
    UltrasonicSensorSpec,
)
from yolo_pi.types import Sector

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "smart-cane-pi5.json"
MS = 1_000_000
MUX_ADDRESS = 0x70
TOF_ADDRESS = 0x29
HAPTIC_ADDRESS = 0x5A
IMU_ADDRESS = 0x68
TOF_CHANNELS = {Sector.LEFT: 0, Sector.CENTER: 1, Sector.RIGHT: 2}
HAPTIC_CHANNELS = {Sector.LEFT: 4, Sector.CENTER: 5, Sector.RIGHT: 6}


# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------


def write_range_result(registers, status, mm):
    """Lay out the 12-byte VL53L0X result block starting at 0x14."""

    registers[0x14] = (status & 0x0F) << 3
    registers[0x15] = 0x00
    registers[0x14 + 10] = (mm >> 8) & 0xFF
    registers[0x14 + 11] = mm & 0xFF


def tof_registers(status=11, mm=850):
    registers = {
        0xC0: 0xEE,  # model id
        0x13: 0x07,  # result interrupt ready
        0x91: 0x5A,  # stop variable
        0x89: 0x00,  # 1V8 I/O by default
        0x60: 0x00,  # MSRC config control
        0x84: 0x10,  # GPIO active-high bit set, must be cleared
        0x01: 0x00,  # sequence config
    }
    write_range_result(registers, status, mm)
    return registers


def haptic_registers():
    """A DRV2605L that starts in standby with LRA/open-loop bits set."""

    return {0x00: 0x00, 0x01: 0x40, 0x1A: 0x80, 0x1D: 0x20}


def imu_registers(who_am_i=0x68):
    return {0x75: who_am_i, 0x6B: 0x40, 0x1A: 0x00}


def build_pi5_bus(config=None):
    """Fake bus wired exactly like ``configs/smart-cane-pi5.json``."""

    config = config or load_hardware_config(CONFIG_PATH)
    bus = FakeSMBus(mux_address=config.mux_address)
    for spec in config.tof_sensors:
        bus.add_device(spec.mux_channel, spec.i2c_address, tof_registers())
    for motor in config.haptic_motors:
        bus.add_device(motor.mux_channel, motor.i2c_address, haptic_registers())
    bus.add_device(config.imu.mux_channel, config.imu.i2c_address, imu_registers())
    return bus


def build_mux(bus, **kwargs):
    kwargs.setdefault("retry_delay_s", 0.0)
    return TCA9548AMux(bus, address=MUX_ADDRESS, **kwargs)


def imu_payload(accel, gyro, temperature_raw):
    values = list(accel) + [temperature_raw] + list(gyro)
    payload = []
    for value in values:
        value &= 0xFFFF
        payload.extend([(value >> 8) & 0xFF, value & 0xFF])
    return payload


class FakeGPIO:
    """GPIOBackend double with a scripted echo reply per pin."""

    def __init__(self, echoes=None, error=None):
        self.echoes = dict(echoes or {})
        self.error = error
        self.outputs = []
        self.inputs = []
        self.trigger_writes = {}
        self.pulse_calls = []
        self.closed = False

    def configure_output(self, pin):
        self.outputs.append(pin)
        self.trigger_writes.setdefault(pin, [])

    def configure_input(self, pin):
        self.inputs.append(pin)

    def write(self, pin, value):
        self.trigger_writes.setdefault(pin, []).append(bool(value))

    def read(self, pin):
        return False

    def wait_for_edge(self, pin, level, timeout_s):
        return False

    def pulse_echo_s(self, pin, timeout_s):
        self.pulse_calls.append((pin, timeout_s))
        if self.error is not None:
            raise self.error
        value = self.echoes.get(pin)
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self):
        self.closed = True


class FakeClock:
    """Deterministic nanosecond clock whose sleeps advance the clock."""

    def __init__(self, start_ns=0):
        self.now_ns = start_ns
        self.sleeps = []

    def __call__(self):
        return self.now_ns

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now_ns += int(round(seconds * 1e9))

    def advance_ms(self, milliseconds):
        self.now_ns += int(round(milliseconds * MS))


def build_ultrasonic(echoes, config=None, **kwargs):
    config = config or load_hardware_config(CONFIG_PATH)
    gpio = FakeGPIO(echoes)
    clock = FakeClock()
    driver = HCSR04PairDriver(
        gpio,
        config.ultrasonic_sensors,
        clock=clock,
        sleeper=clock.sleep,
        **kwargs,
    )
    return driver, gpio, clock


class CountingBus:
    """SMBusProtocol wrapper that records overlapping transactions."""

    def __init__(self, inner, delay_s=0.0002):
        self._inner = inner
        self._delay_s = delay_s
        self._guard = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0

    def _run(self, operation, *args):
        with self._guard:
            self.in_flight += 1
            self.calls += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self._delay_s)
            return operation(*args)
        finally:
            with self._guard:
                self.in_flight -= 1

    def read_byte_data(self, address, register):
        return self._run(self._inner.read_byte_data, address, register)

    def write_byte_data(self, address, register, value):
        return self._run(self._inner.write_byte_data, address, register, value)

    def read_i2c_block_data(self, address, register, length):
        return self._run(self._inner.read_i2c_block_data, address, register, length)

    def write_byte(self, address, value):
        return self._run(self._inner.write_byte, address, value)

    def close(self):
        self._inner.close()


def make_fake_lgpio(unavailable_chips=(4,)):
    module = types.ModuleType("lgpio")
    module.unavailable = set(unavailable_chips)
    module.opened = []
    module.claimed = []
    module.writes = []
    module.levels = {}
    module.closed_handles = []

    def gpiochip_open(chip):
        if chip in module.unavailable:
            raise RuntimeError(f"can't open gpiochip {chip}")
        module.opened.append(chip)
        return 100 + chip

    def gpio_claim_output(handle, pin, level=0):
        module.claimed.append(("out", pin))
        module.levels[pin] = level

    def gpio_claim_input(handle, pin, flags=0):
        module.claimed.append(("in", pin))
        module.levels[pin] = 0

    def gpio_write(handle, pin, level):
        module.writes.append((pin, level))
        module.levels[pin] = level

    def gpio_read(handle, pin):
        return module.levels.get(pin, 0)

    def gpio_free(handle, pin):
        module.claimed.append(("free", pin))

    def gpiochip_close(handle):
        module.closed_handles.append(handle)

    module.gpiochip_open = gpiochip_open
    module.gpio_claim_output = gpio_claim_output
    module.gpio_claim_input = gpio_claim_input
    module.gpio_write = gpio_write
    module.gpio_read = gpio_read
    module.gpio_free = gpio_free
    module.gpiochip_close = gpiochip_close
    return module


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class HardwareConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_hardware_config(CONFIG_PATH)

    def test_matches_pdf_pinout(self):
        config = self.config
        self.assertEqual(config.i2c_bus, 1)
        self.assertEqual(config.mux_address, 0x70)
        self.assertEqual(config.gpio_chip, 4)

        tofs = {spec.sector: spec for spec in config.tof_sensors}
        self.assertEqual(set(tofs), {Sector.LEFT, Sector.CENTER, Sector.RIGHT})
        for sector, channel in TOF_CHANNELS.items():
            self.assertEqual(tofs[sector].mux_channel, channel)
            self.assertEqual(tofs[sector].i2c_address, 0x29)
            self.assertEqual(tofs[sector].sensor_id, f"tof-{sector.value}")
            self.assertEqual(tofs[sector].timing_budget_ms, 33.0)

        ultrasonics = {spec.sensor_id: spec for spec in config.ultrasonic_sensors}
        self.assertEqual(set(ultrasonics), {"us-center-1", "us-center-2"})
        self.assertEqual(
            (ultrasonics["us-center-1"].trigger_gpio, ultrasonics["us-center-1"].echo_gpio), (23, 24)
        )
        self.assertEqual(
            (ultrasonics["us-center-2"].trigger_gpio, ultrasonics["us-center-2"].echo_gpio), (27, 22)
        )
        for spec in ultrasonics.values():
            self.assertEqual(spec.sector, Sector.CENTER)
            self.assertEqual((spec.min_range_m, spec.max_range_m), (0.04, 3.5))

        self.assertEqual(config.imu.mux_channel, 7)
        self.assertEqual(config.imu.i2c_address, 0x68)
        self.assertEqual(config.imu.interrupt_gpio, 17)
        self.assertEqual(config.imu.sample_rate_hz, 100.0)

        motors = {motor.sector: motor for motor in config.haptic_motors}
        self.assertEqual(set(motors), {Sector.LEFT, Sector.CENTER, Sector.RIGHT})
        for sector, channel in HAPTIC_CHANNELS.items():
            motor = motors[sector]
            self.assertEqual(motor.mux_channel, channel)
            self.assertEqual(motor.i2c_address, 0x5A)
            self.assertEqual(motor.motor_type, "erm")
            self.assertEqual(motor.supply_voltage_v, 3.0)
            self.assertEqual(motor.rated_voltage, 0x7F)
            self.assertEqual(motor.od_clamp, 0x96)

    def test_camera_and_fusion_defaults(self):
        camera = self.config.camera
        self.assertEqual(camera.sensor, "imx219")
        self.assertEqual((camera.width, camera.height), (640, 480))
        self.assertEqual(camera.pixel_format, "RGB888")
        self.assertEqual(camera.fps, 30.0)
        self.assertEqual(
            camera.model_path, "artifacts/models/cane-v1-fast/ov-fp32-416/best_openvino_model"
        )
        self.assertEqual(camera.num_threads, 2)

        timing = self.config.timing
        self.assertEqual(timing.ultrasonic_guard_interval_ms, 35.0)
        self.assertEqual(timing.ultrasonic_timeout_ms, 25.0)
        self.assertEqual(timing.ambient_temp_c, 22.0)
        self.assertEqual(timing.warning_distance_m, 1.0)
        self.assertEqual(timing.critical_distance_m, 0.5)
        self.assertEqual(timing.haptic_watchdog_timeout_ms, 500.0)
        self.assertEqual(timing.sensor_stale_after_ms, 250.0)

    def test_json_file_matches_default_pi5_config(self):
        self.assertEqual(self.config, default_pi5_config())
        self.assertEqual(self.config.as_dict(), default_pi5_config().as_dict())

    def test_lookup_helpers(self):
        self.assertEqual(self.config.tof_sensor("tof-center").mux_channel, 1)
        self.assertEqual(self.config.haptic_motor(Sector.RIGHT).motor_id, "haptic-right")
        with self.assertRaises(HardwareConfigError):
            self.config.tof_sensor("tof-missing")

    def test_rejects_incomplete_sector_coverage(self):
        config = self.config.as_dict()
        config["tof_sensors"][2]["sector"] = "center"
        with self.assertRaises(HardwareConfigError):
            SmartCaneHardwareConfig(
                tof_sensors=tuple(ToFSensorSpec(**item) for item in config["tof_sensors"]),
                ultrasonic_sensors=self.config.ultrasonic_sensors,
                imu=self.config.imu,
                haptic_motors=self.config.haptic_motors,
            )

    def test_rejects_invalid_multiplexer_channel(self):
        with self.assertRaises(HardwareConfigError):
            ToFSensorSpec("tof-left", Sector.LEFT, 8, 0x29, 33.0)
        with self.assertRaises(HardwareConfigError):
            HapticMotorSpec("haptic-left", Sector.LEFT, -1)

    def test_rejects_duplicate_bus_address_on_one_channel(self):
        with self.assertRaises(HardwareConfigError):
            SmartCaneHardwareConfig(
                tof_sensors=self.config.tof_sensors,
                ultrasonic_sensors=self.config.ultrasonic_sensors,
                imu=IMUSpec("imu", 0, 0x29, 17, 100.0),
                haptic_motors=self.config.haptic_motors,
            )

    def test_rejects_duplicate_motor_channel(self):
        motors = (
            HapticMotorSpec("haptic-left", Sector.LEFT, 4),
            HapticMotorSpec("haptic-center", Sector.CENTER, 4),
            HapticMotorSpec("haptic-right", Sector.RIGHT, 6),
        )
        with self.assertRaises(HardwareConfigError):
            SmartCaneHardwareConfig(
                tof_sensors=self.config.tof_sensors,
                ultrasonic_sensors=self.config.ultrasonic_sensors,
                imu=self.config.imu,
                haptic_motors=motors,
            )

    def test_rejects_invalid_json_and_unknown_fields(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(HardwareConfigError):
                load_hardware_config(path)

            payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            payload["surprise"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(HardwareConfigError):
                load_hardware_config(path)

            payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            payload["tof_sensors"][0]["surprise"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(HardwareConfigError):
                load_hardware_config(path)

            payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            del payload["imu"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(HardwareConfigError):
                load_hardware_config(path)

    def test_rejects_unsafe_timing_and_camera_values(self):
        with self.assertRaises(HardwareConfigError):
            FusionTimingSpec(critical_distance_m=2.0, warning_distance_m=1.0)
        with self.assertRaises(HardwareConfigError):
            FusionTimingSpec(ultrasonic_guard_interval_ms=0.0)
        with self.assertRaises(HardwareConfigError):
            CameraSpec(width=0)
        with self.assertRaises(HardwareConfigError):
            UltrasonicSensorSpec("us", Sector.CENTER, 23, 23)

    def test_mux_address_must_be_a_tca9548a_address(self):
        with self.assertRaises(HardwareConfigError):
            SmartCaneHardwareConfig(
                tof_sensors=self.config.tof_sensors,
                ultrasonic_sensors=self.config.ultrasonic_sensors,
                imu=self.config.imu,
                haptic_motors=self.config.haptic_motors,
                mux_address=0x50,
            )


# ---------------------------------------------------------------------------
# I2C transport and multiplexer
# ---------------------------------------------------------------------------


class SMBus2AdapterTests(unittest.TestCase):
    def test_reports_install_hint_without_smbus2(self):
        with unittest.mock.patch.dict(sys.modules, {"smbus2": None}):
            with self.assertRaises(RuntimeError) as caught:
                SMBus2Adapter(bus_num=1)
        self.assertIn("smbus2", str(caught.exception))

    def test_wraps_a_bus_object(self):
        bus = FakeSMBus()
        bus.add_device(None, IMU_ADDRESS, {0x75: 0x68, 0x3B: 0x00})
        adapter = SMBus2Adapter(bus_num=1, bus=bus)
        self.assertEqual(adapter.read_byte_data(IMU_ADDRESS, 0x75), 0x68)
        adapter.write_byte_data(IMU_ADDRESS, 0x6B, 0x01)
        self.assertEqual(bus.device(None, IMU_ADDRESS).registers[0x6B], 0x01)
        self.assertEqual(adapter.read_i2c_block_data(IMU_ADDRESS, 0x3B, 3), [0x00, 0x00, 0x00])
        adapter.write_byte(IMU_ADDRESS, 0x5A)
        self.assertEqual(bus.device(None, IMU_ADDRESS).command_log, [0x5A])
        adapter.close()
        self.assertTrue(bus.closed)


class TCA9548AMuxTests(unittest.TestCase):
    def setUp(self):
        self.bus = FakeSMBus()
        for channel in (0, 1, 2):
            self.bus.add_device(channel, TOF_ADDRESS, {0xC0: 0xEE, 0x10: channel})
        self.bus.add_device(4, HAPTIC_ADDRESS, haptic_registers())
        self.bus.add_device(None, IMU_ADDRESS, {0x75: 0x68})
        self.mux = build_mux(self.bus)

    def test_selects_one_hot_channel_and_caches_it(self):
        self.assertEqual(self.mux.read_byte(1, TOF_ADDRESS, 0x10), 1)
        self.assertEqual(self.bus.mux_writes, [0x02])
        self.assertEqual(self.mux.read_byte(1, TOF_ADDRESS, 0x10), 1)
        self.assertEqual(self.bus.mux_writes, [0x02])
        self.assertEqual(self.mux.active_channel, 1)

        self.assertEqual(self.mux.read_byte(0, TOF_ADDRESS, 0x10), 0)
        self.assertEqual(self.bus.mux_writes, [0x02, 0x01])
        self.assertEqual(self.mux.active_channel, 0)

    def test_main_bus_isolation_writes_zero(self):
        self.assertEqual(self.mux.read_byte(None, IMU_ADDRESS, 0x75), 0x68)
        self.assertEqual(self.bus.mux_writes, [0x00])
        self.assertIsNone(self.mux.active_channel)

        # A device behind a channel is unreachable with every channel disabled.
        with self.assertRaises(OSError):
            self.mux.read_byte(None, TOF_ADDRESS, 0xC0)
        # And a main-bus device is unreachable while a channel is selected.
        self.mux.read_byte(2, TOF_ADDRESS, 0xC0)
        with self.assertRaises(OSError):
            self.mux.read_byte(2, IMU_ADDRESS, 0x75)

    def test_channel_context_yields_the_bus_under_the_lock(self):
        with self.mux.channel_context(0) as bus:
            self.assertIs(bus, self.bus)
            self.assertEqual(bus.read_byte_data(TOF_ADDRESS, 0x10), 0)
        self.assertEqual(self.bus.mux_writes, [0x01])

    def test_nested_same_channel_context_does_not_rewrite(self):
        with self.mux.channel_context(2) as outer:
            with self.mux.channel_context(2) as inner:
                self.assertIs(outer, inner)
        self.assertEqual(self.bus.mux_writes, [0x04])

    def test_nested_context_restores_the_outer_channel(self):
        with self.mux.channel_context(0):
            with self.mux.channel_context(1):
                pass
            self.assertEqual(self.bus.read_byte_data(TOF_ADDRESS, 0x10), 0)
        self.assertEqual(self.bus.mux_writes, [0x01, 0x02, 0x01])

    def test_retries_transient_multiplexer_oserror(self):
        self.bus.fail_mux_writes = 2
        mux = build_mux(self.bus, retries=2)
        self.assertEqual(mux.read_byte(1, TOF_ADDRESS, 0x10), 1)
        self.assertEqual(self.bus.mux_writes, [0x02])

    def test_gives_up_when_retries_are_exhausted(self):
        self.bus.fail_mux_writes = 5
        mux = build_mux(self.bus, retries=1)
        with self.assertRaises(OSError):
            mux.read_byte(1, TOF_ADDRESS, 0x10)
        self.assertIsNone(mux.selected_mask)

    def test_concurrent_access_is_serialized(self):
        counting = CountingBus(self.bus)
        mux = build_mux(counting, retries=1)
        results = []
        failures = []

        def worker(channel):
            try:
                for _ in range(8):
                    results.append(mux.read_byte(channel, TOF_ADDRESS, 0x10))
            except BaseException as exc:  # pragma: no cover - failure reporting
                failures.append(exc)

        threads = [threading.Thread(target=worker, args=(channel,)) for channel in (0, 1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        self.assertEqual(failures, [])
        self.assertEqual(len(results), 24)
        self.assertEqual(counting.max_in_flight, 1)
        for channel in (0, 1, 2):
            self.assertEqual(results.count(channel), 8)

    def test_scan_channel_reports_devices_and_skips_the_mux(self):
        self.assertEqual(self.mux.scan_channel(0), [TOF_ADDRESS])
        self.assertEqual(self.mux.scan_channel(4), [HAPTIC_ADDRESS])
        self.assertEqual(self.mux.scan_channel(3), [])

    def test_disable_all_and_close(self):
        self.mux.read_byte(2, TOF_ADDRESS, 0x10)
        self.mux.disable_all()
        self.assertEqual(self.bus.mux_writes[-1], 0x00)
        self.assertIsNone(self.mux.active_channel)

        self.mux.close()
        self.assertTrue(self.bus.closed)
        with self.assertRaises(RuntimeError):
            self.mux.read_byte(0, TOF_ADDRESS, 0xC0)

    def test_rejects_invalid_channels_and_control_bytes(self):
        with self.assertRaises(ValueError):
            self.mux.channel_context(8)
        with self.assertRaises(ValueError):
            self.mux.read_byte(-1, TOF_ADDRESS, 0xC0)
        with self.assertRaises(OSError):
            self.bus.write_byte(MUX_ADDRESS, 0x03)
        with self.assertRaises(ValueError):
            self.mux.write_byte(0, TOF_ADDRESS, 0x01, 0x1FF)

    def test_fake_bus_rejects_writes_to_missing_devices(self):
        with self.assertRaises(OSError) as caught:
            self.mux.read_byte(3, TOF_ADDRESS, 0xC0)
        self.assertEqual(caught.exception.errno, errno.EREMOTEIO)


# ---------------------------------------------------------------------------
# VL53L0X
# ---------------------------------------------------------------------------


class VL53L0XDriverTests(unittest.TestCase):
    def setUp(self):
        self.config = load_hardware_config(CONFIG_PATH)
        self.bus = build_pi5_bus(self.config)
        self.mux = build_mux(self.bus)
        self.spec = self.config.tof_sensor("tof-center")
        self.driver = VL53L0XDriver(self.mux, self.spec)

    def test_initialize_verifies_identity_and_applies_configuration(self):
        self.driver.initialize()
        self.assertTrue(self.driver.initialized)
        self.assertEqual(self.driver.model_id, 0xEE)
        registers = self.bus.device(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS).registers
        self.assertEqual(registers[0x88], 0x00)
        self.assertEqual(registers[0x60], 0x12)
        self.assertEqual(registers[0x01], 0xE8)
        self.assertEqual(registers[0x0A], 0x04)
        self.assertEqual(registers[0x89] & 0x01, 0x01)
        self.assertEqual(registers[0x84] & 0x10, 0x00)
        self.assertEqual((registers[0x44], registers[0x45]), (0x00, 0x20))
        self.assertEqual(registers[0x50], 0x06)
        self.assertEqual(registers[0x70], 0x04)
        self.assertEqual(registers[0x0B], 0x01)

    def test_initialize_reads_the_stop_variable_from_0x91(self):
        self.driver.initialize()
        device = self.bus.device(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS)
        self.assertIn(0x91, device.read_log)
        self.driver.read_measurement(1)
        self.assertIn((0x91, 0x5A), device.write_log)

    def test_rejects_a_wrong_model_id(self):
        device = self.bus.device(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS)
        device.set_registers({0xC0: 0x42})
        with self.assertRaises(VL53L0XError) as caught:
            self.driver.initialize()
        self.assertIn("model id", str(caught.exception))

    def test_rejects_i2c_failure_during_initialize(self):
        self.bus.inject_fault(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS)
        with self.assertRaises(VL53L0XError):
            self.driver.initialize()

    def test_sets_and_reports_the_timing_budget(self):
        self.driver.initialize()
        registers = self.bus.device(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS).registers
        thirty_three_ms = (registers[0x71] << 8) | registers[0x72]
        self.assertEqual(self.driver.measurement_timing_budget_ms, 33.0)
        self.assertGreater(thirty_three_ms, 0x0100)

        fast_spec = ToFSensorSpec("tof-center", Sector.CENTER, 1, 0x29, 50.0)
        fast = VL53L0XDriver(self.mux, fast_spec)
        fast.initialize()
        registers = self.bus.device(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS).registers
        self.assertEqual(fast.measurement_timing_budget_ms, 50.0)
        self.assertNotEqual(thirty_three_ms, (registers[0x71] << 8) | registers[0x72])

    def test_rejects_an_impossible_timing_budget(self):
        self.driver.initialize()
        with self.assertRaises(VL53L0XError):
            self.driver.set_measurement_timing_budget_us(1000)

    def test_read_measurement_decodes_a_valid_range(self):
        self.driver.initialize()
        measurement = self.driver.read_measurement(42 * MS)
        self.assertTrue(measurement.valid)
        self.assertFalse(measurement.no_target)
        self.assertEqual(measurement.sensor_id, "tof-center")
        self.assertEqual(measurement.sector, Sector.CENTER)
        self.assertEqual(measurement.timestamp_ns, 42 * MS)
        self.assertAlmostEqual(measurement.distance_m, 0.85)
        self.assertEqual(measurement.raw_mm, 850)
        self.assertEqual(measurement.range_status, 11)
        self.assertTrue(measurement.as_dict()["valid"])

    def test_read_measurement_clears_the_interrupt(self):
        self.driver.initialize()
        device = self.bus.device(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS)
        device.write_log.clear()
        self.driver.read_measurement(1)
        self.assertIn((0x0B, 0x01), device.write_log)

    def test_decode_boundaries(self):
        self.assertEqual(decode_range_status(11, 30), (True, False, 0.03))
        self.assertEqual(decode_range_status(11, 2000), (True, False, 2.0))
        self.assertEqual(decode_range_status(11, 2001), (True, True, None))
        self.assertEqual(decode_range_status(11, 8190), (True, True, None))
        self.assertEqual(decode_range_status(11, 8191), (True, True, None))
        self.assertEqual(decode_range_status(None, None), (False, False, None))

    def test_decode_near_field_never_reports_clear_space(self):
        valid, no_target, distance = decode_range_status(11, 12)
        self.assertTrue(valid)
        self.assertFalse(no_target)
        self.assertAlmostEqual(distance, 0.012)

    def test_decode_no_target_and_hardware_fault_statuses(self):
        for status in (0, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15):
            with self.subTest(status=status):
                self.assertEqual(decode_range_status(status, 0), (True, True, None))
        for status in (1, 2, 3):
            with self.subTest(status=status):
                self.assertEqual(decode_range_status(status, 0), (False, False, None))
        self.assertEqual(decode_range_status(11, 500), (True, False, 0.5))

    def test_out_of_range_reports_no_target(self):
        self.driver.initialize()
        device = self.bus.device(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS)
        write_range_result(device.registers, 0, 8190)
        measurement = self.driver.read_measurement(5)
        self.assertTrue(measurement.valid)
        self.assertTrue(measurement.no_target)
        self.assertIsNone(measurement.distance_m)

        write_range_result(device.registers, 11, 2600)
        measurement = self.driver.read_measurement(6)
        self.assertTrue(measurement.valid)
        self.assertTrue(measurement.no_target)
        self.assertIsNone(measurement.distance_m)

    def test_i2c_fault_marks_the_reading_invalid_and_recovers(self):
        self.driver.initialize()
        self.bus.inject_fault(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS, occurrences=1)
        measurement = self.driver.read_measurement(7)
        self.assertFalse(measurement.valid)
        self.assertFalse(measurement.no_target)
        self.assertIsNone(measurement.distance_m)
        self.assertIsNone(measurement.raw_mm)

        measurement = self.driver.read_measurement(8)
        self.assertTrue(measurement.valid)
        self.assertAlmostEqual(measurement.distance_m, 0.85)

    def test_ranging_timeout_marks_the_reading_invalid(self):
        driver = VL53L0XDriver(self.mux, self.spec, ready_timeout_s=0.002)
        driver.initialize()
        self.bus.device(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS).set_registers({0x13: 0x00})
        measurement = driver.read_measurement(9)
        self.assertFalse(measurement.valid)
        self.assertIsNone(measurement.distance_m)

    def test_requires_initialize_before_ranging(self):
        with self.assertRaises(VL53L0XError):
            self.driver.read_measurement(1)

    def test_continuous_mode_reads_without_restarting_a_single_shot(self):
        self.driver.initialize()
        self.driver.start_continuous()
        device = self.bus.device(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS)
        self.assertEqual(device.registers[0x00], 0x02)
        device.write_log.clear()

        measurement = self.driver.read_measurement(11)
        self.assertTrue(measurement.valid)
        self.assertIn((0x0B, 0x01), device.write_log)
        self.assertNotIn(0x91, [register for register, _ in device.write_log])

        self.driver.stop_continuous()
        self.assertEqual(device.registers[0x00], 0x00)

    def test_timed_continuous_mode_writes_the_intermeasurement_period(self):
        self.driver.initialize()
        device = self.bus.device(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS)
        device.set_registers({0xF8: 0x00, 0xF9: 0x00})
        self.driver.start_continuous(period_ms=33)
        self.assertEqual(device.registers[0x00], 0x04)
        period = (
            (device.registers[0x04] << 24)
            | (device.registers[0x05] << 16)
            | (device.registers[0x06] << 8)
            | device.registers[0x07]
        )
        self.assertEqual(period, 33)

    def test_close_stops_ranging(self):
        self.driver.initialize()
        self.driver.start_continuous()
        self.driver.close()
        self.assertFalse(self.driver.initialized)
        device = self.bus.device(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS)
        self.assertEqual(device.registers[0x00], 0x00)


class VL53L0XArrayTests(unittest.TestCase):
    def setUp(self):
        self.config = load_hardware_config(CONFIG_PATH)
        self.bus = build_pi5_bus(self.config)
        self.mux = build_mux(self.bus)
        self.array = VL53L0XArray(self.mux, self.config.tof_sensors)

    def test_initializes_and_reads_all_three_sectors(self):
        self.assertEqual(self.array.initialize(), [])
        measurements = {item.sensor_id: item for item in self.array.read_all(100)}
        self.assertEqual(set(measurements), {"tof-left", "tof-center", "tof-right"})
        self.assertEqual(measurements["tof-left"].sector, Sector.LEFT)
        self.assertAlmostEqual(measurements["tof-left"].distance_m, 0.85)
        self.assertAlmostEqual(measurements["tof-center"].distance_m, 0.85)
        self.assertAlmostEqual(measurements["tof-right"].distance_m, 0.85)

    def test_channels_isolate_sensors_sharing_address_0x29(self):
        for channel, mm in ((0, 300), (1, 900), (2, 1500)):
            write_range_result(self.bus.device(channel, TOF_ADDRESS).registers, 11, mm)
        self.assertEqual(self.array.initialize(), [])
        measurements = {item.sensor_id: item for item in self.array.read_all(50)}
        self.assertAlmostEqual(measurements["tof-left"].distance_m, 0.3)
        self.assertAlmostEqual(measurements["tof-center"].distance_m, 0.9)
        self.assertAlmostEqual(measurements["tof-right"].distance_m, 1.5)

    def test_one_faulty_channel_does_not_blind_the_others(self):
        self.bus.inject_fault(TOF_CHANNELS[Sector.LEFT], TOF_ADDRESS)
        failures = self.array.initialize()
        self.assertEqual(failures, ["tof-left"])
        self.assertEqual(self.array.faulted_sensor_ids, ["tof-left"])
        self.assertEqual(self.array.quarantined_sensor_ids, ["tof-left"])
        self.assertIn("tof-left", self.array.fault_reason("tof-left"))

        measurements = {item.sensor_id: item for item in self.array.read_all(200)}
        self.assertFalse(measurements["tof-left"].valid)
        self.assertIsNone(measurements["tof-left"].distance_m)
        for sensor_id in ("tof-center", "tof-right"):
            self.assertTrue(measurements[sensor_id].valid)
            self.assertAlmostEqual(measurements[sensor_id].distance_m, 0.85)

    def test_fault_after_initialization_is_isolated(self):
        self.assertEqual(self.array.initialize(), [])
        self.bus.inject_fault(TOF_CHANNELS[Sector.CENTER], TOF_ADDRESS)
        first = {item.sensor_id: item for item in self.array.read_all(300)}
        self.assertFalse(first["tof-center"].valid)
        self.assertTrue(first["tof-left"].valid)
        self.assertTrue(first["tof-right"].valid)

        # The fault is remembered, so a second poll cannot accidentally report
        # a stale healthy reading from the failed channel.
        second = {item.sensor_id: item for item in self.array.read_all(400)}
        self.assertFalse(second["tof-center"].valid)
        self.assertIn("tof-center", self.array.faulted_sensor_ids)
        self.assertEqual(self.array.quarantined_sensor_ids, [])

        # A transient bus glitch heals on the next poll without re-initializing.
        self.bus.clear_faults()
        third = {item.sensor_id: item for item in self.array.read_all(500)}
        self.assertTrue(third["tof-center"].valid)
        self.assertEqual(self.array.faulted_sensor_ids, [])

    def test_reinitialize_recovers_a_faulted_sensor(self):
        self.bus.inject_fault(TOF_CHANNELS[Sector.LEFT], TOF_ADDRESS)
        self.assertEqual(self.array.initialize(), ["tof-left"])
        self.bus.clear_faults()
        self.array.initialize_sensor("tof-left")
        self.assertEqual(self.array.faulted_sensor_ids, [])
        measurements = {item.sensor_id: item for item in self.array.read_all(500)}
        self.assertTrue(all(item.valid for item in measurements.values()))

    def test_rejects_an_empty_array(self):
        with self.assertRaises(ValueError):
            VL53L0XArray(self.mux, ())


# ---------------------------------------------------------------------------
# HC-SR04
# ---------------------------------------------------------------------------


class HCSR04PairDriverTests(unittest.TestCase):
    def test_configures_trigger_outputs_and_echo_inputs(self):
        driver, gpio, _ = build_ultrasonic({24: 0.005, 22: 0.005})
        self.assertEqual(gpio.outputs, [23, 27])
        self.assertEqual(gpio.inputs, [24, 22])
        self.assertEqual(gpio.trigger_writes[23], [False])
        driver.close()
        self.assertTrue(gpio.closed)

    def test_alternates_sensors_in_strict_tdm(self):
        driver, gpio, _ = build_ultrasonic({24: 0.005, 22: 0.005})
        first = driver.read_next(0)
        self.assertEqual(first.sensor_id, "us-center-1")
        self.assertEqual(gpio.pulse_calls[-1][0], 24)

        second = driver.read_next(60 * MS)
        self.assertEqual(second.sensor_id, "us-center-2")
        self.assertEqual(gpio.pulse_calls[-1][0], 22)

        third = driver.read_next(120 * MS)
        self.assertEqual(third.sensor_id, "us-center-1")
        self.assertEqual(gpio.trigger_writes[23], [False, True, False, True, False])

    def test_enforces_the_guard_interval_between_pings(self):
        driver, _, clock = build_ultrasonic({24: 0.010, 22: 0.010})
        first = driver.read_next(0)
        self.assertEqual(first.timestamp_ns, 0)
        self.assertEqual(driver.last_ping_end_ns, 10 * MS)

        clock.now_ns = 20 * MS
        second = driver.read_next()
        self.assertEqual(clock.sleeps, [0.025])
        self.assertEqual(second.timestamp_ns, 45 * MS)
        self.assertEqual(driver.last_ping_end_ns, 55 * MS)

    def test_does_not_wait_when_the_guard_is_already_satisfied(self):
        driver, _, clock = build_ultrasonic({24: 0.005, 22: 0.005})
        driver.read_next(0)
        clock.advance_ms(100)
        driver.read_next()
        self.assertEqual(clock.sleeps, [])

    def test_temperature_compensated_distance(self):
        driver, _, _ = build_ultrasonic({24: 0.0058}, ambient_temp_c=22.0)
        measurement = driver.read_next(0)
        expected = speed_of_sound_mps(22.0) * 0.0058 / 2.0
        self.assertAlmostEqual(measurement.distance_m, expected, places=9)
        self.assertAlmostEqual(expected, 0.99943, places=5)

        cold, _, _ = build_ultrasonic({24: 0.0058}, ambient_temp_c=0.0)
        self.assertAlmostEqual(cold.read_next(0).distance_m, 331.3 * 0.0058 / 2.0, places=9)
        self.assertGreater(expected, speed_of_sound_mps(0.0) * 0.0058 / 2.0)

        hot, _, _ = build_ultrasonic({24: 0.0058}, ambient_temp_c=35.0)
        self.assertGreater(hot.read_next(0).distance_m, expected)

    def test_listening_timeout_reports_clear_space(self):
        driver, _, _ = build_ultrasonic({24: None})
        measurement = driver.read_next(0)
        self.assertTrue(measurement.valid)
        self.assertTrue(measurement.no_target)
        self.assertIsNone(measurement.distance_m)
        self.assertEqual(driver.last_ping_end_ns, int(25 * MS))

    def test_echo_beyond_max_range_reports_clear_space(self):
        driver, _, _ = build_ultrasonic({24: 0.03})
        measurement = driver.read_next(0)
        self.assertTrue(measurement.valid)
        self.assertTrue(measurement.no_target)
        self.assertIsNone(measurement.distance_m)

    def test_near_field_echo_is_never_reported_clear(self):
        driver, _, _ = build_ultrasonic({24: 0.0001})
        measurement = driver.read_next(0)
        self.assertTrue(measurement.valid)
        self.assertFalse(measurement.no_target)
        self.assertLess(measurement.distance_m, 0.04)

    def test_stuck_high_echo_is_a_fault(self):
        driver, _, _ = build_ultrasonic({24: EchoStuckHighError("stuck")})
        measurement = driver.read_next(0)
        self.assertFalse(measurement.valid)
        self.assertFalse(measurement.no_target)
        self.assertIsNone(measurement.distance_m)

    def test_gpio_errors_are_faults(self):
        for error in (OSError(errno.EIO, "boom"), GPIOException("boom")):
            with self.subTest(error=type(error).__name__):
                driver, _, _ = build_ultrasonic({24: error})
                measurement = driver.read_next(0)
                self.assertFalse(measurement.valid)
                self.assertFalse(measurement.no_target)

    def test_read_all_sequential_pings_each_sensor_with_the_guard(self):
        driver, gpio, clock = build_ultrasonic({24: 0.005, 22: 0.005})
        measurements = driver.read_all_sequential()
        self.assertEqual([item.sensor_id for item in measurements], ["us-center-1", "us-center-2"])
        self.assertEqual([call[0] for call in gpio.pulse_calls], [24, 22])
        self.assertEqual(len(clock.sleeps), 1)
        self.assertAlmostEqual(clock.sleeps[0], 0.035)
        self.assertEqual(driver.read_next().sensor_id, "us-center-1")

    def test_validates_construction_arguments(self):
        config = load_hardware_config(CONFIG_PATH)
        with self.assertRaises(ValueError):
            HCSR04PairDriver(FakeGPIO(), ())
        with self.assertRaises(ValueError):
            HCSR04PairDriver(FakeGPIO(), config.ultrasonic_sensors, guard_interval_ms=0)
        with self.assertRaises(ValueError):
            HCSR04PairDriver(FakeGPIO(), config.ultrasonic_sensors, timeout_ms=0)
        duplicate = (config.ultrasonic_sensors[0], config.ultrasonic_sensors[0])
        with self.assertRaises(ValueError):
            HCSR04PairDriver(FakeGPIO(), duplicate)


class GPIOBackendTests(unittest.TestCase):
    def test_lgpio_reports_install_hint_when_missing(self):
        with unittest.mock.patch.dict(sys.modules, {"lgpio": None}):
            with self.assertRaises(RuntimeError) as caught:
                LgpioBackend()
        self.assertIn("lgpio", str(caught.exception))

    def test_lgpio_falls_back_from_rp1_chip_to_gpiochip_zero(self):
        fake = make_fake_lgpio(unavailable_chips=(4,))
        with unittest.mock.patch.dict(sys.modules, {"lgpio": fake}):
            backend = LgpioBackend(chip=4)
            backend.configure_output(23)
            backend.configure_input(24)
            backend.write(23, True)
            self.assertTrue(backend.read(23))
            self.assertEqual(backend.open_chip, 0)
            self.assertEqual(fake.opened, [0])
            self.assertEqual(fake.claimed, [("out", 23), ("in", 24)])
        backend.close()
        self.assertEqual(fake.closed_handles, [100])

    def test_lgpio_pulse_reports_no_echo_and_stuck_high(self):
        fake = make_fake_lgpio(unavailable_chips=())
        with unittest.mock.patch.dict(sys.modules, {"lgpio": fake}):
            backend = LgpioBackend(chip=4)
            backend.configure_input(24)
            self.assertIsNone(backend.pulse_echo_s(24, 0.001))
            fake.levels[24] = 1
            with self.assertRaises(EchoStuckHighError):
                backend.pulse_echo_s(24, 0.001)
        backend.close()

    def test_factory_reports_both_missing_backends(self):
        with unittest.mock.patch.dict(sys.modules, {"lgpio": None, "gpiozero": None}):
            with self.assertRaises(RuntimeError) as caught:
                create_gpio_backend()
        message = str(caught.exception)
        self.assertIn("lgpio", message)
        self.assertIn("gpiozero", message)


# ---------------------------------------------------------------------------
# MPU-6050
# ---------------------------------------------------------------------------


class MPU6050DriverTests(unittest.TestCase):
    def setUp(self):
        self.config = load_hardware_config(CONFIG_PATH)
        self.bus = build_pi5_bus(self.config)
        self.mux = build_mux(self.bus)
        self.driver = MPU6050Driver(self.mux, self.config.imu, settle_s=0.0)
        self.device = self.bus.device(self.config.imu.mux_channel, IMU_ADDRESS)

    def test_initializes_registers(self):
        self.driver.initialize()
        self.assertTrue(self.driver.initialized)
        self.assertEqual(self.driver.who_am_i, 0x68)
        self.assertEqual(self.device.registers[0x6B], 0x01)
        self.assertEqual(self.device.registers[0x19], 0x09)
        self.assertEqual(self.device.registers[0x1A], 0x03)
        self.assertEqual(self.device.registers[0x1B], 0x00)
        self.assertEqual(self.device.registers[0x1C], 0x00)
        self.assertEqual(self.driver.sample_rate_divider, 9)

    def test_accepts_a_clone_who_am_i_with_a_warning(self):
        self.device.set_registers({0x75: 0x70})
        with self.assertLogs("yolo_pi.hardware.mpu6050", level="WARNING") as logs:
            self.driver.initialize()
        self.assertTrue(self.driver.initialized)
        self.assertIn("0x70", "".join(logs.output))

    def test_rejects_an_unknown_who_am_i(self):
        self.device.set_registers({0x75: 0x12})
        with self.assertRaises(MPU6050Error):
            self.driver.initialize()

    def test_fails_when_the_device_stays_asleep(self):
        original = self.bus.write_byte_data

        def write_byte_data(address, register, value):
            original(address, register, value)
            if address == IMU_ADDRESS and register == 0x6B:
                self.device.registers[0x6B] = 0x41

        self.bus.write_byte_data = write_byte_data
        with self.assertRaises(MPU6050Error):
            self.driver.initialize()

    def test_decodes_a_fourteen_byte_burst(self):
        self.driver.initialize()
        payload = imu_payload(
            accel=(16384, -16384, 0),
            gyro=(131, -131, 0),
            temperature_raw=3300,
        )
        for offset, value in enumerate(payload):
            self.device.registers[0x3B + offset] = value

        sample = self.driver.read_raw(777)
        self.assertEqual(sample.timestamp_ns, 777)
        self.assertEqual(sample.accel_raw, (16384, -16384, 0))
        self.assertEqual(sample.gyro_raw, (131, -131, 0))
        self.assertAlmostEqual(sample.accel_mps2[0], 9.80665, places=5)
        self.assertAlmostEqual(sample.accel_mps2[1], -9.80665, places=5)
        self.assertAlmostEqual(sample.accel_mps2[2], 0.0, places=5)
        self.assertAlmostEqual(sample.gyro_rad_s[0], 0.0174533, places=6)
        self.assertAlmostEqual(sample.gyro_rad_s[1], -0.0174533, places=6)
        self.assertAlmostEqual(sample.temperature_c, 3300 / 340.0 + 36.53, places=6)
        self.assertEqual(list(self.device.read_log[-14:]), list(range(0x3B, 0x3B + 14)))

    def test_sample_exposes_imu_sample_fields(self):
        self.driver.initialize()
        payload = imu_payload(accel=(16384, 0, 0), gyro=(0, 131, 0), temperature_raw=3400)
        for offset, value in enumerate(payload):
            self.device.registers[0x3B + offset] = value

        fields = self.driver.read_raw(3).as_imu_fields()
        self.assertEqual(
            set(fields),
            {
                "timestamp_ns",
                "ax_mps2",
                "ay_mps2",
                "az_mps2",
                "gx_rps",
                "gy_rps",
                "gz_rps",
                "temp_c",
                "valid",
            },
        )
        self.assertEqual(fields["timestamp_ns"], 3)
        self.assertAlmostEqual(fields["ax_mps2"], 9.80665, places=5)
        self.assertAlmostEqual(fields["gy_rps"], 0.0174533, places=6)
        self.assertAlmostEqual(fields["temp_c"], 3400 / 340.0 + 36.53, places=6)
        self.assertTrue(fields["valid"])

    def test_decodes_negative_temperature_and_extreme_values(self):
        self.driver.initialize()
        payload = imu_payload(
            accel=(-32768, 32767, 1),
            gyro=(-32768, 32767, -1),
            temperature_raw=-680,
        )
        for offset, value in enumerate(payload):
            self.device.registers[0x3B + offset] = value
        sample = self.driver.read_raw(1)
        self.assertEqual(sample.accel_raw, (-32768, 32767, 1))
        self.assertEqual(sample.gyro_raw, (-32768, 32767, -1))
        self.assertAlmostEqual(sample.temperature_c, -680 / 340.0 + 36.53, places=6)
        self.assertLess(sample.accel_mps2[0], -19.0)
        self.assertGreater(sample.accel_mps2[1], 19.0)

    def test_calibrate_startup_measures_biases(self):
        self.driver.initialize()
        payload = imu_payload(accel=(0, 0, 16384), gyro=(131, 0, 0), temperature_raw=3400)
        for offset, value in enumerate(payload):
            self.device.registers[0x3B + offset] = value

        calibration = self.driver.calibrate_startup(num_samples=5, delay_s=0.0)
        self.assertEqual(calibration.samples, 5)
        self.assertAlmostEqual(calibration.gyro_bias_rad_s[0], 0.0174533, places=6)
        self.assertAlmostEqual(calibration.gyro_bias_rad_s[1], 0.0, places=9)
        for bias in calibration.accel_bias_mps2:
            self.assertAlmostEqual(bias, 0.0, places=6)
        self.assertAlmostEqual(calibration.temperature_c, 3400 / 340.0 + 36.53, places=6)
        self.assertEqual(self.driver.calibration, calibration)
        self.assertEqual(calibration.as_tuple()[1], calibration.gyro_bias_rad_s)

    def test_calibration_rejects_bad_arguments(self):
        self.driver.initialize()
        with self.assertRaises(ValueError):
            self.driver.calibrate_startup(num_samples=0)
        with self.assertRaises(ValueError):
            self.driver.calibrate_startup(delay_s=-1.0)

    def test_i2c_faults_raise(self):
        self.driver.initialize()
        self.bus.inject_fault(self.config.imu.mux_channel, IMU_ADDRESS)
        with self.assertRaises(MPU6050Error):
            self.driver.read_raw(1)

    def test_requires_initialize(self):
        with self.assertRaises(MPU6050Error):
            self.driver.read_raw(1)

    def test_imu_can_sit_on_the_main_bus_instead_of_a_channel(self):
        bus = FakeSMBus()
        bus.add_device(None, IMU_ADDRESS, imu_registers())
        mux = build_mux(bus)
        driver = MPU6050Driver(mux, IMUSpec("imu", None, 0x68, 17, 100.0), settle_s=0.0)
        driver.initialize()
        self.assertEqual(bus.mux_writes, [0x00])
        self.assertEqual(driver.sample_rate_divider, 9)

    def test_close_puts_the_device_to_sleep(self):
        self.driver.initialize()
        self.driver.close()
        self.assertEqual(self.device.registers[0x6B], 0x40)
        self.assertFalse(self.driver.initialized)


# ---------------------------------------------------------------------------
# DRV2605L
# ---------------------------------------------------------------------------


class DRV2605LControllerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_hardware_config(CONFIG_PATH)
        self.bus = build_pi5_bus(self.config)
        self.mux = build_mux(self.bus)
        self.controller = DRV2605LController(self.mux, self.config.haptic_motors)

    def device(self, sector):
        return self.bus.device(HAPTIC_CHANNELS[sector], HAPTIC_ADDRESS)

    def rtp(self, sector):
        return self.device(sector).registers[0x02]

    def command(self, intensity, period_ms=None, duty_cycle=1.0, waveform_id=None):
        return SimpleNamespace(
            intensity=intensity,
            pulse_period_ms=period_ms,
            duty_cycle=duty_cycle,
            waveform_id=waveform_id,
        )

    def test_initializes_all_three_3v_erm_channels(self):
        self.controller.initialize()
        self.assertTrue(self.controller.initialized)
        for sector in (Sector.LEFT, Sector.CENTER, Sector.RIGHT):
            with self.subTest(sector=sector):
                registers = self.device(sector).registers
                self.assertEqual(registers[0x01], 0x05)
                self.assertEqual(registers[0x02], 0x00)
                self.assertEqual(registers[0x03], 0x01)
                self.assertEqual(registers[0x04], 0x00)
                self.assertEqual(registers[0x05], 0x00)
                self.assertEqual(registers[0x16], 0x7F)
                self.assertEqual(registers[0x17], 0x96)
                self.assertEqual(registers[0x1A] & 0x80, 0x00)
                self.assertEqual(registers[0x1D] & 0x08, 0x08)
                self.assertEqual(registers[0x1D] & 0x20, 0x00)

    def test_init_clears_lra_and_open_loop_bits(self):
        self.controller.initialize()
        registers = self.device(Sector.LEFT).registers
        self.assertEqual(registers[0x1A], 0x00)
        self.assertEqual(registers[0x1D], 0x08)

    def test_one_dead_channel_keeps_the_other_two_working(self):
        self.bus.inject_fault(HAPTIC_CHANNELS[Sector.CENTER], HAPTIC_ADDRESS)
        with self.assertLogs("yolo_pi.hardware.drv2605l", level="WARNING") as logs:
            self.controller.initialize()
        self.assertEqual(self.controller.failed_sectors, [Sector.CENTER])
        self.assertIn("haptic-center", "".join(logs.output))

        self.controller.apply_sector_states(
            {Sector.LEFT: self.command(1.0), Sector.RIGHT: self.command(1.0)}, 1000 * MS
        )
        self.assertEqual(self.rtp(Sector.LEFT), 127)
        self.assertEqual(self.rtp(Sector.RIGHT), 127)

    def test_losing_every_motor_raises(self):
        for channel in HAPTIC_CHANNELS.values():
            self.bus.inject_fault(channel, HAPTIC_ADDRESS)
        with self.assertLogs("yolo_pi.hardware.drv2605l", level="WARNING"):
            with self.assertRaises(DRV2605LError) as caught:
                self.controller.initialize()
        self.assertIn("no haptic motor", str(caught.exception))

    def test_applies_independent_sector_intensities(self):
        self.controller.initialize()
        self.controller.apply_sector_states(
            {
                Sector.LEFT: self.command(1.0),
                Sector.CENTER: self.command(0.4),
                Sector.RIGHT: self.command(0.0),
            },
            1000 * MS,
        )
        self.assertEqual(self.rtp(Sector.LEFT), 127)
        self.assertEqual(self.rtp(Sector.CENTER), 51)
        self.assertEqual(self.rtp(Sector.RIGHT), 0)
        self.assertEqual(self.controller.last_errors, {})

    def test_skips_redundant_rtp_writes(self):
        self.controller.initialize()
        frame = {Sector.LEFT: self.command(0.5), Sector.CENTER: self.command(0.5)}
        self.controller.apply_sector_states(frame, 1000 * MS)
        writes_after_first = len(self.device(Sector.LEFT).write_log)
        self.assertEqual(self.rtp(Sector.LEFT), 64)

        for step in range(1, 5):
            self.controller.apply_sector_states(frame, (1000 + step * 33) * MS)
        self.assertEqual(len(self.device(Sector.LEFT).write_log), writes_after_first)
        self.assertEqual(len(self.device(Sector.CENTER).write_log), writes_after_first)

    def test_pulse_phase_modulation_is_deterministic(self):
        self.controller.initialize()
        frame = {Sector.LEFT: self.command(1.0, period_ms=100, duty_cycle=0.5)}
        start = 5_000 * MS
        for offset_ms, expected in ((0, 127), (30, 127), (49, 127), (51, 0), (99, 0), (101, 127)):
            self.controller.apply_sector_states(frame, start + offset_ms * MS)
            with self.subTest(offset_ms=offset_ms):
                self.assertEqual(self.rtp(Sector.LEFT), expected)

    def test_pulse_duty_cycle_scales_the_on_window(self):
        self.controller.initialize()
        frame = {Sector.RIGHT: self.command(1.0, period_ms=40, duty_cycle=0.25)}
        start = 0
        for offset_ms, expected in ((0, 127), (9, 127), (11, 0), (39, 0), (41, 127)):
            self.controller.apply_sector_states(frame, start + offset_ms * MS)
            with self.subTest(offset_ms=offset_ms):
                self.assertEqual(self.rtp(Sector.RIGHT), expected)

    def test_a_sector_missing_from_the_frame_is_driven_to_zero(self):
        self.controller.initialize()
        self.controller.apply_sector_states(
            {Sector.LEFT: self.command(1.0), Sector.CENTER: self.command(1.0)}, 1000 * MS
        )
        self.controller.apply_sector_states({Sector.LEFT: self.command(0.0)}, 1033 * MS)
        self.assertEqual(self.rtp(Sector.LEFT), 0)
        self.assertEqual(self.rtp(Sector.CENTER), 0)
        self.assertEqual(self.rtp(Sector.RIGHT), 0)

    def test_pulse_restarts_on_the_on_phase_after_a_gap(self):
        self.controller.initialize()
        frame = {Sector.LEFT: self.command(1.0, period_ms=100, duty_cycle=0.5)}
        self.controller.apply_sector_states(frame, 1000 * MS)
        self.assertEqual(self.rtp(Sector.LEFT), 127)
        self.controller.apply_sector_states(frame, 1060 * MS)
        self.assertEqual(self.rtp(Sector.LEFT), 0)

        # The hazard disappears for a frame; the next burst starts immediately
        # in its ON phase instead of waiting out the old phase offset.
        self.controller.apply_sector_states({}, 1070 * MS)
        self.assertEqual(self.rtp(Sector.LEFT), 0)
        self.controller.apply_sector_states(frame, 1080 * MS)
        self.assertEqual(self.rtp(Sector.LEFT), 127)

    def test_watchdog_zeroes_all_motors_after_silence(self):
        self.controller.initialize()
        self.controller.apply_sector_states(
            {
                Sector.LEFT: self.command(1.0),
                Sector.CENTER: self.command(0.8),
                Sector.RIGHT: self.command(0.6),
            },
            1000 * MS,
        )
        self.assertFalse(self.controller.tick_watchdog(1400 * MS))
        self.assertFalse(self.controller.watchdog_tripped)
        self.assertEqual(self.rtp(Sector.CENTER), 102)

        self.assertTrue(self.controller.tick_watchdog(1600 * MS))
        self.assertTrue(self.controller.watchdog_tripped)
        for sector in (Sector.LEFT, Sector.CENTER, Sector.RIGHT):
            self.assertEqual(self.rtp(sector), 0)

    def test_watchdog_does_nothing_before_the_first_frame(self):
        self.controller.initialize()
        self.assertFalse(self.controller.tick_watchdog(10_000 * MS))
        self.assertFalse(self.controller.watchdog_tripped)

    def test_frame_after_watchdog_rearms_the_motors(self):
        self.controller.initialize()
        frame = {Sector.LEFT: self.command(1.0)}
        self.controller.apply_sector_states(frame, 1000 * MS)
        self.assertTrue(self.controller.tick_watchdog(2000 * MS))
        self.assertTrue(self.controller.watchdog_tripped)
        self.controller.apply_sector_states(frame, 3000 * MS)
        self.assertFalse(self.controller.watchdog_tripped)
        self.assertEqual(self.rtp(Sector.LEFT), 127)

    def test_stop_all_survives_one_failed_channel(self):
        self.controller.initialize()
        self.controller.apply_sector_states(
            {
                Sector.LEFT: self.command(1.0),
                Sector.CENTER: self.command(1.0),
                Sector.RIGHT: self.command(1.0),
            },
            1000 * MS,
        )
        self.bus.inject_fault(HAPTIC_CHANNELS[Sector.CENTER], HAPTIC_ADDRESS)
        self.controller.stop_all()

        self.assertEqual(self.rtp(Sector.LEFT), 0)
        self.assertEqual(self.rtp(Sector.RIGHT), 0)
        self.assertEqual(self.rtp(Sector.CENTER), 127)
        self.assertIn(Sector.CENTER, self.controller.last_errors)

        self.bus.clear_faults()
        self.controller.stop_all()
        self.assertEqual(self.rtp(Sector.CENTER), 0)
        self.assertEqual(self.controller.last_errors, {})

    def test_library_mode_triggers_a_waveform_once_per_pulse(self):
        controller = DRV2605LController(self.mux, self.config.haptic_motors, mode="library")
        controller.initialize()
        registers = self.device(Sector.LEFT).registers
        self.assertEqual(registers[0x01], 0x00)

        command = self.command(1.0, waveform_id=5)
        controller.apply_sector_states({Sector.LEFT: command}, 1000 * MS)
        self.assertEqual(registers[0x04], 5)
        self.assertEqual(registers[0x0C], 0x01)
        go_writes = len(self.device(Sector.LEFT).register_writes(0x0C))

        controller.apply_sector_states({Sector.LEFT: command}, 1033 * MS)
        self.assertEqual(len(self.device(Sector.LEFT).register_writes(0x0C)), go_writes)

        controller.stop_all()
        self.assertEqual(registers[0x01], 0x40)
        self.assertEqual(registers[0x02], 0x00)

    def test_emit_frame_accepts_a_duck_typed_directional_frame(self):
        self.controller.initialize()
        frame = SimpleNamespace(
            timestamp_ns=2000 * MS,
            sector_commands={
                "left": self.command(1.0),
                "right": SimpleNamespace(intensity=0.5, pulse_period_ms=None),
            },
        )
        self.controller.emit_frame(frame)
        self.assertEqual(self.rtp(Sector.LEFT), 127)
        self.assertEqual(self.rtp(Sector.RIGHT), 64)
        self.assertEqual(self.rtp(Sector.CENTER), 0)

    def test_emit_frame_accepts_a_fusion_v2_shaped_frame(self):
        """The frame's per-sector commands live in ``motors`` in fusion_v2."""

        self.controller.initialize()
        frame = SimpleNamespace(
            timestamp_ns=2500 * MS,
            overall_level="critical",
            reason="shaped like DirectionalHapticFrame",
            motors={
                Sector.LEFT: SimpleNamespace(
                    intensity=1.0, pulse_period_ms=100, duty_cycle=0.5, waveform_id=1, reason="left"
                ),
                Sector.RIGHT: SimpleNamespace(
                    intensity=0.0, pulse_period_ms=None, duty_cycle=1.0, waveform_id=1, reason="right"
                ),
            },
        )
        self.controller.emit_frame(frame)
        self.assertEqual(self.rtp(Sector.LEFT), 127)
        self.assertEqual(self.rtp(Sector.RIGHT), 0)
        self.assertEqual(self.rtp(Sector.CENTER), 0)

    def test_emit_frame_accepts_a_mapping(self):
        self.controller.initialize()
        self.controller.emit_frame({Sector.CENTER: self.command(0.25), "timestamp_ns": 3000 * MS})
        self.assertEqual(self.rtp(Sector.CENTER), 32)

    def test_stop_zeroes_every_motor(self):
        self.controller.initialize()
        self.controller.apply_sector_states(
            {Sector.LEFT: self.command(1.0), Sector.RIGHT: self.command(1.0)}, 1000 * MS
        )
        self.controller.stop()
        for sector in (Sector.LEFT, Sector.CENTER, Sector.RIGHT):
            self.assertEqual(self.rtp(sector), 0)

    def test_emit_adapts_a_single_haptic_command(self):
        self.controller.initialize()
        self.controller.emit(
            SimpleNamespace(
                timestamp_ns=4000 * MS,
                sectors=(Sector.RIGHT,),
                intensity=1.0,
                pulse_period_ms=None,
            )
        )
        self.assertEqual(self.rtp(Sector.RIGHT), 127)

    def test_rejects_apply_before_initialize_and_after_close(self):
        with self.assertRaises(DRV2605LError):
            self.controller.apply_sector_states({}, 0)
        self.controller.initialize()
        self.controller.close()
        with self.assertRaises(DRV2605LError):
            self.controller.apply_sector_states({}, 0)

    def test_validates_configuration(self):
        with self.assertRaises(ValueError):
            DRV2605LController(self.mux, self.config.haptic_motors, mode="magic")
        with self.assertRaises(ValueError):
            DRV2605LController(self.mux, self.config.haptic_motors, watchdog_timeout_ms=0)
        with self.assertRaises(ValueError):
            DRV2605LController(self.mux, self.config.haptic_motors[:2])
        duplicated = (
            HapticMotorSpec("haptic-left", Sector.LEFT, 4),
            HapticMotorSpec("haptic-center", Sector.CENTER, 4),
            HapticMotorSpec("haptic-right", Sector.RIGHT, 6),
        )
        with self.assertRaises(ValueError):
            DRV2605LController(self.mux, duplicated)


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


class PackageSurfaceTests(unittest.TestCase):
    def test_exports_exist(self):
        import yolo_pi.hardware as hardware

        for name in hardware.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(hardware, name))

    def test_camera_spec_matches_the_report_default(self):
        spec = default_pi5_config().camera
        self.assertEqual(spec, CameraSpec())

    def test_vl53l0x_array_continuous_nonblocking_skips_unready_and_faults_stalled(self):
        from yolo_pi.hardware.vl53l0x import RESULT_INTERRUPT_STATUS, VL53L0XArray

        config = default_pi5_config()
        bus = FakeSMBus()
        for spec in config.tof_sensors:
            bus.add_device(spec.mux_channel, spec.i2c_address, tof_registers(mm=600))
        mux = TCA9548AMux(bus, config.mux_address)
        array = VL53L0XArray(mux, config.tof_sensors)
        self.assertEqual(array.initialize(), [])
        self.assertEqual(array.start_continuous(), [])

        # Simulate CH0 (left) not ready yet while CH1 and CH2 are ready.
        left_spec = config.tof_sensors[0]
        left_dev = bus.device(left_spec.mux_channel, left_spec.i2c_address)
        left_dev.registers[RESULT_INTERRUPT_STATUS] = 0x00
        for spec in config.tof_sensors[1:]:
            bus.device(spec.mux_channel, spec.i2c_address).registers[RESULT_INTERRUPT_STATUS] = 0x04

        t0 = 1_000_000_000
        array._last_ready_ns[left_spec.sensor_id] = t0
        first = array.read_all(now_ns=t0 + 20_000_000)
        self.assertEqual([m.sensor_id for m in first], ["tof-center", "tof-right"])
        self.assertTrue(all(m.valid for m in first))

        # Past the 250 ms ready timeout, left emits a fault without blocking center/right.
        for spec in config.tof_sensors[1:]:
            bus.device(spec.mux_channel, spec.i2c_address).registers[RESULT_INTERRUPT_STATUS] = 0x04
        second = array.read_all(now_ns=t0 + 400_000_000)
        by_id = {m.sensor_id: m for m in second}
        self.assertFalse(by_id["tof-left"].valid)
        self.assertTrue(by_id["tof-center"].valid)
        self.assertTrue(by_id["tof-right"].valid)

    def test_drv2605l_library_mode_writes_go_zero_on_clear(self):
        from yolo_pi.hardware.drv2605l import GO

        config = default_pi5_config()
        bus = FakeSMBus()
        for spec in config.haptic_motors:
            bus.add_device(spec.mux_channel, spec.i2c_address, haptic_registers())
        mux = TCA9548AMux(bus, config.mux_address)
        controller = DRV2605LController(mux, config.haptic_motors, mode="library")
        controller.initialize()

        left_spec = next(s for s in config.haptic_motors if s.sector == Sector.LEFT)
        left_dev = bus.device(left_spec.mux_channel, left_spec.i2c_address)

        controller.apply_sector_states(
            {Sector.LEFT: {"intensity": 0.8, "waveform_id": 14}},
            now_ns=100_000_000,
        )
        self.assertEqual(left_dev.registers[GO], 0x01)

        controller.apply_sector_states(
            {Sector.LEFT: {"intensity": 0.0}},
            now_ns=150_000_000,
        )
        self.assertEqual(left_dev.registers[GO], 0x00)

if __name__ == "__main__":
    unittest.main()
