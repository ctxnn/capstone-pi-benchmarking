import unittest

from yolo_pi.cane.config import TofConfig, UltrasonicConfig, UltrasonicSensorSpec
from yolo_pi.cane.tof import Vl53l0xDevice, convert_tof_mm
from yolo_pi.cane.ultrasonic import Hcsr04, echo_to_distance_m
from yolo_pi.types import Sector


class ConvertTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TofConfig()

    def test_normal_reading_is_metres(self) -> None:
        sample = convert_tof_mm(450, self.config)
        self.assertTrue(sample.valid)
        self.assertAlmostEqual(sample.distance_m, 0.45)
        self.assertFalse(sample.out_of_range)

    def test_out_of_range_codes_are_a_valid_clear_reading_not_a_fault(self) -> None:
        for code in (8190, 8191, 65535):
            sample = convert_tof_mm(code, self.config)
            self.assertTrue(sample.valid, code)
            self.assertTrue(sample.out_of_range, code)
            self.assertEqual(sample.distance_m, self.config.clear_distance_m)

    def test_zero_is_an_invalid_measurement(self) -> None:
        sample = convert_tof_mm(0, self.config)
        self.assertFalse(sample.valid)
        self.assertIsNone(sample.distance_m)

    def test_far_reading_is_capped_at_clear_distance(self) -> None:
        self.assertEqual(convert_tof_mm(3000, self.config).distance_m, self.config.clear_distance_m)


class FakeVl53:
    def __init__(self) -> None:
        self.io_timeout_s = 0
        self.measurement_timing_budget = 0
        self.continuous = False
        self.data_ready = False
        self.range = 0
        self.stopped = False

    def start_continuous(self) -> None:
        self.continuous = True

    def stop_continuous(self) -> None:
        self.stopped = True


class DeviceTests(unittest.TestCase):
    def test_start_configures_and_poll_is_non_blocking(self) -> None:
        fake = FakeVl53()
        device = Vl53l0xDevice(lambda: fake, TofConfig(timing_budget_us=20000, io_timeout_s=0.1))
        device.start()
        self.assertTrue(fake.continuous)
        self.assertEqual(fake.measurement_timing_budget, 20000)
        self.assertEqual(fake.io_timeout_s, 0.1)
        self.assertIsNone(device.poll())  # nothing ready yet: no blocking, no sample
        fake.data_ready, fake.range = True, 800
        self.assertAlmostEqual(device.poll().distance_m, 0.8)

    def test_poll_before_start_is_an_error(self) -> None:
        with self.assertRaises(RuntimeError):
            Vl53l0xDevice(FakeVl53, TofConfig()).poll()

    def test_restart_builds_a_fresh_sensor(self) -> None:
        made = []

        def factory() -> FakeVl53:
            made.append(FakeVl53())
            return made[-1]

        device = Vl53l0xDevice(factory, TofConfig())
        device.start()
        device.restart()
        self.assertEqual(len(made), 2)
        self.assertTrue(made[0].stopped)
        self.assertTrue(made[1].continuous)


class FakeGpio:
    """Plays an echo back synchronously when the trigger line falls."""

    def __init__(self, echo_ns=None, echo_gap_ns: int = 700_000) -> None:
        self.echo_ns = echo_ns
        self.echo_gap_ns = echo_gap_ns
        self.writes = []
        self._callback = None
        self._clock = 1_000_000_000

    def claim_output(self, pin, level=0):
        pass

    def claim_echo(self, pin, on_edge):
        self._callback = on_edge

    def write(self, pin, level):
        self.writes.append((pin, level))
        if level == 0 and self.echo_ns is not None:
            rise = self._clock + self.echo_gap_ns
            self._callback(1, rise)
            self._callback(0, rise + self.echo_ns)
            self._clock += 100_000_000

    def close(self):
        pass


def make_sensor(gpio: FakeGpio, **overrides) -> Hcsr04:
    config = UltrasonicConfig(**overrides)
    sensor = Hcsr04(UltrasonicSensorSpec("us-t", 23, 24, Sector.CENTER), gpio, config)
    sensor.start()
    return sensor


class UltrasonicTests(unittest.TestCase):
    def test_echo_time_to_distance(self) -> None:
        # 5.83 ms of echo is a 1.00 m round trip at 343 m/s
        self.assertAlmostEqual(echo_to_distance_m(5_831_000, 343.0), 1.0, places=3)

    def test_measure_converts_echo_to_distance_and_pulses_trigger(self) -> None:
        gpio = FakeGpio(echo_ns=2_915_500)  # 0.5 m
        sample = make_sensor(gpio).measure()
        self.assertTrue(sample.valid)
        self.assertFalse(sample.out_of_range)
        self.assertAlmostEqual(sample.distance_m, 0.5, places=3)
        self.assertEqual(gpio.writes, [(23, 1), (23, 0)])  # trigger goes high then low

    def test_no_echo_is_a_valid_clear_reading_at_max_range(self) -> None:
        sensor = make_sensor(FakeGpio(echo_ns=None))
        sample = sensor.measure()
        self.assertTrue(sample.valid)
        self.assertTrue(sample.out_of_range)
        self.assertEqual(sample.distance_m, 4.0)
        self.assertEqual(sensor.no_echo_streak, 1)

    def test_echo_longer_than_max_range_is_out_of_range(self) -> None:
        sample = make_sensor(FakeGpio(echo_ns=30_000_000)).measure()  # ~5.1 m
        self.assertTrue(sample.out_of_range)
        self.assertEqual(sample.distance_m, 4.0)

    def test_dead_zone_is_clamped_to_minimum_range(self) -> None:
        sample = make_sensor(FakeGpio(echo_ns=20_000)).measure()  # ~3 mm
        self.assertEqual(sample.distance_m, 0.02)
        self.assertTrue(sample.valid)

    def test_measure_before_start_is_an_error(self) -> None:
        sensor = Hcsr04(UltrasonicSensorSpec("u", 23, 24, Sector.CENTER), FakeGpio(), UltrasonicConfig())
        with self.assertRaises(RuntimeError):
            sensor.measure()

    def test_streak_resets_after_a_real_echo(self) -> None:
        gpio = FakeGpio(echo_ns=None)
        sensor = make_sensor(gpio)
        sensor.measure()
        gpio.echo_ns = 2_915_500
        sensor.measure()
        self.assertEqual(sensor.no_echo_streak, 0)
        self.assertEqual(sensor.echo_count, 1)


if __name__ == "__main__":
    unittest.main()
