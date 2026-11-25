import struct
import unittest

from yolo_pi.cane.config import ImuConfig
from yolo_pi.cane.imu import (
    ImuSample,
    MotionMonitor,
    Mpu6050,
    calibrate_gyro,
    decode_sample,
)

SECOND = 1_000_000_000


def burst(accel=(0, 0, 8192), temp=0, gyro=(0, 0, 0)) -> bytes:
    return struct.pack(">hhhhhhh", *accel, temp, *gyro)


class FakeRegs:
    address = 0x68

    def __init__(self, frames=None, who=0x68) -> None:
        self.writes = []
        self.frames = list(frames or [])
        self.who = who
        self._last = burst()

    def write_reg(self, reg, value):
        self.writes.append((reg, value))

    def read_reg(self, reg):
        return self.who

    def read_regs(self, reg, length):
        if self.frames:
            self._last = self.frames.pop(0)
        return self._last


class DecodeTests(unittest.TestCase):
    def test_gravity_and_rotation_are_scaled(self) -> None:
        sample = decode_sample(burst(accel=(0, 0, 8192), temp=0, gyro=(655, 0, -655)))
        self.assertAlmostEqual(sample.accel_g[2], 1.0)
        self.assertAlmostEqual(sample.gyro_dps[0], 10.0, places=2)
        self.assertAlmostEqual(sample.gyro_dps[2], -10.0, places=2)
        self.assertAlmostEqual(sample.temperature_c, 36.53, places=2)

    def test_negative_values_decode_as_signed(self) -> None:
        self.assertAlmostEqual(decode_sample(burst(accel=(-8192, 0, 0))).accel_g[0], -1.0)

    def test_wrong_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decode_sample(b"\x00" * 13)


class DriverTests(unittest.TestCase):
    def test_configure_wakes_and_sets_ranges(self) -> None:
        regs = FakeRegs()
        Mpu6050(regs, sleep=lambda _s: None).configure()
        written = dict(regs.writes)
        self.assertEqual(written[0x1B], 0x08)  # gyro +/-500 dps
        self.assertEqual(written[0x1C], 0x08)  # accel +/-4 g
        self.assertEqual(written[0x6B], 0x01)  # awake, PLL clock
        self.assertLess(regs.writes.index((0x6B, 0x80)), regs.writes.index((0x6B, 0x01)))


class CalibrationTests(unittest.TestCase):
    def test_bias_is_the_mean_of_a_still_run(self) -> None:
        regs = FakeRegs([burst(gyro=(131, -65, 0))] * 20)  # 2.0 and -1.0 dps
        imu = Mpu6050(regs, sleep=lambda _s: None)
        result = calibrate_gyro(imu, ImuConfig(calibration_samples=20), sleep=lambda _s: None)
        self.assertTrue(result.calibrated)
        self.assertAlmostEqual(result.bias_dps[0], 2.0, places=1)
        self.assertAlmostEqual(result.bias_dps[1], -1.0, places=1)

    def test_a_moving_cane_is_rejected_and_bias_stays_zero(self) -> None:
        moving = [burst(gyro=(6550, 0, 0))] * 40  # 100 dps
        regs = FakeRegs(moving)
        imu = Mpu6050(regs, sleep=lambda _s: None)
        config = ImuConfig(calibration_samples=10, calibration_attempts=3)
        result = calibrate_gyro(imu, config, sleep=lambda _s: None)
        self.assertFalse(result.calibrated)
        self.assertEqual(result.bias_dps, (0.0, 0.0, 0.0))

    def test_a_noisy_first_run_is_retried(self) -> None:
        noisy = [burst(gyro=(6550, 0, 0))] * 10
        still = [burst()] * 10
        regs = FakeRegs(noisy + still)
        imu = Mpu6050(regs, sleep=lambda _s: None)
        result = calibrate_gyro(imu, ImuConfig(calibration_samples=10), sleep=lambda _s: None)
        self.assertTrue(result.calibrated)
        self.assertEqual(result.attempts, 2)


class MotionTests(unittest.TestCase):
    def still(self) -> ImuSample:
        return ImuSample((0.0, 0.0, 1.0), (0.2, 0.0, 0.0), 30.0)

    def test_still_only_after_the_hold_time(self) -> None:
        monitor = MotionMonitor(ImuConfig(still_after_s=3.0))
        self.assertFalse(monitor.update(self.still(), 0))
        self.assertFalse(monitor.update(self.still(), 2 * SECOND))
        self.assertTrue(monitor.update(self.still(), 3 * SECOND))

    def test_any_movement_resets_the_timer(self) -> None:
        monitor = MotionMonitor(ImuConfig(still_after_s=1.0))
        monitor.update(self.still(), 0)
        self.assertTrue(monitor.update(self.still(), SECOND))
        swing = ImuSample((0.0, 0.0, 1.0), (50.0, 0.0, 0.0), 30.0)
        self.assertFalse(monitor.update(swing, SECOND + 1))
        self.assertFalse(monitor.update(self.still(), SECOND + 2))
        self.assertFalse(monitor.is_still)

    def test_bias_is_removed_before_the_stillness_test(self) -> None:
        monitor = MotionMonitor(ImuConfig(still_after_s=0.0), bias_dps=(5.0, 0.0, 0.0))
        biased = ImuSample((0.0, 0.0, 1.0), (5.2, 0.0, 0.0), 30.0)
        self.assertTrue(monitor.update(biased, 0))

    def test_acceleration_away_from_one_g_counts_as_motion(self) -> None:
        monitor = MotionMonitor(ImuConfig(still_after_s=0.0))
        shaken = ImuSample((0.0, 0.0, 1.5), (0.0, 0.0, 0.0), 30.0)
        self.assertFalse(monitor.update(shaken, 0))


if __name__ == "__main__":
    unittest.main()
