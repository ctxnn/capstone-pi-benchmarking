import math
import unittest

from yolo_pi.imu import (
    GRAVITY_MPS2,
    IMUConfig,
    IMUProcessor,
    IMUSample,
    IMUState,
)

MS = 1_000_000


def accel_vector(pitch_deg: float, roll_deg: float = 0.0, magnitude: float = GRAVITY_MPS2):
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    return (
        -magnitude * math.cos(roll) * math.sin(pitch),
        magnitude * math.sin(roll),
        magnitude * math.cos(roll) * math.cos(pitch),
    )


def sample(
    timestamp_ns: int,
    pitch_deg: float = 12.0,
    roll_deg: float = 0.0,
    gyro=(0.0, 0.0, 0.0),
    magnitude: float = GRAVITY_MPS2,
    valid: bool = True,
) -> IMUSample:
    ax, ay, az = accel_vector(pitch_deg, roll_deg, magnitude)
    return IMUSample(
        timestamp_ns=timestamp_ns,
        ax_mps2=ax,
        ay_mps2=ay,
        az_mps2=az,
        gx_rps=gyro[0],
        gy_rps=gyro[1],
        gz_rps=gyro[2],
        temp_c=31.0,
        valid=valid,
    )


def tilt_down(
    processor: IMUProcessor,
    start_pitch_deg: float,
    end_pitch_deg: float,
    start_ns: int,
    rate_deg_s: float = 140.0,
    step_ms: float = 5.0,
) -> int:
    """Tip the shaft with a gyro rate the accelerometer agrees with."""

    span = abs(end_pitch_deg - start_pitch_deg)
    steps = max(int(round(span / (rate_deg_s * step_ms / 1000.0))), 1)
    sign = 1.0 if end_pitch_deg >= start_pitch_deg else -1.0
    gyro = (0.0, math.radians(rate_deg_s * sign), 0.0)
    timestamp_ns = start_ns
    for step in range(1, steps + 1):
        timestamp_ns = start_ns + int(step * step_ms * MS)
        pitch = start_pitch_deg + sign * min(span, rate_deg_s * step * step_ms / 1000.0)
        processor.update(sample(timestamp_ns, pitch_deg=pitch, gyro=gyro))
    return timestamp_ns


def roll_right(
    processor: IMUProcessor,
    start_roll_deg: float,
    end_roll_deg: float,
    start_ns: int,
    rate_deg_s: float = 120.0,
    step_ms: float = 5.0,
) -> int:
    span = abs(end_roll_deg - start_roll_deg)
    steps = max(int(round(span / (rate_deg_s * step_ms / 1000.0))), 1)
    sign = 1.0 if end_roll_deg >= start_roll_deg else -1.0
    gyro = (math.radians(rate_deg_s * sign), 0.0, 0.0)
    timestamp_ns = start_ns
    for step in range(1, steps + 1):
        timestamp_ns = start_ns + int(step * step_ms * MS)
        roll = start_roll_deg + sign * min(span, rate_deg_s * step * step_ms / 1000.0)
        processor.update(sample(timestamp_ns, pitch_deg=12.0, roll_deg=roll, gyro=gyro))
    return timestamp_ns


class IMUConfigTests(unittest.TestCase):
    def test_defaults_match_the_cane_geometry(self) -> None:
        config = IMUConfig()
        self.assertEqual(config.nominal_pitch_deg, 12.0)
        self.assertEqual(config.complementary_alpha, 0.96)
        self.assertEqual(config.accel_trust_band_mps2, 2.0)
        self.assertEqual(config.sweep_tau_s, 0.6)
        self.assertEqual(config.ground_tilt_warn_deg, 25.0)
        self.assertEqual(config.lifted_pitch_deg, -35.0)
        self.assertEqual(config.extreme_roll_deg, 55.0)
        self.assertEqual(config.sensor_height_m, 0.75)
        self.assertEqual(config.stale_after_ms, 250.0)
        self.assertEqual(config.camera_hfov_deg, 90.0)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            IMUConfig(complementary_alpha=1.5)
        with self.assertRaises(ValueError):
            IMUConfig(sweep_tau_s=0.0)
        with self.assertRaises(ValueError):
            IMUConfig(sensor_height_m=0.0)
        with self.assertRaises(ValueError):
            IMUConfig(camera_hfov_deg=180.0)


class IMUCalibrationTests(unittest.TestCase):
    def test_calibrate_recovers_gyro_bias_and_baseline(self) -> None:
        processor = IMUProcessor()
        bias = (0.02, -0.01, 0.05)
        samples = [
            sample(index * 10 * MS, pitch_deg=12.0, gyro=bias) for index in range(24)
        ]
        measured = processor.calibrate(samples)
        self.assertAlmostEqual(measured[0], 0.02, places=6)
        self.assertAlmostEqual(measured[1], -0.01, places=6)
        self.assertAlmostEqual(measured[2], 0.05, places=6)
        self.assertTrue(processor.calibrated)
        self.assertAlmostEqual(processor.baseline_pitch_deg, 12.0, places=3)

        state = processor.update(sample(300 * MS, pitch_deg=12.0, gyro=bias))
        self.assertTrue(state.calibrated)
        self.assertAlmostEqual(state.pitch_deg, 12.0, places=3)
        self.assertAlmostEqual(state.relative_pitch_deg, 0.0, places=3)
        self.assertEqual(state.gyro_bias_rps, measured)
        self.assertAlmostEqual(state.yaw_rate_rps, 0.0, places=9)

    def test_calibrate_handles_empty_input(self) -> None:
        processor = IMUProcessor()
        self.assertEqual(processor.calibrate([]), (0.0, 0.0, 0.0))
        self.assertFalse(processor.calibrated)
        self.assertFalse(processor.state(0).fresh)

    def test_calibrate_ignores_invalid_samples(self) -> None:
        processor = IMUProcessor()
        samples = [
            sample(index * 10 * MS, pitch_deg=12.0, gyro=(0.01, 0.0, 0.0))
            for index in range(5)
        ]
        samples.append(
            sample(60 * MS, pitch_deg=-70.0, gyro=(9.0, 9.0, 9.0), valid=False)
        )
        measured = processor.calibrate(samples)
        self.assertAlmostEqual(measured[0], 0.01, places=6)
        self.assertAlmostEqual(processor.baseline_pitch_deg, 12.0, places=3)

    def test_set_bias_installs_a_known_attitude(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.1, 0.2, 0.3, baseline_pitch_deg=8.0, baseline_roll_deg=-4.0)
        state = processor.update(sample(0, pitch_deg=8.0, roll_deg=-4.0))
        self.assertTrue(state.calibrated)
        self.assertAlmostEqual(state.pitch_deg, 8.0, delta=0.1)
        self.assertAlmostEqual(state.roll_deg, -4.0, delta=0.1)
        self.assertAlmostEqual(state.relative_pitch_deg, 0.0, delta=0.1)
        self.assertEqual(state.gyro_bias_rps, (0.1, 0.2, 0.3))

    def test_reset_forgets_everything(self) -> None:
        processor = IMUProcessor()
        processor.calibrate([sample(index * 10 * MS) for index in range(10)])
        processor.update(sample(200 * MS))
        processor.reset()
        self.assertFalse(processor.calibrated)
        self.assertEqual(processor.gyro_bias_rps, (0.0, 0.0, 0.0))
        state = processor.state(400 * MS)
        self.assertFalse(state.fresh)
        self.assertFalse(state.calibrated)


class AttitudeFilterTests(unittest.TestCase):
    def test_complementary_filter_tracks_a_pitch_ramp(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        rate_deg_s = 10.0
        gyro = (0.0, math.radians(rate_deg_s), 0.0)
        state = processor.state(0)
        for index in range(1, 101):
            timestamp_ns = index * 10 * MS
            true_pitch = 12.0 + rate_deg_s * index * 0.01
            state = processor.update(sample(timestamp_ns, pitch_deg=true_pitch, gyro=gyro))
        self.assertAlmostEqual(state.pitch_deg, 22.0, places=2)
        self.assertAlmostEqual(state.relative_pitch_deg, 10.0, places=2)

    def test_footstep_shock_does_not_corrupt_pitch(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        processor.update(sample(10 * MS, pitch_deg=12.0))
        before = processor.update(sample(20 * MS, pitch_deg=12.0)).pitch_deg
        shock = IMUSample(30 * MS, -8.0, 0.0, 25.0, 0.0, 0.0, 0.0)
        after = processor.update(shock).pitch_deg
        self.assertAlmostEqual(after, before, places=3)

    def test_pitch_survives_a_burst_of_footstep_shocks(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        for index in range(1, 21):
            magnitude = GRAVITY_MPS2 + (7.0 if index % 2 else -6.0)
            processor.update(
                sample(index * 10 * MS, pitch_deg=-40.0, magnitude=magnitude)
            )
        state = processor.update(sample(210 * MS, pitch_deg=12.0))
        self.assertAlmostEqual(state.pitch_deg, 12.0, delta=1.0)

    def test_invalid_and_non_finite_samples_are_rejected(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        processor.update(sample(10 * MS, pitch_deg=12.0))
        baseline = processor.update(sample(20 * MS, pitch_deg=12.0))
        processor.update(sample(30 * MS, pitch_deg=-60.0, valid=False))
        nan_sample = IMUSample(
            40 * MS, float("nan"), 0.0, GRAVITY_MPS2, 0.0, 0.0, 0.0
        )
        processor.update(nan_sample)
        state = processor.state(40 * MS)
        self.assertAlmostEqual(state.pitch_deg, baseline.pitch_deg, places=6)

    def test_backwards_timestamps_are_ignored(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        forward = processor.update(
            sample(100 * MS, pitch_deg=12.0, gyro=(0.0, math.radians(90.0), 0.0))
        )
        self.assertGreater(forward.pitch_deg, 12.0)
        stale = processor.update(sample(50 * MS, pitch_deg=-45.0))
        self.assertAlmostEqual(stale.pitch_deg, forward.pitch_deg, places=6)
        self.assertEqual(stale.timestamp_ns, forward.timestamp_ns)

    def test_roll_tracks_with_gyro_and_triggers_extreme_roll(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        timestamp_ns = roll_right(processor, 0.0, 60.0, 0)
        state = processor.update(
            sample(timestamp_ns + 10 * MS, pitch_deg=12.0, roll_deg=60.0)
        )
        self.assertAlmostEqual(state.roll_deg, 60.0, delta=1.0)
        self.assertFalse(state.is_pitched_down)
        self.assertTrue(state.is_lifted_or_tilted)

    def test_sweep_yaw_integrates_and_leaks_back_to_zero(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0, gyro=(0.0, 0.0, 0.5)))
        for index in range(1, 61):
            state = processor.update(
                sample(index * 10 * MS, pitch_deg=12.0, gyro=(0.0, 0.0, 0.5))
            )
        expected = 0.5 * 180.0 / math.pi * 0.6 * (1.0 - math.exp(-1.0))
        self.assertAlmostEqual(state.sweep_yaw_deg, expected, delta=0.6)
        self.assertGreater(state.sweep_yaw_deg, 8.0)
        for index in range(61, 121):
            state = processor.update(sample(index * 10 * MS, pitch_deg=12.0))
        self.assertAlmostEqual(state.sweep_yaw_deg, expected * math.exp(-1.0), delta=0.6)
        self.assertAlmostEqual(state.yaw_rate_rps, 0.0, places=9)

    def test_yaw_rate_is_bias_corrected(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.1, baseline_pitch_deg=12.0)
        state = processor.update(sample(0, pitch_deg=12.0, gyro=(0.0, 0.0, 0.1)))
        self.assertAlmostEqual(state.yaw_rate_rps, 0.0, places=9)

    def test_sweep_yaw_leaks_while_stale(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0, gyro=(0.0, 0.0, 0.6)))
        for index in range(1, 31):
            state = processor.update(
                sample(index * 10 * MS, pitch_deg=12.0, gyro=(0.0, 0.0, 0.6))
            )
        integrated = state.sweep_yaw_deg
        stale = processor.state(30 * 10 * MS + 600 * MS)
        self.assertFalse(stale.fresh)
        self.assertLess(stale.sweep_yaw_deg, integrated)


class CaneDetectorTests(unittest.TestCase):
    def test_stationary_requires_the_full_dwell(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0, gyro=(0.001, 0.001, 0.001)))
        for index in range(1, 50):
            state = processor.update(
                sample(index * 10 * MS, pitch_deg=12.0, gyro=(0.001, 0.001, 0.001))
            )
        self.assertFalse(state.is_stationary)
        for index in range(50, 200):
            state = processor.update(
                sample(index * 10 * MS, pitch_deg=12.0, gyro=(0.001, 0.001, 0.001))
            )
        self.assertTrue(state.is_stationary)
        moving = processor.update(
            sample(2000 * MS, pitch_deg=12.0, gyro=(0.5, 0.5, 0.5))
        )
        self.assertFalse(moving.is_stationary)

    def test_noisy_acceleration_breaks_stationarity(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        for index in range(1, 160):
            magnitude = GRAVITY_MPS2 + (1.2 if index % 2 else -1.2)
            state = processor.update(
                sample(index * 10 * MS, pitch_deg=12.0, magnitude=magnitude)
            )
        self.assertFalse(state.is_stationary)

    def test_freefall_then_impact_sets_drop_detected(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        freefall = processor.update(IMUSample(10 * MS, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0))
        self.assertFalse(freefall.drop_detected)
        impact = processor.update(IMUSample(20 * MS, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0))
        self.assertTrue(impact.drop_detected)

    def test_freefall_without_impact_does_not_flag_a_drop(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        processor.update(IMUSample(10 * MS, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0))
        recovered = processor.update(sample(20 * MS, pitch_deg=12.0))
        self.assertFalse(recovered.drop_detected)

    def test_lifted_tip_is_flagged_against_the_baseline(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        timestamp_ns = tilt_down(processor, 12.0, -30.0, 0)
        state = processor.update(sample(timestamp_ns + 10 * MS, pitch_deg=-30.0))
        self.assertAlmostEqual(state.relative_pitch_deg, -42.0, delta=1.0)
        self.assertTrue(state.is_lifted_or_tilted)
        self.assertFalse(state.is_pitched_down)

    def test_shallow_tilt_down_is_not_flagged(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        timestamp_ns = tilt_down(processor, 12.0, 30.0, 0)
        state = processor.update(sample(timestamp_ns + 10 * MS, pitch_deg=30.0))
        self.assertAlmostEqual(state.relative_pitch_deg, 18.0, delta=1.0)
        self.assertFalse(state.is_pitched_down)
        self.assertFalse(state.is_lifted_or_tilted)
        self.assertIsNone(state.expected_ground_distance_m)


class ProjectionAndEchoTests(unittest.TestCase):
    def _pitched_down_state(self) -> IMUState:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        timestamp_ns = tilt_down(processor, 12.0, 40.0, 0)
        for index in range(1, 21):
            state = processor.update(
                sample(timestamp_ns + index * 5 * MS, pitch_deg=40.0)
            )
        return state

    def test_expected_ground_distance_matches_floor_geometry(self) -> None:
        state = self._pitched_down_state()
        self.assertAlmostEqual(state.pitch_deg, 40.0, delta=1.0)
        self.assertAlmostEqual(state.relative_pitch_deg, 28.0, delta=1.0)
        self.assertTrue(state.is_pitched_down)
        expected = 0.75 / math.sin(math.radians(40.0))
        self.assertAlmostEqual(state.expected_ground_distance_m, expected, delta=0.02)

    def test_ground_echo_gating_requires_a_matching_floor_return(self) -> None:
        processor = IMUProcessor()
        state = self._pitched_down_state()
        expected = state.expected_ground_distance_m
        self.assertTrue(processor.is_likely_ground_echo(expected, state))
        self.assertTrue(processor.is_likely_ground_echo(expected * 1.10, state))
        self.assertFalse(processor.is_likely_ground_echo(expected * 0.70, state))
        self.assertFalse(processor.is_likely_ground_echo(expected * 1.40, state))

    def test_ground_echo_respects_the_safety_floor(self) -> None:
        processor = IMUProcessor()
        state = IMUState(
            timestamp_ns=0,
            fresh=True,
            calibrated=True,
            pitch_deg=70.0,
            roll_deg=0.0,
            relative_pitch_deg=58.0,
            yaw_rate_rps=0.0,
            sweep_yaw_deg=0.0,
            is_pitched_down=True,
            expected_ground_distance_m=0.34,
        )
        self.assertFalse(processor.is_likely_ground_echo(0.34, state))
        self.assertTrue(processor.is_likely_ground_echo(0.35, state, min_safety_floor_m=0.50) is False)
        self.assertTrue(processor.is_likely_ground_echo(0.34, state, min_safety_floor_m=0.20))

    def test_ground_echo_requires_fresh_and_pitched_down_state(self) -> None:
        processor = IMUProcessor()
        fresh = IMUState(
            timestamp_ns=0,
            fresh=True,
            calibrated=True,
            pitch_deg=40.0,
            roll_deg=0.0,
            relative_pitch_deg=28.0,
            yaw_rate_rps=0.0,
            sweep_yaw_deg=0.0,
            is_pitched_down=True,
            expected_ground_distance_m=1.16,
        )
        stale = IMUState(
            timestamp_ns=0,
            fresh=False,
            calibrated=True,
            pitch_deg=40.0,
            roll_deg=0.0,
            relative_pitch_deg=28.0,
            yaw_rate_rps=0.0,
            sweep_yaw_deg=0.0,
            is_pitched_down=True,
            expected_ground_distance_m=1.16,
        )
        level = IMUState(
            timestamp_ns=0,
            fresh=True,
            calibrated=True,
            pitch_deg=12.0,
            roll_deg=0.0,
            relative_pitch_deg=0.0,
            yaw_rate_rps=0.0,
            sweep_yaw_deg=0.0,
            is_pitched_down=False,
            expected_ground_distance_m=None,
        )
        self.assertTrue(processor.is_likely_ground_echo(1.16, fresh))
        self.assertFalse(processor.is_likely_ground_echo(1.16, stale))
        self.assertFalse(processor.is_likely_ground_echo(1.16, level))

    def test_project_horizontal_distance_clamps_the_tilt_angle(self) -> None:
        processor = IMUProcessor()

        def state_with(relative_pitch_deg: float) -> IMUState:
            return IMUState(
                timestamp_ns=0,
                fresh=True,
                calibrated=True,
                pitch_deg=12.0 + relative_pitch_deg,
                roll_deg=0.0,
                relative_pitch_deg=relative_pitch_deg,
                yaw_rate_rps=0.0,
                sweep_yaw_deg=0.0,
            )

        self.assertAlmostEqual(
            processor.project_horizontal_distance(1.0, state_with(0.0)), 1.0, places=6
        )
        self.assertAlmostEqual(
            processor.project_horizontal_distance(1.0, state_with(30.0)),
            math.cos(math.radians(30.0)),
            places=6,
        )
        self.assertAlmostEqual(
            processor.project_horizontal_distance(1.0, state_with(60.0)),
            math.cos(math.radians(30.0)),
            places=6,
        )
        self.assertAlmostEqual(
            processor.project_horizontal_distance(1.0, state_with(-45.0)),
            math.cos(math.radians(-30.0)),
            places=6,
        )

    def test_project_horizontal_distance_rejects_invalid_input(self) -> None:
        processor = IMUProcessor()
        state = processor.state(0)
        self.assertEqual(processor.project_horizontal_distance(0.0, state), 0.0)
        self.assertEqual(processor.project_horizontal_distance(-1.0, state), 0.0)
        self.assertEqual(
            processor.project_horizontal_distance(float("nan"), state), 0.0
        )
        self.assertAlmostEqual(
            processor.project_horizontal_distance(1.5, state), 1.5, places=6
        )


class VisionCompensationTests(unittest.TestCase):
    def _yawing_state(self, yaw_rate_rps: float) -> IMUState:
        processor = IMUProcessor()
        processor.set_bias(0.0, 0.0, 0.0, baseline_pitch_deg=12.0)
        processor.update(sample(0, pitch_deg=12.0))
        processor.update(
            sample(
                10 * MS,
                pitch_deg=12.0,
                gyro=(0.0, 0.0, yaw_rate_rps),
            )
        )
        return processor.state(10 * MS)

    def test_center_shift_scales_with_yaw_rate_and_age(self) -> None:
        processor = IMUProcessor()
        state = self._yawing_state(1.0)
        expected = 320.0 + 0.1 * (640.0 / math.radians(90.0))
        self.assertAlmostEqual(
            processor.compensate_vision_center_x(320.0, 640, 0.1, state),
            expected,
            places=6,
        )

    def test_center_shift_is_clamped_to_the_frame(self) -> None:
        processor = IMUProcessor()
        state = self._yawing_state(1.0)
        self.assertEqual(
            processor.compensate_vision_center_x(639.0, 640, 1.0, state), 640.0
        )
        left = self._yawing_state(-1.0)
        self.assertEqual(
            processor.compensate_vision_center_x(1.0, 640, 1.0, left), 0.0
        )

    def test_center_shift_is_identity_without_a_fresh_state(self) -> None:
        processor = IMUProcessor()
        neutral = processor.state(0)
        self.assertFalse(neutral.fresh)
        self.assertAlmostEqual(
            processor.compensate_vision_center_x(300.0, 640, 0.2, neutral),
            300.0,
            places=6,
        )

    def test_center_shift_rejects_a_bad_frame_width(self) -> None:
        processor = IMUProcessor()
        state = self._yawing_state(1.0)
        with self.assertRaises(ValueError):
            processor.compensate_vision_center_x(100.0, 0, 0.1, state)


class StateFallbackTests(unittest.TestCase):
    def test_state_is_neutral_before_the_first_update(self) -> None:
        processor = IMUProcessor()
        state = processor.state(5 * MS)
        self.assertFalse(state.fresh)
        self.assertFalse(state.calibrated)
        self.assertEqual(state.pitch_deg, 0.0)
        self.assertEqual(state.roll_deg, 0.0)
        self.assertEqual(state.relative_pitch_deg, 0.0)
        self.assertEqual(state.yaw_rate_rps, 0.0)
        self.assertFalse(state.is_stationary)
        self.assertFalse(state.is_pitched_down)
        self.assertIsNone(state.expected_ground_distance_m)
        self.assertEqual(state.gyro_bias_rps, (0.0, 0.0, 0.0))

    def test_state_goes_stale_after_the_configured_window(self) -> None:
        processor = IMUProcessor()
        processor.calibrate([sample(index * 10 * MS) for index in range(10)])
        processor.update(sample(100 * MS, pitch_deg=12.0))
        fresh = processor.state(100 * MS + 200 * MS)
        self.assertTrue(fresh.fresh)
        self.assertAlmostEqual(fresh.pitch_deg, 12.0, places=3)
        stale = processor.state(100 * MS + 300 * MS)
        self.assertFalse(stale.fresh)
        self.assertTrue(stale.calibrated)
        self.assertEqual(stale.pitch_deg, 0.0)
        self.assertEqual(stale.relative_pitch_deg, 0.0)
        self.assertEqual(stale.yaw_rate_rps, 0.0)

    def test_as_dict_exposes_every_field(self) -> None:
        processor = IMUProcessor()
        processor.set_bias(0.01, 0.02, 0.03, baseline_pitch_deg=12.0)
        state = processor.update(sample(0, pitch_deg=12.0))
        payload = state.as_dict()
        for key in (
            "timestamp_ns",
            "fresh",
            "calibrated",
            "pitch_deg",
            "roll_deg",
            "relative_pitch_deg",
            "yaw_rate_rps",
            "sweep_yaw_deg",
            "is_stationary",
            "is_lifted_or_tilted",
            "is_pitched_down",
            "expected_ground_distance_m",
            "drop_detected",
            "gyro_bias_rps",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["gyro_bias_rps"], [0.01, 0.02, 0.03])
        self.assertTrue(payload["fresh"])


if __name__ == "__main__":
    unittest.main()
