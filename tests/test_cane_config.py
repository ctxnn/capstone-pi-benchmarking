import json
import tempfile
import unittest
from pathlib import Path

from yolo_pi.cane.config import CaneConfig, ConfigError, config_from_dict, load_config
from yolo_pi.types import Sector

ROOT = Path(__file__).resolve().parents[1]


class CaneConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        config = load_config()
        self.assertEqual(len(config.tof.sensors), 3)
        self.assertEqual(len(config.ultrasonic.sensors), 2)
        self.assertEqual((config.haptics.taps_left, config.haptics.taps_center, config.haptics.taps_right), (1, 2, 3))

    def test_shipped_hardware_file_loads_and_matches_report_wiring(self) -> None:
        config = load_config(ROOT / "configs" / "cane-hardware.json")
        self.assertEqual(config.i2c.mux_address, 0x70)
        self.assertEqual(config.i2c.tof_address, 0x29)
        self.assertEqual(config.i2c.haptic_address, 0x5A)
        self.assertEqual(config.i2c.imu_address, 0x68)
        pins = {s.sensor_id: (s.trig_gpio, s.echo_gpio) for s in config.ultrasonic.sensors}
        self.assertEqual(pins, {"us-1": (23, 24), "us-2": (27, 22)})
        self.assertEqual(
            [s.sector for s in config.tof.sensors], [Sector.LEFT, Sector.CENTER, Sector.RIGHT]
        )
        self.assertEqual(config.fusion.warning_distance_m, 1.0)
        self.assertEqual(config.fusion.critical_distance_m, 0.5)

    def test_hex_string_addresses_are_accepted(self) -> None:
        config = config_from_dict({"i2c": {"mux_address": "0x71", "imu_address": 105}})
        self.assertEqual(config.i2c.mux_address, 0x71)
        self.assertEqual(config.i2c.imu_address, 0x69)

    def test_unknown_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            config_from_dict({"tof": {"timing_budjet_us": 20000}})
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            config_from_dict({"sensors": []})

    def test_tof_sensors_must_not_share_a_mux_channel(self) -> None:
        payload = {"tof": {"sensors": [
            {"id": "a", "mux_channel": 0, "sector": "left"},
            {"id": "b", "mux_channel": 0, "sector": "right"},
        ]}}
        with self.assertRaisesRegex(ConfigError, "own mux channel"):
            config_from_dict(payload)

    def test_the_single_haptic_driver_may_sit_on_the_main_bus_or_a_mux_channel(self) -> None:
        self.assertIsNone(config_from_dict({}).haptics.driver_mux_channel)
        self.assertEqual(config_from_dict({"haptics": {"driver_mux_channel": 3}}).haptics.driver_mux_channel, 3)
        with self.assertRaisesRegex(ConfigError, "TCA9548A channel"):
            config_from_dict({"haptics": {"driver_mux_channel": 9}})

    def test_a_per_motor_list_is_no_longer_accepted(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            config_from_dict({"haptics": {"motors": []}})

    def test_the_three_directions_must_have_different_tap_counts(self) -> None:
        with self.assertRaisesRegex(ConfigError, "all differ"):
            config_from_dict({"haptics": {"taps_left": 2, "taps_center": 2, "taps_right": 3}})

    def test_tap_groups_must_stay_separable_when_fast(self) -> None:
        with self.assertRaisesRegex(ConfigError, "blur"):
            config_from_dict({"haptics": {"group_gap_fast_ms": 70, "tap_off_fast_ms": 60}})
        with self.assertRaisesRegex(ConfigError, "slower"):
            config_from_dict({"haptics": {"tap_on_fast_ms": 120}})
        with self.assertRaisesRegex(ConfigError, "urgent_distance_m"):
            config_from_dict({"haptics": {"urgent_distance_m": 1.5}})

    def test_i2c_gpio_pins_cannot_be_used_for_ranging(self) -> None:
        payload = {"ultrasonic": {"sensors": [
            {"id": "u", "trig_gpio": 2, "echo_gpio": 24, "sector": "center"}]}}
        with self.assertRaisesRegex(ConfigError, "I2C bus"):
            config_from_dict(payload)

    def test_duplicate_gpio_is_rejected(self) -> None:
        payload = {"ultrasonic": {"sensors": [
            {"id": "u1", "trig_gpio": 23, "echo_gpio": 24, "sector": "center"},
            {"id": "u2", "trig_gpio": 24, "echo_gpio": 22, "sector": "center"}]}}
        with self.assertRaisesRegex(ConfigError, "different"):
            config_from_dict(payload)

    def test_unknown_sector_is_rejected(self) -> None:
        payload = {"tof": {"sensors": [{"id": "t", "mux_channel": 0, "sector": "unknown"}]}}
        with self.assertRaises(ConfigError):
            config_from_dict(payload)

    def test_clear_distance_must_exceed_warning_distance(self) -> None:
        with self.assertRaisesRegex(ConfigError, "clear_distance_m"):
            config_from_dict({"tof": {"clear_distance_m": 0.8}})

    def test_slow_loop_that_would_starve_the_motor_watchdog_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "loop_hz"):
            config_from_dict({"runtime": {"loop_hz": 2}})

    def test_bad_fusion_distances_become_config_errors(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict({"fusion": {"warning_distance_m": 0.4, "critical_distance_m": 0.5}})

    def test_invalid_json_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "not valid JSON"):
                load_config(path)

    def test_describe_is_json_serialisable(self) -> None:
        json.dumps(CaneConfig().describe())


if __name__ == "__main__":
    unittest.main()
