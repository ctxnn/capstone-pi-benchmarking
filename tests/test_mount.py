import json
import tempfile
import unittest
from pathlib import Path

from yolo_pi.mount import CameraMountProfile, MountProfileError, load_mount_profile


class CameraMountProfileTests(unittest.TestCase):
    def make_profile(self, **overrides):
        payload = {
            "profile_id": "test-top-handle",
            "mount_zone": "near_top_handle",
            "distance_below_handle_mm": 40.0,
            "max_distance_below_handle_mm": 80.0,
            "optical_axis": "forward",
            "nominal_pitch_down_deg": 12.0,
            "pitch_tolerance_deg": 5.0,
            "nominal_roll_deg": 0.0,
            "roll_tolerance_deg": 5.0,
            "hand_occlusion_check_required": True,
            "intrinsic_calibration_required": True,
            "extrinsic_calibration_required": True,
            "hardware_verified": False,
        }
        payload.update(overrides)
        return CameraMountProfile(**payload)

    def test_accepts_top_handle_profile(self):
        profile = self.make_profile()
        profile.validate()
        self.assertEqual(profile.as_dict()["mount_zone"], "near_top_handle")

    def test_rejects_camera_too_far_below_handle(self):
        profile = self.make_profile(distance_below_handle_mm=81.0)
        with self.assertRaises(MountProfileError):
            profile.validate()

    def test_rejects_missing_occlusion_gate(self):
        profile = self.make_profile(hand_occlusion_check_required=False)
        with self.assertRaises(MountProfileError):
            profile.validate()

    def test_loads_json_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mount.json"
            path.write_text(json.dumps(self.make_profile().as_dict()), encoding="utf-8")
            loaded = load_mount_profile(path)
        self.assertEqual(loaded.profile_id, "test-top-handle")


if __name__ == "__main__":
    unittest.main()
