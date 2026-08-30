import hashlib
import json
import unittest
from pathlib import Path


class EvaluationManifestTests(unittest.TestCase):
    def test_manifest_is_traceable_and_checksum_is_well_formed(self) -> None:
        path = Path("data/evaluation/manifest.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertTrue(payload["inputs"])
        for item in payload["inputs"]:
            self.assertEqual(len(item["sha256"]), 64)
            int(item["sha256"], 16)
            self.assertTrue(item["url"].startswith("https://"))
            self.assertTrue(item["expected_visible_classes"])


if __name__ == "__main__":
    unittest.main()
