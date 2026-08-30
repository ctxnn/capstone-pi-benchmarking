import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_export_smoke.py"


class ExportSmokeEvidenceTests(unittest.TestCase):
    def test_merges_only_checksum_matched_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            evidence_path = root / "evidence.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "imgsz": 640,
                        "precision": "FP32",
                        "exports": {
                            "openvino": {
                                "status": "smoke_failed",
                                "tree_sha256": "artifact-tree",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            evidence_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "format": "openvino",
                        "imgsz": 640,
                        "precision": "FP32",
                        "artifact": {"tree_sha256": "artifact-tree"},
                        "environment": {"platform": "linux"},
                        "smoke": {
                            "status": "complete",
                            "backend": "openvino-ultralytics",
                            "detections": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--manifest",
                    str(manifest_path),
                    "--format",
                    "openvino",
                    "--evidence",
                    str(evidence_path),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            merged = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = merged["exports"]["openvino"]
            self.assertEqual(entry["status"], "complete")
            self.assertEqual(entry["smoke"]["validation_environment"]["platform"], "linux")
            self.assertEqual(len(entry["smoke"]["evidence_sha256"]), 64)

    def test_rejects_mismatched_artifact_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            evidence_path = root / "evidence.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "imgsz": 640,
                        "precision": "FP32",
                        "exports": {"openvino": {"tree_sha256": "expected"}},
                    }
                ),
                encoding="utf-8",
            )
            evidence_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "format": "openvino",
                        "imgsz": 640,
                        "precision": "FP32",
                        "artifact": {"tree_sha256": "different"},
                        "environment": {},
                        "smoke": {"backend": "openvino-ultralytics"},
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--manifest",
                    str(manifest_path),
                    "--format",
                    "openvino",
                    "--evidence",
                    str(evidence_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
