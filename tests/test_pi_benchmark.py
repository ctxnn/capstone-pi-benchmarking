import csv
import json
import tempfile
import unittest
from pathlib import Path

from yolo_pi.pi_benchmark import compile_tables, tree_sha256


class PiBenchmarkTableTests(unittest.TestCase):
    def test_tree_hash_ignores_runtime_cache_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.ncnn.param").write_text("model", encoding="utf-8")
            original = tree_sha256(root)
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "model.cpython-313.pyc").write_bytes(b"host-specific")
            (root / ".DS_Store").write_bytes(b"finder")
            self.assertEqual(tree_sha256(root), original)

    def test_compiles_exact_rows_and_preserves_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = {
                "architecture_experiments": [
                    {"id": "M1", "model": "Cane", "model_key": "cane", "runtime": "PyTorch", "imgsz": 640, "precision": "FP32"}
                ],
                "runtime_experiments": [
                    {"id": "R1", "model": "Cane", "model_key": "cane", "runtime": "PyTorch", "imgsz": 640, "precision": "FP32"},
                    {"id": "R2", "model": "Cane", "model_key": "missing", "runtime": "ONNX Runtime", "imgsz": 640, "precision": "FP32"},
                ],
            }
            registry = {
                "artifacts": {
                    "cane": {"quality_metrics": {"map50": 0.5, "map50_95": 0.25, "recall": 0.75}},
                    "missing": {},
                }
            }
            matrix_path = root / "matrix.json"
            registry_path = root / "registry.json"
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            rows = root / "runs" / "rows"
            rows.mkdir(parents=True)
            summary = lambda mean: {"mean": mean, "p50": mean, "p95": mean + 1, "p99": mean + 2, "max": mean + 3}
            report = {
                "status": "complete",
                "metrics": {
                    "preprocess_ms": summary(1.0),
                    "inference_ms": summary(10.0),
                    "postprocess_ms": summary(2.0),
                    "wall_total_ms": summary(14.0),
                    "system_cpu_percent": summary(60.0),
                    "rss_mb": summary(400.0),
                    "temperature_c": summary(65.0),
                    "capture_ms": summary(5.0),
                    "frame_age_at_model_start_ms": summary(7.0),
                    "camera_to_model_finish_ms": summary(21.0),
                },
                "fps_from_wall_mean": 71.428,
                "power_w": None,
            }
            (rows / "M1.json").write_text(json.dumps(report), encoding="utf-8")
            camera_report = dict(report)
            camera_report["row"] = {
                **matrix["runtime_experiments"][0],
                "source": "camera",
                "threads": 3,
            }
            (rows / "R1.json").write_text(json.dumps(camera_report), encoding="utf-8")

            output = root / "tables"
            compile_tables(matrix_path, registry_path, root / "runs", output)
            architecture = list(
                csv.reader(
                    (output / "model-architecture-benchmark.csv")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            )
            runtime = list(
                csv.reader(
                    (output / "runtime-backend-benchmark.csv")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            )
            self.assertEqual(architecture[1][0], "M1")
            self.assertEqual(float(architecture[1][15]), 50.0)
            self.assertEqual(runtime[1][0], "R1")
            self.assertEqual(runtime[1][5], "3")
            self.assertEqual(runtime[1][6], "camera")
            self.assertEqual(float(runtime[1][7]), 5.0)
            self.assertEqual(runtime[2][-1], "pending")
            self.assertIn("NA", (output / "runtime-backend-benchmark.md").read_text())


if __name__ == "__main__":
    unittest.main()
