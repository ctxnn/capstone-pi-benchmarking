import tempfile
import unittest
from pathlib import Path

from yolo_pi.detector import infer_backend_name, mnn_runtime_config


class BackendNameTests(unittest.TestCase):
    def test_file_backends(self) -> None:
        self.assertEqual(infer_backend_name("model.pt"), "pytorch-ultralytics")
        self.assertEqual(infer_backend_name("model.onnx"), "onnxruntime-ultralytics")
        self.assertEqual(infer_backend_name("model.xml"), "openvino-ultralytics")
        self.assertEqual(infer_backend_name("model.mnn"), "mnn-high-precision-ultralytics")
        self.assertEqual(infer_backend_name("model.tflite"), "litert-ultralytics")

    def test_mnn_fp32_configuration_is_explicit(self) -> None:
        self.assertEqual(
            mnn_runtime_config(threads=4, precision="FP32"),
            {"precision": "high", "backend": "CPU", "numThread": 4},
        )
        with self.assertRaises(ValueError):
            mnn_runtime_config(threads=4, precision="FP16")

    def test_ncnn_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.ncnn.param").write_text("fixture", encoding="utf-8")
            self.assertEqual(infer_backend_name(path), "ncnn-ultralytics")


if __name__ == "__main__":
    unittest.main()
