import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from yolo_pi.openvino_detector import (
    OpenVINODetector,
    decode_yolo_output,
    letterbox_preprocess,
    nms_boxes,
)

MODEL_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "models" / "cane-v1-fast"
MODEL_416 = MODEL_ROOT / "ov-fp32-416" / "best_openvino_model"
MODEL_320 = MODEL_ROOT / "ov-fp32-320" / "best_openvino_model"

CLASS_NAMES = {0: "person", 1: "vehicle", 2: "pole"}


def synthetic_output(
    box_rows,
    score_rows,
    transposed: bool = False,
) -> np.ndarray:
    """Build a raw head tensor from ``(N, 4)`` boxes and ``(nc, N)`` scores."""

    boxes = np.asarray(box_rows, dtype=np.float32).T
    scores = np.asarray(score_rows, dtype=np.float32)
    if boxes.shape[1] != scores.shape[1]:
        raise ValueError("box and score rows must describe the same detections")
    tensor = np.concatenate([boxes, scores])[None]
    return tensor.transpose(0, 2, 1) if transposed else tensor


class LetterboxPreprocessTests(unittest.TestCase):
    def test_geometry_and_padding(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        batch, scale, pad_x, pad_y, orig_h, orig_w = letterbox_preprocess(image, (416, 416))

        self.assertEqual(batch.shape, (1, 3, 416, 416))
        self.assertEqual(batch.dtype, np.float32)
        self.assertTrue(batch.flags["C_CONTIGUOUS"])
        self.assertAlmostEqual(scale, 0.65)
        self.assertAlmostEqual(pad_x, 0.0)
        self.assertAlmostEqual(pad_y, 52.0)
        self.assertEqual((orig_h, orig_w), (480, 640))

        padded = batch[0, 0]
        self.assertAlmostEqual(float(padded[0, 0]), 114 / 255.0, places=6)
        self.assertAlmostEqual(float(padded[51, 200]), 114 / 255.0, places=6)
        self.assertAlmostEqual(float(padded[52, 200]), 0.0, places=6)

    def test_transpose_and_value_scaling(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[..., 0] = 255
        batch, _, _, _, _, _ = letterbox_preprocess(image, (100, 100))

        self.assertAlmostEqual(float(batch[0, 0].max()), 1.0, places=6)
        self.assertEqual(float(batch[0, 1].max()), 0.0)
        self.assertEqual(float(batch[0, 2].max()), 0.0)

    def test_marker_maps_back_through_inverse(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[100:200, 400:500] = 255
        batch, scale, pad_x, pad_y, _, _ = letterbox_preprocess(image, (416, 416))

        column = batch[0, 0]
        rows, cols = np.nonzero(column > 0.5)
        x1 = (cols.min() - pad_x) / scale
        y1 = (rows.min() - pad_y) / scale
        x2 = (cols.max() + 1 - pad_x) / scale
        y2 = (rows.max() + 1 - pad_y) / scale
        self.assertAlmostEqual(x1, 400.0, delta=2.0)
        self.assertAlmostEqual(y1, 100.0, delta=2.0)
        self.assertAlmostEqual(x2, 500.0, delta=2.0)
        self.assertAlmostEqual(y2, 200.0, delta=2.0)

    def test_out_buffer_is_reused(self) -> None:
        buffer = np.zeros((416, 416, 3), dtype=np.uint8)
        image = np.full((480, 640, 3), 7, dtype=np.uint8)
        batch, _, _, _, _, _ = letterbox_preprocess(image, (416, 416), out_buffer=buffer)
        self.assertEqual(batch.shape, (1, 3, 416, 416))
        self.assertTrue(np.all(buffer[0] == 114))

    def test_rejects_bad_shapes(self) -> None:
        with self.assertRaises(ValueError):
            letterbox_preprocess(np.zeros((480, 640), dtype=np.uint8), (416, 416))


class NmsBoxesTests(unittest.TestCase):
    def test_suppresses_overlapping_boxes(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [50.0, 50.0, 60.0, 60.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        self.assertEqual(nms_boxes(boxes, scores, 0.7).tolist(), [0, 2])

    def test_keeps_disjoint_boxes_sorted_by_score(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [100.0, 100.0, 110.0, 110.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.2, 0.95], dtype=np.float32)
        self.assertEqual(nms_boxes(boxes, scores, 0.7).tolist(), [1, 0])

    def test_empty_input(self) -> None:
        result = nms_boxes(np.empty((0, 4)), np.empty((0,)), 0.7)
        self.assertEqual(result.size, 0)


class DecodeYoloOutputTests(unittest.TestCase):
    SCALE = 0.5
    PAD_X = 10.0
    PAD_Y = 20.0
    ORIG_W = 640
    ORIG_H = 480

    def decode(self, raw_output: np.ndarray, **kwargs):
        options = {
            "class_names": CLASS_NAMES,
            "scale": self.SCALE,
            "pad_x": self.PAD_X,
            "pad_y": self.PAD_Y,
            "orig_w": self.ORIG_W,
            "orig_h": self.ORIG_H,
            "confidence": 0.35,
            "iou_threshold": 0.70,
        }
        options.update(kwargs)
        return decode_yolo_output(raw_output, **options)

    def test_unletterboxes_coordinates(self) -> None:
        raw = synthetic_output(
            [[100.0, 120.0, 40.0, 60.0]],
            [[0.9], [0.05], [0.05]],
        )
        detections = self.decode(raw)
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.class_id, 0)
        self.assertEqual(detection.class_name, "person")
        self.assertAlmostEqual(detection.box.x1, (100 - 20 - 10) / 0.5, places=4)
        self.assertAlmostEqual(detection.box.y1, (120 - 30 - 20) / 0.5, places=4)
        self.assertAlmostEqual(detection.box.x2, (100 + 20 - 10) / 0.5, places=4)
        self.assertAlmostEqual(detection.box.y2, (120 + 30 - 20) / 0.5, places=4)

    def test_same_class_nms_and_different_class_keep(self) -> None:
        raw = synthetic_output(
            [
                [100.0, 120.0, 40.0, 60.0],
                [101.0, 121.0, 40.0, 60.0],
                [400.0, 300.0, 40.0, 60.0],
            ],
            [[0.9, 0.8, 0.0], [0.0, 0.0, 0.7], [0.1, 0.2, 0.1]],
        )
        detections = self.decode(raw)
        self.assertEqual(len(detections), 2)
        self.assertEqual([d.class_id for d in detections], [0, 1])
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=5)
        self.assertGreaterEqual(detections[0].confidence, detections[1].confidence)

    def test_per_class_thresholds(self) -> None:
        raw = synthetic_output(
            [[100.0, 120.0, 40.0, 60.0], [400.0, 300.0, 40.0, 60.0]],
            [[0.5, 0.5], [0.9, 0.5], [0.0, 0.0]],
        )
        self.assertEqual(len(self.decode(raw)), 2)
        filtered = self.decode(raw, class_thresholds={"person": 0.6})
        self.assertEqual([d.class_name for d in filtered], ["vehicle"])

    def test_class_id_filter(self) -> None:
        raw = synthetic_output(
            [[100.0, 120.0, 40.0, 60.0], [400.0, 300.0, 40.0, 60.0]],
            [[0.9, 0.0], [0.0, 0.0], [0.0, 0.9]],
        )
        self.assertEqual([d.class_id for d in self.decode(raw, class_ids=[2])], [2])

    def test_clamps_to_frame(self) -> None:
        raw = synthetic_output(
            [[20.0, 20.0, 200.0, 200.0]],
            [[0.9], [0.0], [0.0]],
        )
        detection = self.decode(raw)[0]
        self.assertGreaterEqual(detection.box.x1, 0.0)
        self.assertGreaterEqual(detection.box.y1, 0.0)
        self.assertLessEqual(detection.box.x2, float(self.ORIG_W))
        self.assertLessEqual(detection.box.y2, float(self.ORIG_H))

    def test_transposed_layout_matches(self) -> None:
        channels_first = synthetic_output(
            [[100.0, 120.0, 40.0, 60.0]],
            [[0.9], [0.05], [0.05]],
        )
        class_first = self.decode(channels_first)
        channels_last = self.decode(channels_first.transpose(0, 2, 1))
        self.assertEqual(len(class_first), len(channels_last))
        self.assertAlmostEqual(class_first[0].box.x1, channels_last[0].box.x1, places=4)

    def test_empty_when_below_threshold(self) -> None:
        raw = synthetic_output([[100.0, 120.0, 40.0, 60.0]], [[0.1], [0.1], [0.1]])
        self.assertEqual(self.decode(raw), [])

    def test_max_detections(self) -> None:
        rows = [[10.0 + index * 100.0, 100.0, 20.0, 20.0] for index in range(6)]
        scores = [[0.9 - index * 0.05 for index in range(6)], [0.0] * 6, [0.0] * 6]
        detections = self.decode(synthetic_output(rows, scores), max_detections=3)
        self.assertEqual(len(detections), 3)


class RealModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frame[:, :, 0] = np.linspace(0, 60, 640, dtype=np.uint8)[None, :]
        self.frame[150:330, 260:380] = (180, 40, 40)

    def check_detector(self, model_path: Path, expected_side: int) -> None:
        self.assertTrue(
            (model_path / "best.xml").exists(), f"missing model at {model_path}"
        )
        detector = OpenVINODetector(model_path, num_threads=2)
        self.assertEqual(detector.backend_name, "openvino-native")
        self.assertEqual(detector.input_size, (expected_side, expected_side))
        self.assertEqual(detector.class_names[0], "person")

        for frame in (self.frame, Image.fromarray(self.frame)):
            result = detector.detect(frame, frame_id=7, source="unit-test")
            self.assertEqual(result.backend, "openvino-native")
            self.assertEqual((result.width, result.height), (640, 480))
            self.assertEqual(result.frame_id, 7)
            self.assertEqual(result.source, "unit-test")
            self.assertEqual(
                set(result.stage_ms), {"preprocess", "inference", "postprocess"}
            )
            for value in result.stage_ms.values():
                self.assertGreaterEqual(value, 0.0)
            self.assertGreater(result.model_call_ms, 0.0)
            for detection in result.detections:
                self.assertGreaterEqual(detection.box.x1, 0.0)
                self.assertLessEqual(detection.box.x2, 640.0)
                self.assertGreaterEqual(detection.box.y1, 0.0)
                self.assertLessEqual(detection.box.y2, 480.0)

    def test_fp32_416_model(self) -> None:
        self.check_detector(MODEL_416, 416)

    def test_fp32_320_model(self) -> None:
        self.check_detector(MODEL_320, 320)

    def test_warmup_and_benchmark(self) -> None:
        detector = OpenVINODetector(MODEL_320, num_threads=2)
        detector.warmup(runs=1)
        stats = detector.benchmark_frame(self.frame, warmup=1, runs=3)
        for key in ("preprocess", "inference", "postprocess", "total", "fps"):
            self.assertIn(key, stats)
            self.assertGreater(stats[key], 0.0)

    def test_rejects_mismatched_imgsz(self) -> None:
        with self.assertRaises(ValueError):
            OpenVINODetector(MODEL_416, imgsz=320, num_threads=2)
        matching = OpenVINODetector(MODEL_416, imgsz=416, num_threads=2)
        self.assertEqual(matching.input_size, (416, 416))

    def test_no_torch_or_ultralytics_imported(self) -> None:
        detector = OpenVINODetector(MODEL_320, num_threads=2)
        detector.detect(self.frame, frame_id=0)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("ultralytics", sys.modules)

    def test_detects_objects_on_a_real_image(self) -> None:
        fixture = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "presentation"
            / "cane-v1-showcase"
            / "all-annotated"
            / "fixture-005.jpg"
        )
        if not fixture.exists():
            self.skipTest("showcase fixture not present")
        detector = OpenVINODetector(MODEL_416, confidence=0.2, num_threads=2)
        result = detector.detect(fixture, frame_id=0)
        self.assertGreater(len(result.detections), 0)
        self.assertTrue(all(d.class_name for d in result.detections))


if __name__ == "__main__":
    unittest.main()
