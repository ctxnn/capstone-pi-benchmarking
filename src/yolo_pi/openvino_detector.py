"""Native OpenVINO YOLO runtime with no torch or ultralytics dependency.

The exported OpenVINO IR produced by Ultralytics already carries the
post-sigmoid class probabilities, so decoding only needs letterbox geometry
inversion, per-class thresholding, and class-aware NMS.
"""

from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from .sectors import assign_sectors
from .types import BoundingBox, Detection, FrameDetections

BACKEND_NAME = "openvino-native"
_MODEL_FILES = ("best.xml", "model.xml")


def _resize_bilinear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an HWC RGB uint8 image, preferring OpenCV when present."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        from PIL import Image

        return np.asarray(
            Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
        )
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def letterbox_preprocess(
    image_rgb: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 114,
    out_buffer: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, float, int, int]:
    """Letterbox an HWC image into a contiguous NCHW float32 batch.

    Returns the batch tensor, the uniform scale applied to the source, the
    symmetric padding offsets, and the source height and width.
    """

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    target_h, target_w = int(target_size[0]), int(target_size[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_size must be positive")

    orig_h, orig_w = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    scale = min(target_h / orig_h, target_w / orig_w)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0
    left = int(pad_x)
    top = int(pad_y)

    if out_buffer is None:
        canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    else:
        canvas = out_buffer
        canvas[...] = pad_value
    canvas[top : top + new_h, left : left + new_w] = _resize_bilinear(
        image_rgb, new_w, new_h
    )

    batch = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
    batch /= 255.0
    return batch, float(scale), float(pad_x), float(pad_y), orig_h, orig_w


def nms_boxes(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Return kept indices of an xyxy box set, highest score first."""

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    keep: List[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        inter_w = np.minimum(x2[current], x2[rest]) - np.maximum(x1[current], x1[rest])
        inter_h = np.minimum(y2[current], y2[rest]) - np.maximum(y1[current], y1[rest])
        intersection = np.maximum(inter_w, 0.0) * np.maximum(inter_h, 0.0)
        union = areas[current] + areas[rest] - intersection
        iou = np.where(union > 0.0, intersection / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _as_channel_first(raw_output: np.ndarray) -> np.ndarray:
    """Normalize a raw head output to a ``(4 + nc, N)`` float array."""

    output = np.asarray(raw_output, dtype=np.float32)
    if output.ndim == 3:
        if output.shape[0] != 1:
            raise ValueError("expected a batch of exactly one sample")
        middle, last = int(output.shape[1]), int(output.shape[2])
        channel_first = middle >= 5 and (last < 5 or middle <= last)
        output = output[0] if channel_first else output[0].T
    elif output.ndim != 2:
        raise ValueError(f"unsupported output rank: {output.ndim}")
    if output.shape[0] < 5:
        raise ValueError("output must carry four box rows and at least one class row")
    return output


def decode_yolo_output(
    raw_output: np.ndarray,
    class_names: Dict[int, str],
    scale: float,
    pad_x: float,
    pad_y: float,
    orig_w: int,
    orig_h: int,
    confidence: float = 0.35,
    iou_threshold: float = 0.70,
    class_thresholds: Optional[Dict[str, float]] = None,
    class_ids: Optional[Sequence[int]] = None,
    max_detections: int = 100,
) -> List[Detection]:
    """Decode raw YOLO head output into detections in original-frame pixels."""

    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if max_detections <= 0:
        return []

    output = _as_channel_first(raw_output)
    class_scores = output[4:]
    if class_scores.size == 0:
        return []

    scores = class_scores.max(axis=0)
    class_index = class_scores.argmax(axis=0)
    if class_thresholds:
        thresholds = np.full(scores.shape, confidence, dtype=np.float32)
        for name, value in class_thresholds.items():
            for class_id, class_name in class_names.items():
                if class_name == name:
                    thresholds[class_index == int(class_id)] = float(value)
        keep = scores >= thresholds
    else:
        keep = scores >= confidence
    if class_ids is not None:
        allowed = np.asarray(list(class_ids), dtype=class_index.dtype)
        keep &= np.isin(class_index, allowed)
    if not keep.any():
        return []

    scores = scores[keep]
    class_index = class_index[keep]
    boxes = output[:4, keep].T.astype(np.float32, copy=True)
    cx, cy, width, height = (boxes[:, index].copy() for index in range(4))
    boxes[:, 0] = (cx - width / 2.0 - pad_x) / scale
    boxes[:, 1] = (cy - height / 2.0 - pad_y) / scale
    boxes[:, 2] = (cx + width / 2.0 - pad_x) / scale
    boxes[:, 3] = (cy + height / 2.0 - pad_y) / scale
    np.clip(boxes[:, 0::2], 0.0, float(orig_w), out=boxes[:, 0::2])
    np.clip(boxes[:, 1::2], 0.0, float(orig_h), out=boxes[:, 1::2])

    order = scores.argsort()[::-1]
    boxes, scores, class_index = boxes[order], scores[order], class_index[order]

    kept: List[int] = []
    for class_id in np.unique(class_index):
        members = np.flatnonzero(class_index == class_id)
        kept.extend(
            members[nms_boxes(boxes[members], scores[members], iou_threshold)].tolist()
        )
    ranked = sorted(kept, key=lambda index: -float(scores[index]))[:max_detections]
    if not ranked:
        return []

    detections: List[Detection] = []
    for index in ranked:
        class_id = int(class_index[index])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_names.get(class_id, str(class_id)),
                confidence=float(min(max(scores[index], 0.0), 1.0)),
                box=BoundingBox(*[float(value) for value in boxes[index]]),
            )
        )
    return detections


def _to_rgb_array(frame: Any) -> np.ndarray:
    """Convert an HWC array, PIL image, or image path to an RGB uint8 array."""

    if isinstance(frame, np.ndarray):
        array = frame
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("frame array must have shape (height, width, 3)")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array
    if hasattr(frame, "convert") and hasattr(frame, "size"):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    path = Path(frame)
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


class OpenVINODetector:
    """Native OpenVINO detector implementing the shared ``Detector`` protocol."""

    def __init__(
        self,
        model_path: Union[str, Path],
        imgsz: Optional[int] = None,
        confidence: float = 0.35,
        iou_threshold: float = 0.70,
        class_thresholds: Optional[Dict[str, float]] = None,
        class_ids: Optional[Sequence[int]] = None,
        center_fraction: float = 1 / 3,
        num_threads: int = 2,
        device: str = "CPU",
        input_color_order: str = "RGB",
        max_detections: int = 100,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        order = input_color_order.upper()
        if order not in {"RGB", "BGR"}:
            raise ValueError("input_color_order must be RGB or BGR")

        self.model_path = str(model_path)
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.class_thresholds = dict(class_thresholds or {})
        self.class_ids = list(class_ids) if class_ids is not None else None
        self.center_fraction = center_fraction
        self.device = device
        self.input_color_order = order
        self.max_detections = int(max_detections)
        self.backend_name = BACKEND_NAME

        xml_path = self._resolve_xml(Path(self.model_path))
        self.class_names = self._load_class_names(xml_path)

        from openvino import Core

        self._core = Core()
        self._model = self._core.read_model(str(xml_path))
        shape = list(self._model.input(0).shape)
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError("expected a single NCHW 3-channel input")
        dynamic = not (isinstance(shape[2], int) and isinstance(shape[3], int))
        if imgsz is not None and imgsz > 0 and (dynamic or int(imgsz) != int(shape[2])):
            if not dynamic and int(imgsz) != int(shape[2]):
                raise ValueError(
                    f"model is exported for {shape[2]}x{shape[3]} inputs; "
                    "re-export the IR to change imgsz"
                )
            self._model.reshape({0: [1, 3, int(imgsz), int(imgsz)]})
            shape = [1, 3, int(imgsz), int(imgsz)]
        self.input_size = (int(shape[2]), int(shape[3]))
        self.imgsz = self.input_size[0]

        self._compiled = self._core.compile_model(
            self._model,
            device,
            {
                "INFERENCE_NUM_THREADS": int(num_threads),
                "NUM_STREAMS": "1",
                "PERFORMANCE_HINT": "LATENCY",
            },
        )
        self._request = self._compiled.create_infer_request()
        self._input_tensor = self._request.get_input_tensor(0)
        self._input_buffer = np.empty(
            (1, 3, self.input_size[0], self.input_size[1]), dtype=np.float32
        )
        self._canvas = np.empty(
            (self.input_size[0], self.input_size[1], 3), dtype=np.uint8
        )

    @staticmethod
    def _resolve_xml(path: Path) -> Path:
        if path.is_dir():
            for name in _MODEL_FILES:
                candidate = path / name
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"no OpenVINO IR (.xml) in {path}")
        if path.suffix.lower() != ".xml":
            raise ValueError(f"expected a .xml model file or directory: {path}")
        return path

    @staticmethod
    def _load_class_names(xml_path: Path) -> Dict[int, str]:
        metadata_path = xml_path.parent / "metadata.yaml"
        if metadata_path.exists():
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            names = metadata.get("names") or {}
            if names:
                return {int(key): str(value) for key, value in names.items()}

        from openvino import Core

        model = Core().read_model(str(xml_path))
        labels = model.get_rt_info(["model_info", "labels"])
        value = getattr(labels, "value", None)
        if isinstance(value, (list, tuple)):
            return {index: str(name) for index, name in enumerate(value)}
        if isinstance(value, str) and value:
            return {index: name for index, name in enumerate(value.split(","))}
        raise ValueError(f"no class names found for {xml_path}")

    def _prepare(
        self, frame: Any
    ) -> Tuple[int, int, float, float, float]:
        array = _to_rgb_array(frame)
        if self.input_color_order == "BGR":
            array = np.ascontiguousarray(array[:, :, ::-1])
        batch, scale, pad_x, pad_y, height, width = letterbox_preprocess(
            array, self.input_size, out_buffer=self._canvas
        )
        np.copyto(self._input_buffer, batch)
        return height, width, scale, pad_x, pad_y

    def detect(
        self,
        frame: Any,
        frame_id: int,
        captured_ns: Optional[int] = None,
        source: Optional[str] = None,
    ) -> FrameDetections:
        captured_ns = perf_counter_ns() if captured_ns is None else captured_ns
        started_ns = perf_counter_ns()

        preprocess_start = perf_counter_ns()
        orig_h, orig_w, scale, pad_x, pad_y = self._prepare(frame)
        preprocess_ms = (perf_counter_ns() - preprocess_start) / 1_000_000.0

        inference_start = perf_counter_ns()
        self._input_tensor.data[:] = self._input_buffer
        self._request.infer()
        raw_output = self._request.get_output_tensor(0).data
        inference_ms = (perf_counter_ns() - inference_start) / 1_000_000.0

        postprocess_start = perf_counter_ns()
        detections = decode_yolo_output(
            raw_output,
            self.class_names,
            scale,
            pad_x,
            pad_y,
            orig_w,
            orig_h,
            confidence=self.confidence,
            iou_threshold=self.iou_threshold,
            class_thresholds=self.class_thresholds,
            class_ids=self.class_ids,
            max_detections=self.max_detections,
        )
        detections = assign_sectors(detections, orig_w, self.center_fraction)
        postprocess_ms = (perf_counter_ns() - postprocess_start) / 1_000_000.0

        finished_ns = perf_counter_ns()
        return FrameDetections(
            frame_id=frame_id,
            captured_ns=captured_ns,
            model_started_ns=started_ns,
            model_finished_ns=finished_ns,
            width=orig_w,
            height=orig_h,
            detections=detections,
            backend=BACKEND_NAME,
            model_id=self.model_path,
            stage_ms={
                "preprocess": preprocess_ms,
                "inference": inference_ms,
                "postprocess": postprocess_ms,
            },
            source=source,
        )

    def warmup(self, runs: int = 2) -> None:
        """Run throwaway inferences so the first measured frame is not cold."""

        if runs <= 0:
            return
        dummy = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        for _ in range(runs):
            self.detect(dummy, frame_id=-1)

    def benchmark_frame(
        self, frame: Any, warmup: int = 3, runs: int = 10
    ) -> Dict[str, float]:
        """Return mean per-stage milliseconds over repeated runs of one frame."""

        if runs <= 0:
            raise ValueError("runs must be positive")
        self.warmup(warmup)
        totals: Dict[str, float] = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        for _ in range(runs):
            result = self.detect(frame, frame_id=0)
            for stage, value in result.stage_ms.items():
                totals[stage] = totals.get(stage, 0.0) + float(value)
        stage_ms = {stage: value / runs for stage, value in totals.items()}
        stage_ms["total"] = sum(stage_ms[stage] for stage in totals)
        stage_ms["fps"] = 1000.0 / stage_ms["total"] if stage_ms["total"] > 0 else 0.0
        return stage_ms
