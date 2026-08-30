"""Detector interface and optional runtime-specific Ultralytics adapters."""

import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

from .sectors import assign_sectors
from .types import BoundingBox, Detection, FrameDetections


def infer_backend_name(model_path: Union[str, Path]) -> str:
    path = Path(model_path)
    if path.is_dir() and (path / "model.ncnn.param").exists():
        return "ncnn-ultralytics"
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return "pytorch-ultralytics"
    if suffix == ".onnx":
        return "onnxruntime-ultralytics"
    if suffix == ".mnn":
        return "mnn-high-precision-ultralytics"
    if suffix == ".tflite":
        return "litert-ultralytics"
    if suffix in {".xml", ".bin"} or path.name.endswith("_openvino_model"):
        return "openvino-ultralytics"
    return "ultralytics-auto"


def mnn_runtime_config(threads: int, precision: str) -> Dict[str, Any]:
    """Return the explicit native-MNN CPU configuration for one matrix row."""

    if threads <= 0:
        raise ValueError("threads must be positive")
    if precision.upper() != "FP32":
        raise ValueError("the exact MNN benchmark adapter currently supports FP32 only")
    return {"precision": "high", "backend": "CPU", "numThread": threads}


def _install_mnn_backend(threads: int, precision: str) -> None:
    """Replace Ultralytics' low-precision MNN default with an explicit FP32 adapter."""

    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.nn.backends.mnn import MNNBackend
    from ultralytics.utils import LOGGER
    from ultralytics.utils.checks import check_requirements

    config = mnn_runtime_config(threads, precision)

    class ExactMNNBackend(MNNBackend):
        def load_model(self, weight: Union[str, Path]) -> None:
            LOGGER.info(
                f"Loading {weight} for MNN inference with "
                f"precision={config['precision']} threads={config['numThread']}..."
            )
            check_requirements("MNN")
            import MNN

            runtime = MNN.nn.create_runtime_manager((config,))
            self.net = MNN.nn.load_module_from_file(
                weight, [], [], runtime_manager=runtime, rearrange=True
            )
            self.expr = MNN.expr
            info = self.net.get_info()
            if "bizCode" in info:
                try:
                    self.apply_metadata(json.loads(info["bizCode"]))
                except json.JSONDecodeError:
                    pass

    AutoBackend._BACKEND_MAP["mnn"] = ExactMNNBackend


class Detector(Protocol):
    """Common interface shared by laptop and Raspberry Pi detectors."""

    def detect(
        self,
        frame: Any,
        frame_id: int,
        captured_ns: Optional[int] = None,
        source: Optional[str] = None,
    ) -> FrameDetections:
        """Run detection and return backend-neutral results."""


class UltralyticsDetector:
    """YOLO adapter with lazy imports so core tests need no ML installation."""

    def __init__(
        self,
        model_path: Union[str, Path],
        imgsz: int = 416,
        confidence: float = 0.35,
        device: Optional[str] = None,
        class_ids: Optional[Sequence[int]] = None,
        center_fraction: float = 1 / 3,
        runtime_threads: int = 4,
        runtime_precision: str = "FP32",
    ) -> None:
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install the 'vision' extra before "
                "constructing UltralyticsDetector."
            ) from exc

        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.confidence = confidence
        self.device = device
        self.class_ids = list(class_ids) if class_ids is not None else None
        self.center_fraction = center_fraction
        self.backend_name = infer_backend_name(self.model_path)
        if Path(self.model_path).suffix.lower() == ".mnn":
            _install_mnn_backend(runtime_threads, runtime_precision)
        self._model = YOLO(self.model_path)

    def detect(
        self,
        frame: Any,
        frame_id: int,
        captured_ns: Optional[int] = None,
        source: Optional[str] = None,
    ) -> FrameDetections:
        captured_ns = perf_counter_ns() if captured_ns is None else captured_ns
        started_ns = perf_counter_ns()
        results = self._model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )
        finished_ns = perf_counter_ns()
        if len(results) != 1:
            raise RuntimeError("Single-frame detect expected exactly one result")

        result = results[0]
        height, width = (int(result.orig_shape[0]), int(result.orig_shape[1]))
        names = result.names
        parsed: List[Detection] = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()
            for coordinates, confidence, class_id_value in zip(xyxy, confidences, class_ids):
                class_id = int(class_id_value)
                parsed.append(
                    Detection(
                        class_id=class_id,
                        class_name=str(names[class_id]),
                        confidence=float(confidence),
                        box=BoundingBox(*[float(value) for value in coordinates]),
                    )
                )

        stage_ms = {
            str(name): float(duration)
            for name, duration in getattr(result, "speed", {}).items()
        }
        return FrameDetections(
            frame_id=frame_id,
            captured_ns=captured_ns,
            model_started_ns=started_ns,
            model_finished_ns=finished_ns,
            width=width,
            height=height,
            detections=assign_sectors(parsed, width, self.center_fraction),
            backend=self.backend_name,
            model_id=self.model_path,
            stage_ms=stage_ms,
            source=source,
        )
