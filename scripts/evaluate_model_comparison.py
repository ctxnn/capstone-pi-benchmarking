#!/usr/bin/env python3
"""Evaluate stock and Cane V1 YOLO26n on the same held-out target taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import platform
import time
from pathlib import Path
from typing import Any, Dict, Mapping


COCO_TO_CANE = {
    "person": "person",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "bicycle": "bicycle_motorcycle",
    "motorcycle": "bicycle_motorcycle",
    "dog": "dog",
    "stop sign": "signboard",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable_metrics(metrics: Any) -> Dict[str, float]:
    return {
        str(name): float(value)
        for name, value in getattr(metrics, "results_dict", {}).items()
        if isinstance(value, numbers.Real)
    }


def normalized_detection_metrics(metrics: Any) -> Dict[str, float]:
    raw = jsonable_metrics(metrics)
    aliases = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    normalized = {name: raw[key] for name, key in aliases.items() if key in raw}
    normalized["fitness"] = float(getattr(metrics, "fitness", 0.0))
    missing = sorted(set(aliases) - set(normalized))
    if missing:
        raise RuntimeError(f"validator did not return required metrics: {missing}")
    return normalized


def names_dict(names: Any) -> Dict[int, str]:
    if isinstance(names, Mapping):
        return {int(index): str(name) for index, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def absolute_data_yaml(source: Path, destination: Path) -> Path:
    """Write a runtime YAML whose dataset root is independent of current cwd."""

    lines = source.resolve().read_text(encoding="utf-8").splitlines()
    rewritten = []
    replaced = False
    for line in lines:
        if line.strip().startswith("path:"):
            value = line.split(":", 1)[1].strip().strip("'\"")
            candidate = Path(value) if value else Path(".")
            root = candidate if candidate.is_absolute() else source.resolve().parent / candidate
            rewritten.append(f"path: {root.resolve()}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        rewritten.insert(0, f"path: {source.resolve().parent}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return destination


def build_mapped_validator() -> type:
    """Create a validator that maps COCO predictions into the Cane taxonomy."""

    import torch
    from torchvision.ops import batched_nms
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import ConfusionMatrix

    class MappedCocoValidator(DetectionValidator):
        def init_metrics(self, model: Any) -> None:
            source_names = names_dict(model.names)
            super().init_metrics(model)
            target_names = names_dict(self.data["names"])
            target_ids = {name: index for index, name in target_names.items()}
            self.source_to_target = {
                source_id: target_ids[COCO_TO_CANE[source_name]]
                for source_id, source_name in source_names.items()
                if source_name in COCO_TO_CANE
                and COCO_TO_CANE[source_name] in target_ids
            }
            self.names = target_names
            self.nc = len(target_names)
            self.metrics.names = target_names
            self.confusion_matrix = ConfusionMatrix(
                names=target_names,
                save_matches=self.args.plots and self.args.visualize,
            )

        def _prepare_pred(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            pred = super()._prepare_pred(pred)
            if pred["cls"].numel() == 0:
                return pred
            mapped = torch.full_like(pred["cls"], -1)
            for source_id, target_id in self.source_to_target.items():
                mapped[pred["cls"] == source_id] = target_id
            retained = mapped >= 0
            output = {name: value[retained] for name, value in pred.items()}
            output["cls"] = mapped[retained]
            if output["cls"].numel():
                keep = batched_nms(
                    output["bboxes"],
                    output["conf"],
                    output["cls"],
                    float(self.args.iou),
                )
                output = {name: value[keep] for name, value in output.items()}
            return output

    return MappedCocoValidator


def model_record(model: Any, checkpoint: Path, metrics: Any) -> Dict[str, Any]:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "bytes": checkpoint.stat().st_size,
        "sha256": sha256(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.model.parameters()),
        "normalized": normalized_detection_metrics(metrics),
        "raw": jsonable_metrics(metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fine-tuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO

    runtime_yaml = absolute_data_yaml(
        args.data, args.output.with_name("model-comparison-data.yaml")
    )
    common = {
        "data": str(runtime_yaml),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "split": "test",
        "fraction": 1.0,
        "plots": False,
        "verbose": False,
        "project": str(args.output.parent / "quality-runs"),
        "exist_ok": True,
    }

    baseline = YOLO(str(args.baseline))
    baseline_metrics = baseline.val(
        validator=build_mapped_validator(), name="baseline-mapped", **common
    )
    fine_tuned = YOLO(str(args.fine_tuned))
    fine_tuned_metrics = fine_tuned.val(name="fine-tuned", **common)

    data_contract = yaml.safe_load(runtime_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(data_contract["path"])
    test_path = dataset_root / str(data_contract["test"])
    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    if test_path.is_file():
        evaluated_image_count = sum(
            1 for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    else:
        evaluated_image_count = sum(
            1
            for path in test_path.rglob("*")
            if path.is_file() and path.suffix.lower() in image_suffixes
        )
    if evaluated_image_count <= 0:
        raise ValueError(f"resolved test split contains no images: {test_path}")

    summary = {
        "schema_version": 1,
        "purpose": "Comparable held-out architecture quality at the Pi matrix resolution; not Pi performance evidence",
        "created_at_epoch_ns": time.time_ns(),
        "data": str(args.data.resolve()),
        "runtime_data_yaml": str(runtime_yaml.resolve()),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "fraction": 1.0,
        "evaluated_image_count": evaluated_image_count,
        "target_class_names": names_dict(data_contract["names"]),
        "device": str(args.device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "baseline_protocol": {
            "source_taxonomy": "COCO 80",
            "target_taxonomy": "Cane V1 15",
            "prediction_name_mapping": COCO_TO_CANE,
            "unmapped_prediction_policy": "drop",
            "unsupported_target_ground_truth_policy": "retain as false negatives",
            "post_mapping_nms_iou": 0.7,
        },
        "models": {
            "v0-pytorch-fp32": model_record(baseline, args.baseline, baseline_metrics),
            "cane-v1-pytorch-fp32": model_record(
                fine_tuned, args.fine_tuned, fine_tuned_metrics
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
