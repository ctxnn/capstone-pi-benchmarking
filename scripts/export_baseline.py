#!/usr/bin/env python3
"""Export V0 to ONNX and FP16 NCNN, smoke-test each, and record checksums."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_checksums(path: Path) -> Dict[str, str]:
    if path.is_file():
        return {str(path): sha256(path)}
    return {
        str(item): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and "__pycache__" not in item.parts
    }


def detection_summary(model_path: Path, sample: Path, imgsz: int) -> Dict[str, object]:
    result = YOLO(str(model_path)).predict(
        source=str(sample), imgsz=imgsz, conf=0.35, device="cpu", verbose=False
    )[0]
    names = result.names
    labels = [] if result.boxes is None else [names[int(value)] for value in result.boxes.cls]
    return {
        "detection_count": len(labels),
        "class_names": sorted(str(label) for label in labels),
        "stage_ms": {name: float(value) for name, value in result.speed.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/models/v0-export-manifest.json")
    )
    args = parser.parse_args()
    if not args.model.exists() or not args.sample.exists():
        raise FileNotFoundError("Model and sample must exist before export")

    onnx_path = Path(
        YOLO(str(args.model)).export(
            format="onnx", imgsz=args.imgsz, batch=1, device="cpu"
        )
    )
    ncnn_path = Path(
        YOLO(str(args.model)).export(
            format="ncnn",
            imgsz=args.imgsz,
            batch=1,
            device="cpu",
            end2end=False,
            quantize=16,
        )
    )
    manifest = {
        "schema_version": 1,
        "model_role": "V0 pretrained functional baseline",
        "imgsz": args.imgsz,
        "scope": "export parity and laptop correctness; not Pi performance",
        "artifacts": {
            "pytorch": artifact_checksums(args.model),
            "onnx": artifact_checksums(onnx_path),
            "ncnn_fp16": artifact_checksums(ncnn_path),
        },
        "sample_sha256": sha256(args.sample),
        "smoke_detections": {
            "pytorch": detection_summary(args.model, args.sample, args.imgsz),
            "onnx": detection_summary(onnx_path, args.sample, args.imgsz),
            "ncnn_fp16": detection_summary(ncnn_path, args.sample, args.imgsz),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
