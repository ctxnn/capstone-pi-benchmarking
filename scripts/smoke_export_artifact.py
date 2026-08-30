#!/usr/bin/env python3
"""Create checksum-bound functional-smoke evidence for one exported model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
        and item.name != ".DS_Store"
    )


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    root = path if path.is_dir() else path.parent
    for item in artifact_files(path):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def backend_name(path: Path) -> str:
    if path.is_dir() and (path / "model.ncnn.param").exists():
        return "ncnn-ultralytics"
    if path.name.endswith("_openvino_model") or path.suffix.lower() == ".xml":
        return "openvino-ultralytics"
    return {
        ".onnx": "onnxruntime-ultralytics",
        ".mnn": "mnn-ultralytics",
        ".tflite": "litert-ultralytics",
    }.get(path.suffix.lower(), "ultralytics-auto")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--precision", default="FP32")
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    started_ns = time.time_ns()
    evidence: Dict[str, Any] = {
        "schema_version": 1,
        "format": args.format,
        "imgsz": args.imgsz,
        "precision": args.precision,
        "artifact": {
            "path": str(args.artifact.resolve()),
            "tree_sha256": tree_sha256(args.artifact),
        },
        "sample": {
            "path": str(args.sample.resolve()),
            "sha256": sha256(args.sample),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "started_at_epoch_ns": started_ns,
    }
    try:
        results = YOLO(str(args.artifact)).predict(
            source=str(args.sample),
            imgsz=args.imgsz,
            conf=0.25,
            device="cpu",
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError("functional smoke expected exactly one result")
        result = results[0]
        evidence.update(
            {
                "status": "complete",
                "smoke": {
                    "status": "complete",
                    "backend": backend_name(args.artifact),
                    "detections": len(result.boxes) if result.boxes is not None else 0,
                    "speed_ms": {
                        str(name): float(value)
                        for name, value in getattr(result, "speed", {}).items()
                    },
                },
            }
        )
    except Exception as exc:
        evidence.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    evidence["finished_at_epoch_ns"] = time.time_ns()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
