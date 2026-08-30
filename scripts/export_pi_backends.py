#!/usr/bin/env python3
"""Export one trained YOLO model to the exact 640/FP32 Pi runtime matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import types
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
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def disable_mnn_converter_telemetry() -> None:
    """Prevent MNN's converter helper from auto-installing/sending telemetry."""

    module_name = "MNN.tools.utils.log"
    if module_name not in sys.modules:
        stub = types.ModuleType(module_name)
        stub.mnn_logger = None
        sys.modules[module_name] = stub


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--formats", default="onnx,openvino,mnn,ncnn,litert"
    )
    args = parser.parse_args()

    from ultralytics import YOLO, __version__ as ultralytics_version
    from yolo_pi.detector import UltralyticsDetector

    args.output.mkdir(parents=True, exist_ok=True)
    pytorch_path = args.output / "best.pt"
    if args.model.resolve() != pytorch_path.resolve():
        shutil.copy2(args.model, pytorch_path)
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "purpose": "YOLO26n-Cane V1 Pi runtime matrix at 640/FP32",
        "created_at_epoch_ns": time.time_ns(),
        "python": platform.python_version(),
        "ultralytics": ultralytics_version,
        "imgsz": args.imgsz,
        "precision": "FP32",
        "source_model": {
            "path": str(pytorch_path),
            "bytes": pytorch_path.stat().st_size,
            "sha256": sha256(pytorch_path),
        },
        "exports": {},
    }
    failures = []
    for export_format in formats:
        if export_format == "mnn":
            disable_mnn_converter_telemetry()
        options: Dict[str, Any] = {
            "format": export_format,
            "imgsz": args.imgsz,
            "batch": 1,
            "device": "cpu",
            "end2end": False,
        }
        try:
            exported = Path(YOLO(str(pytorch_path)).export(**options)).resolve()
        except Exception as exc:
            failures.append(export_format)
            manifest["exports"][export_format] = {
                "status": "export_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "export_options": options,
            }
            (args.output / "export-manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            continue

        entry: Dict[str, Any] = {
            "status": "converted",
            "path": str(exported),
            "export_options": options,
        }
        try:
            smoke = UltralyticsDetector(
                exported,
                imgsz=args.imgsz,
                confidence=0.25,
                device="cpu",
                runtime_threads=4,
                runtime_precision="FP32",
            ).detect(
                str(args.sample),
                frame_id=0,
                source="export-functional-smoke",
            )
            entry.update(
                {
                    "status": "complete",
                    "smoke": {
                        "status": "complete",
                        "sample": str(args.sample),
                        "detections": len(smoke.detections),
                        "backend": smoke.backend,
                        "runtime_threads": 4,
                        "runtime_precision": "FP32",
                        "speed_ms": smoke.stage_ms,
                    },
                }
            )
        except Exception as exc:
            failures.append(export_format)
            entry.update(
                {
                    "status": "smoke_failed",
                    "smoke": {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                }
            )
        files = artifact_files(exported)
        entry.update(
            {
                "tree_sha256": tree_sha256(exported),
                "files": [
                    {
                        "path": str(item),
                        "bytes": item.stat().st_size,
                        "sha256": sha256(item),
                    }
                    for item in files
                ],
            }
        )
        manifest["exports"][export_format] = entry
        (args.output / "export-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # Some converters reuse and overwrite an earlier intermediate such as
    # best.onnx. Reconcile any changed artifact only after every conversion has
    # finished, then bind smoke evidence to the bytes that will be transferred.
    for export_format, entry in manifest["exports"].items():
        exported_path = Path(entry.get("path", ""))
        if not exported_path.exists():
            continue
        current_tree = tree_sha256(exported_path)
        if current_tree == entry.get("tree_sha256"):
            continue
        files = artifact_files(exported_path)
        entry["tree_sha256"] = current_tree
        entry["files"] = [
            {
                "path": str(item),
                "bytes": item.stat().st_size,
                "sha256": sha256(item),
            }
            for item in files
        ]
        try:
            smoke = UltralyticsDetector(
                exported_path,
                imgsz=args.imgsz,
                confidence=0.25,
                device="cpu",
                runtime_threads=4,
                runtime_precision="FP32",
            ).detect(str(args.sample), frame_id=0, source="final-export-functional-smoke")
            entry["status"] = "complete"
            entry["smoke"] = {
                "status": "complete",
                "sample": str(args.sample),
                "detections": len(smoke.detections),
                "backend": smoke.backend,
                "runtime_threads": 4,
                "runtime_precision": "FP32",
                "speed_ms": smoke.stage_ms,
            }
        except Exception as exc:
            failures.append(export_format)
            entry["status"] = "smoke_failed"
            entry["smoke"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    (args.output / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
