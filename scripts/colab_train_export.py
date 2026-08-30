#!/usr/bin/env python3
"""Self-contained Colab GPU training, held-out validation, export, and packaging."""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
    return normalized


def extract_archive(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        destination_root = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise ValueError(f"unsafe archive path: {member.name}")
        roots = sorted(
            {Path(member.name).parts[0] for member in handle.getmembers() if member.name}
        )
        handle.extractall(destination, filter="data")
    if len(roots) != 1:
        raise ValueError(f"dataset archive must contain exactly one root, found {roots}")
    data_yaml = destination / roots[0] / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"archive does not contain {data_yaml}")
    # Ultralytics resolves an explicit `path: .` against the process working
    # directory in this runtime, not reliably against the YAML's directory.
    # Rewrite only the extracted working copy; the immutable archive is intact.
    yaml_text = data_yaml.read_text(encoding="utf-8")
    lines = yaml_text.splitlines()
    rewritten = []
    replaced_path = False
    for line in lines:
        if line.strip().startswith("path:"):
            rewritten.append(f"path: {data_yaml.parent.resolve()}")
            replaced_path = True
        else:
            rewritten.append(line)
    if not replaced_path:
        rewritten.insert(0, f"path: {data_yaml.parent.resolve()}")
    data_yaml.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return data_yaml


def export_model(
    YOLO: Any, best_path: Path, formats: Iterable[str], imgsz: int
) -> Dict[str, Dict[str, Any]]:
    exports: Dict[str, Dict[str, Any]] = {}
    for export_format in formats:
        options: Dict[str, Any] = {
            "format": export_format,
            "imgsz": imgsz,
            "batch": 1,
            "device": "cpu",
        }
        if export_format == "ncnn":
            options.update({"end2end": False, "quantize": 16})
        try:
            exported = Path(YOLO(str(best_path)).export(**options))
            if not exported.exists():
                raise FileNotFoundError(str(exported))
            artifact_files = (
                [exported]
                if exported.is_file()
                else sorted(path for path in exported.rglob("*") if path.is_file())
            )
            exports[export_format] = {
                "status": "complete",
                "path": str(exported),
                "files": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in artifact_files
                ],
            }
        except Exception as exc:
            exports[export_format] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return exports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--data", default="coco8.yaml")
    parser.add_argument("--dataset-archive", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="gpu-smoke")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", choices=("false", "disk", "ram"), default="false")
    parser.add_argument("--export-backends", default="onnx,ncnn")
    parser.add_argument("--validate-exports", action="store_true")
    parser.add_argument("--require-all-exports", action="store_true")
    args = parser.parse_args()

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.132"]
    )

    import torch
    import ultralytics
    from ultralytics import YOLO

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu:
        raise RuntimeError("CUDA is required for this training run")
    device: Any = 0 if cuda_available else "cpu"
    hardware = torch.cuda.get_device_name(0) if cuda_available else platform.processor()
    project = Path("/content/yolo-pi-runs")
    data = str(args.data)
    dataset_archive = None
    if args.dataset_archive:
        if not args.dataset_archive.exists():
            raise FileNotFoundError(args.dataset_archive)
        dataset_archive = {
            "path": str(args.dataset_archive),
            "bytes": args.dataset_archive.stat().st_size,
            "sha256": sha256(args.dataset_archive),
        }
        data = str(extract_archive(args.dataset_archive, Path("/content/datasets")))

    run_root = project / args.name
    best_path = run_root / "weights" / "best.pt"
    last_path = run_root / "weights" / "last.pt"
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    training_mode = "new"
    if checkpoint:
        best_path = checkpoint
        training_mode = "checkpoint-supplied"
    elif args.resume and last_path.exists():
        model = YOLO(str(last_path))
        train_result = model.train(resume=True)
        best_path = Path(train_result.save_dir) / "weights" / "best.pt"
        training_mode = "resumed"
    elif best_path.exists() and not args.force_train:
        training_mode = "existing-best"
    else:
        model = YOLO(args.model)
        train_result = model.train(
            data=data,
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            cache=False if args.cache == "false" else args.cache,
            project=str(project),
            name=args.name,
            exist_ok=True,
            seed=42,
            deterministic=True,
            plots=True,
            amp=True,
            verbose=True,
        )
        best_path = Path(train_result.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"training completed without best weights at {best_path}")

    best_model = YOLO(str(best_path))
    validation_results: Dict[str, Dict[str, Any]] = {}
    for split in ("val", "test"):
        try:
            result = best_model.val(
                data=data,
                imgsz=args.imgsz,
                device=device,
                split=split,
                plots=False,
                verbose=False,
            )
            validation_results[split] = {
                "status": "complete",
                "normalized": normalized_detection_metrics(result),
                "raw": jsonable_metrics(result),
            }
        except Exception as exc:
            validation_results[split] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    export_formats = [
        item.strip() for item in args.export_backends.split(",") if item.strip()
    ]
    exports = export_model(YOLO, best_path, export_formats, args.imgsz)
    failed_exports = [
        name for name, result in exports.items() if result["status"] != "complete"
    ]
    if args.require_all_exports and failed_exports:
        raise RuntimeError(f"required exports failed: {failed_exports}")

    export_validation: Dict[str, Dict[str, Any]] = {}
    if args.validate_exports:
        for export_format, export in exports.items():
            if export["status"] != "complete":
                continue
            try:
                result = YOLO(str(export["path"])).val(
                    data=data,
                    imgsz=args.imgsz,
                    device="cpu",
                    split="val",
                    plots=False,
                    verbose=False,
                )
                export_validation[export_format] = {
                    "status": "complete",
                    "normalized": normalized_detection_metrics(result),
                    "raw": jsonable_metrics(result),
                }
            except Exception as exc:
                export_validation[export_format] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

    model_info = best_model.info(verbose=False)
    parameter_count = sum(parameter.numel() for parameter in best_model.model.parameters())
    summary = {
        "schema_version": 2,
        "purpose": "Cane model quality and artifact generation; not Raspberry Pi performance evidence",
        "base_model": args.model,
        "data": data,
        "dataset_archive": dataset_archive,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": 42,
        "training_mode": training_mode,
        "cuda_available": cuda_available,
        "hardware": hardware,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "model_info": model_info,
        "parameter_count": parameter_count,
        "best_weights": {
            "path": str(best_path),
            "bytes": best_path.stat().st_size,
            "sha256": sha256(best_path),
        },
        "quality_metrics": validation_results,
        "exports": exports,
        "export_validation": export_validation,
    }
    summary_path = run_root / "validation-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    archive_base = Path("/content/yolo-pi-colab-artifacts")
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", project))
    summary["artifact_archive"] = str(archive_path)
    print("RESULT_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
