#!/usr/bin/env python3
"""Resume Cane V1 training safely inside a persistent Lightning AI Studio.

The script is intentionally self-contained so the prepared handoff folder can be
uploaded without the rest of the repository. Re-running the same command selects
the newest resumable checkpoint saved in ``lightning-output/checkpoint-snapshots``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import os
import platform
import shutil
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


EXPECTED_ARCHIVE_BYTES = 2_013_204_992
EXPECTED_ARCHIVE_SHA256 = (
    "f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91"
)
EXPECTED_INPUT_CHECKPOINT_SHA256 = (
    "0da75726f7a058e63d7b030601447ddeb4bd98b420d1387482761aa5799a0eea"
)
EXPECTED_NAMES = [
    "person",
    "vehicle",
    "bicycle_motorcycle",
    "pole",
    "tree",
    "stairs",
    "curb",
    "barrier",
    "cone",
    "dog",
    "signboard",
    "door",
    "slope",
    "overhead_obstacle",
    "generic_obstacle",
]
RUN_NAME = "cane-v1-yolo26n-e30-img416"
LATEST_RECOVERY_ZIP = "cane-v1-latest-recovery.zip"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        destination_root = destination.resolve()
        members = handle.getmembers()
        roots = sorted({Path(member.name).parts[0] for member in members if member.name})
        if len(roots) != 1:
            raise ValueError(f"dataset archive must have one root, found {roots}")
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"dataset archive links are not allowed: {member.name}")
            target = (destination / member.name).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise ValueError(f"unsafe archive path: {member.name}")
        try:
            handle.extractall(destination, filter="data")
        except TypeError:  # Python versions before extraction filters were added.
            handle.extractall(destination)
    data_yaml = destination / roots[0] / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    rewritten = []
    replaced = False
    for line in lines:
        if line.strip().startswith("path:"):
            rewritten.append(f"path: {data_yaml.parent.resolve()}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        rewritten.insert(0, f"path: {data_yaml.parent.resolve()}")
    data_yaml.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return data_yaml


def ensure_dataset(archive: Path, workspace: Path) -> Path:
    extraction_root = workspace / "datasets"
    receipt_path = extraction_root / "extraction-receipt.json"
    data_yaml = extraction_root / "cane-v1-training-640" / "data.yaml"
    if receipt_path.is_file() and data_yaml.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256:
            return data_yaml
    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    data_yaml = safe_extract(archive, extraction_root)
    atomic_json(
        receipt_path,
        {
            "schema_version": 1,
            "archive": str(archive.resolve()),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "data_yaml": str(data_yaml.resolve()),
            "extracted_at": utc_now(),
        },
    )
    return data_yaml


def checkpoint_metadata(path: Path, torch: Any) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    completed_epoch = int(checkpoint.get("epoch", -1)) + 1
    optimizer_present = checkpoint.get("optimizer") is not None
    model = checkpoint.get("ema") or checkpoint.get("model")
    raw_names = getattr(model, "names", {})
    if isinstance(raw_names, dict):
        names = [raw_names[index] for index in sorted(raw_names)]
    else:
        names = list(raw_names)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "completed_epoch": completed_epoch,
        "optimizer_present": optimizer_present,
        "names": names,
        "best_fitness": float(checkpoint.get("best_fitness", 0.0)),
    }


def select_resume_checkpoint(
    input_checkpoint: Path, output: Path, torch: Any
) -> Dict[str, Any]:
    candidates = [input_checkpoint]
    run_last = output / "runs" / RUN_NAME / "weights" / "last.pt"
    if run_last.is_file():
        candidates.append(run_last)
    candidates.extend(sorted((output / "checkpoint-snapshots").glob("last-epoch-*.pt")))
    valid = []
    rejected = []
    for candidate in candidates:
        try:
            metadata = checkpoint_metadata(candidate, torch)
            if metadata["completed_epoch"] <= 0:
                raise ValueError("checkpoint has no resumable epoch")
            if not metadata["optimizer_present"]:
                raise ValueError("checkpoint has no optimizer state")
            if metadata["names"] != EXPECTED_NAMES:
                raise ValueError(f"taxonomy mismatch: {metadata['names']}")
            valid.append(metadata)
        except Exception as exc:
            rejected.append(
                {
                    "path": str(candidate.resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    if not valid:
        raise RuntimeError(f"no valid resumable checkpoint; rejected={rejected}")
    selected = max(valid, key=lambda item: (item["completed_epoch"], item["sha256"]))
    if selected["completed_epoch"] >= 30:
        raise RuntimeError(
            "selected checkpoint already reports 30 completed epochs but no completion receipt exists"
        )
    return {"selected": selected, "valid_candidates": valid, "rejected": rejected}


def normalized_metrics(metrics: Any) -> Dict[str, float]:
    raw = {
        str(name): float(value)
        for name, value in getattr(metrics, "results_dict", {}).items()
        if isinstance(value, numbers.Real)
    }
    aliases = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    result = {name: raw[key] for name, key in aliases.items() if key in raw}
    result["fitness"] = float(getattr(metrics, "fitness", 0.0))
    return result


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def protect_resumable_checkpoint(
    source: Path,
    completed_epoch: int,
    output: Path,
    torch: Any,
    reason: str,
) -> Dict[str, Any]:
    """Atomically preserve one resumable epoch and a download-ready recovery ZIP."""

    source_metadata = checkpoint_metadata(source, torch)
    if source_metadata["completed_epoch"] != completed_epoch:
        raise RuntimeError(
            "protected checkpoint epoch mismatch: "
            f"{source_metadata['completed_epoch']} != {completed_epoch}"
        )
    if not source_metadata["optimizer_present"]:
        raise RuntimeError("protected checkpoint has no optimizer state")
    if source_metadata["names"] != EXPECTED_NAMES:
        raise RuntimeError("protected checkpoint taxonomy mismatch")

    snapshots = output / "checkpoint-snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    destination = snapshots / f"last-epoch-{completed_epoch:03d}.pt"
    if source.resolve() != destination.resolve():
        copy_atomic(source, destination)
    metadata = checkpoint_metadata(destination, torch)
    if metadata["sha256"] != source_metadata["sha256"]:
        raise RuntimeError("protected checkpoint checksum mismatch after copy")

    latest_receipt = output / "LATEST_RESUMABLE.json"
    receipt = {
        "schema_version": 1,
        "saved_at": utc_now(),
        "reason": reason,
        "checkpoint": metadata,
    }
    atomic_json(latest_receipt, receipt)

    archive_path = output / LATEST_RECOVERY_ZIP
    temporary_archive = output / f".{LATEST_RECOVERY_ZIP}.tmp"
    temporary_archive.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary_archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as bundle:
            bundle.write(
                destination,
                arcname=f"checkpoint-snapshots/{destination.name}",
            )
            bundle.write(latest_receipt, arcname="LATEST_RESUMABLE.json")
        with zipfile.ZipFile(temporary_archive) as bundle:
            if bundle.testzip() is not None:
                raise RuntimeError("latest recovery ZIP failed its CRC check")
        temporary_archive.replace(archive_path)
    finally:
        temporary_archive.unlink(missing_ok=True)

    bundle_receipt = {
        "schema_version": 1,
        "status": "recoverable",
        "saved_at": utc_now(),
        "completed_epoch": completed_epoch,
        "checkpoint_sha256": metadata["sha256"],
        "archive": {
            "path": str(archive_path.resolve()),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256(archive_path),
        },
        "restore_command": f"unzip {LATEST_RECOVERY_ZIP} -d lightning-output",
    }
    atomic_json(output / "LATEST_RECOVERY_BUNDLE.json", bundle_receipt)
    print(
        "PROTECTED_CHECKPOINT="
        + json.dumps(
            {"checkpoint": metadata, "recovery_bundle": bundle_receipt},
            sort_keys=True,
        ),
        flush=True,
    )
    return {"checkpoint": metadata, "recovery_bundle": bundle_receipt}


def existing_completion(output: Path) -> Dict[str, Any] | None:
    receipt_path = output / "LIGHTNING_COMPLETE.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    best = Path(receipt.get("deliverables", {}).get("best", {}).get("path", ""))
    if not best.is_file():
        return None
    if sha256(best) != receipt["deliverables"]["best"]["sha256"]:
        raise RuntimeError("existing completion receipt best.pt checksum mismatch")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-archive", type=Path, default=Path("cane-v1-training-640.tar"))
    parser.add_argument("--checkpoint", type=Path, default=Path("resume-last-epoch22.pt"))
    parser.add_argument("--output", type=Path, default=Path("lightning-output"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    archive = args.dataset_archive if args.dataset_archive.is_absolute() else workspace / args.dataset_archive
    input_checkpoint = args.checkpoint if args.checkpoint.is_absolute() else workspace / args.checkpoint
    output = args.output if args.output.is_absolute() else workspace / args.output
    output.mkdir(parents=True, exist_ok=True)

    if not args.force:
        completion = existing_completion(output)
        if completion:
            print("TRAINING_ALREADY_COMPLETE=" + json.dumps(completion, sort_keys=True), flush=True)
            return 0

    if not archive.is_file():
        raise FileNotFoundError(archive)
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"dataset archive size mismatch: {archive.stat().st_size} != {EXPECTED_ARCHIVE_BYTES}"
        )
    archive_sha = sha256(archive)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"dataset archive checksum mismatch: {archive_sha}")
    if not input_checkpoint.is_file():
        raise FileNotFoundError(input_checkpoint)
    input_sha = sha256(input_checkpoint)
    if input_sha != EXPECTED_INPUT_CHECKPOINT_SHA256:
        raise RuntimeError(f"input checkpoint checksum mismatch: {input_sha}")

    from ultralytics import YOLO, __version__ as ultralytics_version
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; switch the Lightning Studio to a GPU machine")

    data_yaml = ensure_dataset(archive, workspace)
    selection = select_resume_checkpoint(input_checkpoint, output, torch)
    selection.update(
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_archive_sha256": archive_sha,
            "data_yaml": str(data_yaml),
        }
    )
    atomic_json(output / "resume-selection.json", selection)

    run_dir = output / "runs" / RUN_NAME
    run_last = run_dir / "weights" / "last.pt"
    copy_atomic(Path(selection["selected"]["path"]), run_last)
    protect_resumable_checkpoint(
        run_last,
        int(selection["selected"]["completed_epoch"]),
        output,
        torch,
        reason="startup-selection",
    )
    snapshots = output / "checkpoint-snapshots"

    model = YOLO(str(run_last))

    def protect_checkpoint(trainer: Any) -> None:
        source = Path(trainer.last)
        if not source.is_file():
            return
        completed_epoch = int(trainer.epoch) + 1
        protect_resumable_checkpoint(
            source,
            completed_epoch,
            output,
            torch,
            reason="on-model-save",
        )

    model.add_callback("on_model_save", protect_checkpoint)
    model.train(
        resume=True,
        data=str(data_yaml),
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        cache=False,
        imgsz=416,
        patience=8,
        plots=True,
        val=True,
        save_dir=str(run_dir),
    )

    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError(f"training ended without best/last weights in {run_dir / 'weights'}")

    best_model = YOLO(str(best_path))
    quality: Dict[str, Any] = {}
    for split in ("val", "test"):
        result = best_model.val(
            data=str(data_yaml),
            imgsz=416,
            device=args.device,
            split=split,
            plots=False,
            verbose=False,
        )
        quality[split] = normalized_metrics(result)

    deliverables = output / "deliverables"
    deliverables.mkdir(parents=True, exist_ok=True)
    copy_atomic(best_path, deliverables / "best.pt")
    copy_atomic(last_path, deliverables / "last.pt")
    for name in ("results.csv", "args.yaml"):
        source = run_dir / name
        if source.is_file():
            copy_atomic(source, deliverables / name)
    latest_resume = output / "LATEST_RESUMABLE.json"
    if latest_resume.is_file():
        copy_atomic(latest_resume, deliverables / latest_resume.name)
    latest_snapshots = sorted(snapshots.glob("last-epoch-*.pt"))
    if not latest_snapshots:
        raise RuntimeError("training completed without a protected resumable checkpoint")
    copy_atomic(latest_snapshots[-1], deliverables / "latest-resumable.pt")
    resume_info = checkpoint_metadata(deliverables / "latest-resumable.pt", torch)
    if not resume_info["optimizer_present"]:
        raise RuntimeError("protected final resume checkpoint has no optimizer state")

    summary = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "purpose": "Cane model quality artifact; Raspberry Pi performance must be measured on the Pi",
        "hardware": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics_version,
        "input_checkpoint": selection["selected"],
        "dataset_archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": archive_sha,
        },
        "training": {
            "target_epochs": 30,
            "imgsz": 416,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "seed": 42,
            "deterministic": True,
        },
        "quality_metrics": quality,
        "deliverables": {
            "best": {
                "path": str((deliverables / "best.pt").resolve()),
                "bytes": (deliverables / "best.pt").stat().st_size,
                "sha256": sha256(deliverables / "best.pt"),
            },
            "last": {
                "path": str((deliverables / "last.pt").resolve()),
                "bytes": (deliverables / "last.pt").stat().st_size,
                "sha256": sha256(deliverables / "last.pt"),
            },
            "latest_resumable": resume_info,
        },
        "environment": {
            "cwd": str(Path.cwd()),
            "workspace": str(workspace),
            "pid": os.getpid(),
            "platform": platform.platform(),
        },
    }
    atomic_json(deliverables / "validation-summary.json", summary)
    archive_base = output / "cane-v1-lightning-deliverable"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", output, "deliverables"))
    summary["deliverable_archive"] = {
        "path": str(archive_path.resolve()),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256(archive_path),
    }
    atomic_json(deliverables / "validation-summary.json", summary)
    atomic_json(output / "LIGHTNING_COMPLETE.json", summary)
    print("LIGHTNING_COMPLETE=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"LIGHTNING_FAILED={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
