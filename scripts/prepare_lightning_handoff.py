#!/usr/bin/env python3
"""Rebuild and verify the ignored large inputs in ``lightning-ai-upload``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict


ARCHIVE_BYTES = 2_013_204_992
ARCHIVE_SHA256 = "f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91"
CHECKPOINT_SHA256 = "0da75726f7a058e63d7b030601447ddeb4bd98b420d1387482761aa5799a0eea"
EXPECTED_NAMES = {
    0: "person",
    1: "vehicle",
    2: "bicycle_motorcycle",
    3: "pole",
    4: "tree",
    5: "stairs",
    6: "curb",
    7: "barrier",
    8: "cone",
    9: "dog",
    10: "signboard",
    11: "door",
    12: "slope",
    13: "overhead_obstacle",
    14: "generic_obstacle",
}
SCAFFOLD = (
    "README-FIRST.md",
    "LIGHTNING-INSTRUCTIONS.md",
    "lightning_train_resume.py",
    "start-training.sh",
    "status-training.sh",
    "resume-manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_info(path: Path) -> Dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    names = {int(index): str(name) for index, name in getattr(model, "names", {}).items()}
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "completed_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "optimizer_present": checkpoint.get("optimizer") is not None,
        "class_names": names,
    }


def validate_source_inputs(archive: Path, checkpoint: Path) -> Dict[str, Any]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError(f"training archive byte count mismatch: {archive.stat().st_size}")
    archive_hash = sha256(archive)
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError(f"training archive SHA-256 mismatch: {archive_hash}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_record = checkpoint_info(checkpoint)
    if checkpoint_record["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("resume checkpoint SHA-256 mismatch")
    if checkpoint_record["completed_epoch"] != 22:
        raise ValueError("resume checkpoint is not completed epoch 22")
    if not checkpoint_record["optimizer_present"]:
        raise ValueError("resume checkpoint has no optimizer state")
    if checkpoint_record["class_names"] != EXPECTED_NAMES:
        raise ValueError("resume checkpoint taxonomy mismatch")
    return {
        "archive": {
            "source": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "sha256": archive_hash,
        },
        "checkpoint": {
            "source": str(checkpoint.resolve()),
            **checkpoint_record,
        },
    }


def materialize(source: Path, destination: Path) -> str:
    expected_hash = sha256(source)
    if destination.exists():
        if not destination.is_file() or sha256(destination) != expected_hash:
            raise FileExistsError(
                f"refusing to replace mismatched handoff file: {destination}"
            )
        return "already-present"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    method = "hardlink"
    try:
        os.link(source, temporary)
    except OSError:
        method = "copy"
        shutil.copy2(source, temporary)
    if sha256(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"materialized file checksum mismatch: {destination}")
    temporary.replace(destination)
    return method


def validate_scaffold(output: Path, root: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for name in SCAFFOLD:
        candidate = output / name
        if not candidate.is_file():
            raise FileNotFoundError(f"tracked Lightning scaffold is missing: {candidate}")
        records[name] = {
            "bytes": candidate.stat().st_size,
            "sha256": sha256(candidate),
        }
    canonical = {
        "lightning_train_resume.py": root / "scripts/lightning_train_resume.py",
        "LIGHTNING-INSTRUCTIONS.md": root / "docs/lightning-ai-training.md",
    }
    for name, source in canonical.items():
        if records[name]["sha256"] != sha256(source):
            raise ValueError(f"tracked handoff copy is stale relative to {source}")
    return records


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("artifacts/training/cane-v1-training-640.tar"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/training/checkpoint-guard/checkpoints/snapshots/"
            "last-epoch-022-0da75726f7a0.pt"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("lightning-ai-upload"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    output = args.output.resolve()
    sources = validate_source_inputs(args.archive.resolve(), args.checkpoint.resolve())
    scaffold = validate_scaffold(output, root)
    destinations = {
        "archive": output / "cane-v1-training-640.tar",
        "checkpoint": output / "resume-last-epoch22.pt",
    }
    methods: Dict[str, str] = {}
    for key, destination in destinations.items():
        source = Path(sources[key]["source"])
        if args.check:
            if not destination.is_file() or sha256(destination) != sources[key]["sha256"]:
                raise ValueError(f"handoff package check failed: {destination}")
            methods[key] = "verified"
        else:
            methods[key] = materialize(source, destination)

    receipt = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "checksum-gated Lightning upload package",
        "output": str(output),
        "sources": sources,
        "materialization": methods,
        "scaffold": scaffold,
    }
    if not args.check:
        atomic_json(output / "PACKAGE-RECEIPT.json", receipt)
    print("LIGHTNING_HANDOFF=" + json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
