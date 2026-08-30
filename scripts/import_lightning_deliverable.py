#!/usr/bin/env python3
"""Validate and atomically import the completed Lightning Cane V1 deliverable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_member(member: zipfile.ZipInfo) -> None:
    path = PurePosixPath(member.filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP path: {member.filename}")
    unix_mode = member.external_attr >> 16
    if stat.S_ISLNK(unix_mode):
        raise ValueError(f"ZIP symlink is not allowed: {member.filename}")


def model_info(path: Path) -> Dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    names = {int(index): str(name) for index, name in getattr(model, "names", {}).items()}
    if names != EXPECTED_NAMES:
        raise ValueError(f"unexpected checkpoint class taxonomy: {names}")
    epoch_zero_based = int(checkpoint.get("epoch", -1))
    completed_epoch = epoch_zero_based + 1
    if completed_epoch <= 0:
        history_epochs = checkpoint.get("train_results", {}).get("epoch", [])
        if history_epochs:
            completed_epoch = int(history_epochs[-1])
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "completed_epoch": completed_epoch,
        "has_optimizer": checkpoint.get("optimizer") is not None,
        "class_names": names,
    }


def validate_summary(summary: Dict[str, Any], best: Path, last: Path) -> None:
    if summary.get("status") != "complete":
        raise ValueError("Lightning summary status is not complete")
    if summary.get("training", {}).get("target_epochs") != 30:
        raise ValueError("Lightning summary target_epochs is not 30")
    if summary.get("training", {}).get("imgsz") != 416:
        raise ValueError("Lightning summary imgsz is not 416")
    quality = summary.get("quality_metrics", {})
    required_metrics = {"precision", "recall", "map50", "map50_95", "fitness"}
    for split in ("val", "test"):
        missing = required_metrics - set(quality.get(split, {}))
        if missing:
            raise ValueError(f"Lightning summary {split} metrics missing {sorted(missing)}")
    expected_best = summary.get("deliverables", {}).get("best", {}).get("sha256")
    expected_last = summary.get("deliverables", {}).get("last", {}).get("sha256")
    if sha256(best) != expected_best:
        raise ValueError("best.pt does not match Lightning summary SHA-256")
    if sha256(last) != expected_last:
        raise ValueError("last.pt does not match Lightning summary SHA-256")


def import_archive(archive_path: Path, output: Path) -> Dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing import: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"corrupt ZIP member: {bad_member}")
        members = archive.infolist()
        for member in members:
            validate_member(member)
        names = {member.filename for member in members}
        required = {
            "deliverables/best.pt",
            "deliverables/last.pt",
            "deliverables/results.csv",
            "deliverables/args.yaml",
            "deliverables/latest-resumable.pt",
            "deliverables/LATEST_RESUMABLE.json",
            "deliverables/validation-summary.json",
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"Lightning ZIP missing required members: {missing}")

        temporary_root = Path(
            tempfile.mkdtemp(prefix=".lightning-import-", dir=output.parent)
        )
        try:
            archive.extractall(temporary_root)
            extracted = temporary_root / "deliverables"
            best = extracted / "best.pt"
            last = extracted / "last.pt"
            latest_resumable = extracted / "latest-resumable.pt"
            summary = json.loads(
                (extracted / "validation-summary.json").read_text(encoding="utf-8")
            )
            latest_receipt = json.loads(
                (extracted / "LATEST_RESUMABLE.json").read_text(encoding="utf-8")
            )
            validate_summary(summary, best, last)
            best_info = model_info(best)
            last_info = model_info(last)
            resume_info = model_info(latest_resumable)
            best_info["path"] = str((output / "best.pt").resolve())
            last_info["path"] = str((output / "last.pt").resolve())
            resume_info["path"] = str((output / "latest-resumable.pt").resolve())
            expected_resume = summary.get("deliverables", {}).get(
                "latest_resumable", {}
            )
            receipt_resume = latest_receipt.get("checkpoint", {})
            expected_resume_sha = expected_resume.get("sha256") or receipt_resume.get(
                "sha256"
            )
            if resume_info["sha256"] != expected_resume_sha:
                raise ValueError(
                    "latest-resumable.pt does not match its Lightning SHA-256 receipt"
                )
            if receipt_resume.get("sha256") != resume_info["sha256"]:
                raise ValueError("LATEST_RESUMABLE.json checkpoint hash mismatch")
            if int(receipt_resume.get("completed_epoch", -1)) != resume_info[
                "completed_epoch"
            ]:
                raise ValueError("LATEST_RESUMABLE.json completed epoch mismatch")
            if not resume_info["has_optimizer"]:
                raise ValueError("latest-resumable.pt has no optimizer state")
            if resume_info["completed_epoch"] < 23:
                raise ValueError(
                    "latest-resumable.pt does not contain a post-epoch-22 update"
                )
            receipt = {
                "schema_version": 1,
                "status": "accepted",
                "archive": {
                    "source": str(archive_path.resolve()),
                    "bytes": archive_path.stat().st_size,
                    "sha256": sha256(archive_path),
                    "members": len(members),
                },
                "best": best_info,
                "last": last_info,
                "latest_resumable": resume_info,
                "quality_metrics": summary["quality_metrics"],
                "training": summary["training"],
            }
            normalized_summary = {
                **summary,
                "schema_version": 2,
                "source_schema_version": summary.get("schema_version"),
                "best_weights": {
                    "path": str((output / "best.pt").resolve()),
                    "bytes": best.stat().st_size,
                    "sha256": best_info["sha256"],
                },
                "last_weights": {
                    "path": str((output / "last.pt").resolve()),
                    "bytes": last.stat().st_size,
                    "sha256": last_info["sha256"],
                },
                "latest_resumable": resume_info,
            }
            atomic_json(extracted / "training-summary.json", normalized_summary)
            atomic_json(extracted / "local-import-receipt.json", receipt)
            os.replace(extracted, output)
            temporary_root.rmdir()
        except Exception:
            import shutil

            shutil.rmtree(temporary_root, ignore_errors=True)
            raise
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "artifacts/training/lightning-inbox/cane-v1-lightning-deliverable.zip"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/training/lightning-final")
    )
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    receipt = import_archive(args.archive, args.output)
    print("LIGHTNING_IMPORT_ACCEPTED=" + json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
