#!/usr/bin/env python3
"""Run the checksum-gated post-training pipeline for a Lightning return ZIP.

This command is intentionally resumable. It records each completed or pending
stage and never overwrites an accepted import with a different archive/model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable


REQUIRED_EXPORTS = ("onnx", "openvino", "mnn", "ncnn", "litert")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def update_stage(
    state_path: Path, state: Dict[str, Any], name: str, **values: Any
) -> None:
    state.setdefault("stages", {})[name] = {
        **state.get("stages", {}).get(name, {}),
        **values,
        "updated_at_epoch_ns": time.time_ns(),
    }
    atomic_json(state_path, state)


def run_logged(
    command: list[str], log: Path, environment: Dict[str, str] | None = None
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN=" + " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\nRUN=" + repr(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        handle.write(f"EXIT={completed.returncode}\n")
    return completed.returncode


def export_status(manifest: Dict[str, Any]) -> Dict[str, Any]:
    entries = manifest.get("exports", {})
    return {
        name: entries.get(name, {}).get("status", "missing")
        for name in REQUIRED_EXPORTS
    }


def validate_import(
    archive: Path, imported: Path, expected_archive_sha: str | None = None
) -> Dict[str, Any]:
    receipt_path = imported / "local-import-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = load_json(receipt_path)
    archive_sha = sha256(archive)
    if receipt.get("archive", {}).get("sha256") != archive_sha:
        raise ValueError("accepted import does not match the supplied Lightning ZIP")
    if expected_archive_sha and archive_sha != expected_archive_sha:
        raise ValueError("Lightning ZIP changed since finalization state was created")
    for key, filename in (
        ("best", "best.pt"),
        ("last", "last.pt"),
        ("latest_resumable", "latest-resumable.pt"),
    ):
        model_path = imported / filename
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if sha256(model_path) != receipt[key]["sha256"]:
            raise ValueError(f"accepted {filename} checksum changed")
    return receipt


def validate_history(manifest: Dict[str, Any], final_epoch: int) -> None:
    if manifest.get("accepted_epoch_start") != 1:
        raise ValueError("consolidated history does not start at epoch 1")
    if manifest.get("accepted_epoch_end") != final_epoch:
        raise ValueError(
            "consolidated history does not end at the protected final epoch"
        )
    if manifest.get("accepted_epoch_count") != final_epoch:
        raise ValueError("consolidated history is not contiguous")


def validate_comparison(
    comparison: Dict[str, Any], baseline_sha: str, fine_tuned_sha: str
) -> None:
    if comparison.get("split") != "test" or comparison.get("fraction") != 1.0:
        raise ValueError("model comparison is not the complete test split")
    if comparison.get("imgsz") != 640:
        raise ValueError("model comparison is not at image size 640")
    if comparison.get("evaluated_image_count") != 1515:
        raise ValueError("model comparison did not resolve exactly 1,515 test images")
    models = comparison.get("models", {})
    if models.get("v0-pytorch-fp32", {}).get("sha256") != baseline_sha:
        raise ValueError("model comparison baseline hash mismatch")
    if models.get("cane-v1-pytorch-fp32", {}).get("sha256") != fine_tuned_sha:
        raise ValueError("model comparison fine-tuned hash mismatch")


def prepare_openvino_linux_bundle(
    export_manifest: Path, output: Path, sample: Path, root: Path
) -> Path | None:
    manifest = load_json(export_manifest)
    entry = manifest.get("exports", {}).get("openvino", {})
    if entry.get("status") == "complete" or not entry.get("path"):
        return None
    artifact = Path(entry["path"])
    if not artifact.exists():
        return None
    bundle = output / "openvino-linux-smoke"
    bundle.mkdir(parents=True, exist_ok=True)
    archive = bundle / "openvino-export.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(artifact, arcname="best_openvino_model")
    for source in (
        root / "scripts" / "smoke_export_artifact.py",
        root / "scripts" / "colab_openvino_smoke.py",
        sample,
    ):
        destination = bundle / source.name
        if not destination.exists() or sha256(destination) != sha256(source):
            destination.write_bytes(source.read_bytes())
    atomic_json(
        bundle / "bundle-manifest.json",
        {
            "schema_version": 1,
            "purpose": "Linux OpenVINO functional smoke only; not Pi performance evidence",
            "export_manifest": str(export_manifest.resolve()),
            "expected_tree_sha256": entry.get("tree_sha256"),
            "archive": {
                "path": str(archive.resolve()),
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
            },
            "sample_sha256": sha256(sample),
        },
    )
    return bundle


def all_complete(values: Iterable[str]) -> bool:
    return all(value == "complete" for value in values)


def history_checkpoint_paths(root: Path, imported: Path) -> list[Path]:
    """Return every immutable VM boundary needed for epochs 1 through 30."""

    return [
        root / "artifacts/training/checkpoints/cane-v1-best-epoch6.pt",
        root
        / "artifacts/training/checkpoint-guard/checkpoints/snapshots/last-epoch-015-cfeb7c9a252b.pt",
        root
        / "artifacts/training/checkpoint-guard/checkpoints/snapshots/last-epoch-022-0da75726f7a0.pt",
        imported / "latest-resumable.pt",
    ]


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
        "--imported", type=Path, default=Path("artifacts/training/lightning-final")
    )
    parser.add_argument(
        "--models", type=Path, default=Path("artifacts/models/cane-v1")
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("artifacts/training/finalization-state.json"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    archive = args.archive.resolve()
    imported = args.imported.resolve()
    models = args.models.resolve()
    state_path = args.state.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    state = load_json(state_path) if state_path.is_file() else {
        "schema_version": 1,
        "purpose": "resumable post-Lightning model finalization",
        "created_at_epoch_ns": time.time_ns(),
        "stages": {},
    }
    expected_archive_sha = state.get("lightning_archive_sha256")
    state["lightning_archive_sha256"] = sha256(archive)
    atomic_json(state_path, state)

    # 1. Import and normalize the Lightning package.
    if not (imported / "local-import-receipt.json").is_file():
        update_stage(state_path, state, "import", status="running")
        result = run_logged(
            [
                sys.executable,
                str(root / "scripts" / "import_lightning_deliverable.py"),
                "--archive",
                str(archive),
                "--output",
                str(imported),
            ],
            state_path.with_name("finalization-import.log"),
        )
        if result != 0:
            update_stage(state_path, state, "import", status="failed", exit_code=result)
            return 1
    receipt = validate_import(archive, imported, expected_archive_sha)
    update_stage(
        state_path,
        state,
        "import",
        status="complete",
        best_sha256=receipt["best"]["sha256"],
        protected_epoch=receipt["latest_resumable"]["completed_epoch"],
    )

    # 2. Consolidate the accepted multi-VM training trajectory.
    history_csv = imported.parent / "cane-v1-training-history.csv"
    history_manifest = imported.parent / "cane-v1-training-history.json"
    if not history_manifest.is_file():
        update_stage(state_path, state, "history", status="running")
        history_command = [
            sys.executable,
            str(root / "scripts" / "consolidate_training_history.py"),
        ]
        for checkpoint in history_checkpoint_paths(root, imported):
            history_command.extend(["--checkpoint", str(checkpoint)])
        history_command.extend(
            [
                "--output",
                str(history_csv),
                "--manifest",
                str(history_manifest),
            ]
        )
        result = run_logged(
            history_command,
            state_path.with_name("finalization-history.log"),
        )
        if result != 0:
            update_stage(state_path, state, "history", status="failed", exit_code=result)
            return 1
    history = load_json(history_manifest)
    validate_history(history, receipt["latest_resumable"]["completed_epoch"])
    update_stage(
        state_path,
        state,
        "history",
        status="complete",
        exit_code=0,
        accepted_epoch_end=history["accepted_epoch_end"],
        manifest=str(history_manifest),
    )

    # 3. Export and smoke the exact 640/FP32 backend matrix.
    export_manifest = models / "export-manifest.json"
    if not export_manifest.is_file():
        update_stage(state_path, state, "exports", status="running")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(root / "src")
        result = run_logged(
            [
                sys.executable,
                str(root / "scripts" / "export_pi_backends.py"),
                "--model",
                str(imported / "best.pt"),
                "--output",
                str(models),
                "--sample",
                str(root / "artifacts/models/bus.jpg"),
                "--imgsz",
                "640",
            ],
            state_path.with_name("finalization-exports.log"),
            environment,
        )
        update_stage(state_path, state, "exports", last_exit_code=result)
    exports = load_json(export_manifest)
    if exports.get("source_model", {}).get("sha256") != receipt["best"]["sha256"]:
        raise ValueError("export manifest source does not match accepted best.pt")
    statuses = export_status(exports)
    export_stage = "complete" if all_complete(statuses.values()) else "needs_linux_smoke"
    bundle = prepare_openvino_linux_bundle(
        export_manifest, models, root / "artifacts/models/bus.jpg", root
    )
    update_stage(
        state_path,
        state,
        "exports",
        status=export_stage,
        formats=statuses,
        linux_smoke_bundle=str(bundle) if bundle else None,
    )

    # 4. Compare M1 and M2 on the same complete held-out split at 640.
    comparison_path = models / "model-comparison-640.json"
    baseline = root / "artifacts/models/yolo26n.pt"
    if not comparison_path.is_file():
        update_stage(state_path, state, "quality_comparison", status="running")
        result = run_logged(
            [
                sys.executable,
                str(root / "scripts" / "evaluate_model_comparison.py"),
                "--data",
                str(root / "data/processed/cane-v1-training-640/data.yaml"),
                "--baseline",
                str(baseline),
                "--fine-tuned",
                str(imported / "best.pt"),
                "--output",
                str(comparison_path),
                "--imgsz",
                "640",
                "--batch",
                str(args.batch),
                "--workers",
                str(args.workers),
                "--device",
                args.device,
            ],
            state_path.with_name("finalization-quality.log"),
        )
        if result != 0:
            update_stage(
                state_path, state, "quality_comparison", status="failed", exit_code=result
            )
            return 1
    comparison = load_json(comparison_path)
    validate_comparison(
        comparison, sha256(baseline), receipt["best"]["sha256"]
    )
    update_stage(
        state_path,
        state,
        "quality_comparison",
        status="complete",
        evaluated_image_count=1515,
        output=str(comparison_path),
    )

    # 5. Build the production registry only after every runtime smoke is complete.
    registry = root / "configs/pi-model-registry.json"
    if all_complete(statuses.values()):
        if not registry.is_file():
            update_stage(state_path, state, "registry", status="running")
            result = run_logged(
                [
                    sys.executable,
                    str(root / "scripts" / "build_pi_model_registry.py"),
                    "--training-summary",
                    str(imported / "training-summary.json"),
                    "--quality-comparison",
                    str(comparison_path),
                    "--export-manifest",
                    str(export_manifest),
                    "--baseline",
                    str(baseline),
                    "--output",
                    str(registry),
                ],
                state_path.with_name("finalization-registry.log"),
            )
            if result != 0:
                update_stage(
                    state_path, state, "registry", status="failed", exit_code=result
                )
                return 1
        update_stage(
            state_path, state, "registry", status="complete", output=str(registry)
        )
    else:
        update_stage(
            state_path,
            state,
            "registry",
            status="pending_export_smoke",
            incomplete_formats=[name for name, status_value in statuses.items() if status_value != "complete"],
        )

    state["status"] = (
        "complete" if state["stages"]["registry"]["status"] == "complete" else "awaiting_linux_export_smoke"
    )
    state["updated_at_epoch_ns"] = time.time_ns()
    atomic_json(state_path, state)
    print("FINALIZATION_STATE=" + json.dumps(state, sort_keys=True), flush=True)
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
