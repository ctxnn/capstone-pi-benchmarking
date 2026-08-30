import json
import zipfile
from types import SimpleNamespace

import pytest
import torch

from scripts import import_lightning_deliverable as importer


def _write_checkpoint(path, completed_epoch=30, optimizer=False):
    torch.save(
        {
            "epoch": -1,
            "optimizer": {"state": {}} if optimizer else None,
            "model": SimpleNamespace(names=importer.EXPECTED_NAMES),
            "train_results": {"epoch": list(range(1, completed_epoch + 1))},
        },
        path,
    )


def _make_archive(tmp_path, wrong_best_hash=False, legacy_summary=False):
    source = tmp_path / "source" / "deliverables"
    source.mkdir(parents=True)
    best = source / "best.pt"
    last = source / "last.pt"
    latest_resumable = source / "latest-resumable.pt"
    _write_checkpoint(best)
    _write_checkpoint(last)
    _write_checkpoint(latest_resumable, optimizer=True)
    latest_receipt = {
        "checkpoint": {
            "completed_epoch": 30,
            "sha256": importer.sha256(latest_resumable),
        }
    }
    (source / "LATEST_RESUMABLE.json").write_text(
        json.dumps(latest_receipt), encoding="utf-8"
    )
    (source / "results.csv").write_text("epoch,metric\n30,0.5\n", encoding="utf-8")
    (source / "args.yaml").write_text("epochs: 30\n", encoding="utf-8")
    metrics = {
        "precision": 0.7,
        "recall": 0.6,
        "map50": 0.65,
        "map50_95": 0.45,
        "fitness": 0.45,
    }
    summary = {
        "status": "complete",
        "training": {"target_epochs": 30, "imgsz": 416},
        "quality_metrics": {"val": metrics, "test": metrics},
        "deliverables": {
            "best": {
                "sha256": "wrong" if wrong_best_hash else importer.sha256(best)
            },
            "last": {"sha256": importer.sha256(last)},
            **(
                {}
                if legacy_summary
                else {
                    "latest_resumable": {
                        "sha256": importer.sha256(latest_resumable),
                        "completed_epoch": 30,
                        "optimizer_present": True,
                    }
                }
            ),
        },
    }
    (source / "validation-summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    archive = tmp_path / "deliverable.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(source.iterdir()):
            handle.write(path, f"deliverables/{path.name}")
    return archive


def test_import_accepts_complete_hash_matched_deliverable(tmp_path):
    archive = _make_archive(tmp_path)
    output = tmp_path / "accepted"

    receipt = importer.import_archive(archive, output)

    assert receipt["status"] == "accepted"
    assert receipt["best"]["completed_epoch"] == 30
    assert receipt["best"]["class_names"] == importer.EXPECTED_NAMES
    assert receipt["latest_resumable"]["has_optimizer"] is True
    assert (output / "best.pt").is_file()
    assert (output / "local-import-receipt.json").is_file()
    normalized = json.loads((output / "training-summary.json").read_text())
    assert normalized["best_weights"]["sha256"] == receipt["best"]["sha256"]


def test_import_rejects_summary_checkpoint_hash_mismatch(tmp_path):
    archive = _make_archive(tmp_path, wrong_best_hash=True)

    with pytest.raises(ValueError, match="best.pt does not match"):
        importer.import_archive(archive, tmp_path / "rejected")


def test_import_rejects_unsafe_zip_member(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")

    with pytest.raises(ValueError, match="unsafe ZIP path"):
        importer.import_archive(archive, tmp_path / "rejected")


def test_import_rejects_package_without_latest_recovery_checkpoint(tmp_path):
    complete = _make_archive(tmp_path)
    incomplete = tmp_path / "missing-recovery.zip"
    with zipfile.ZipFile(complete) as source, zipfile.ZipFile(incomplete, "w") as target:
        for member in source.infolist():
            if member.filename != "deliverables/latest-resumable.pt":
                target.writestr(member, source.read(member.filename))

    with pytest.raises(ValueError, match="latest-resumable.pt"):
        importer.import_archive(incomplete, tmp_path / "rejected")


def test_import_accepts_earlier_summary_when_atomic_resume_receipt_matches(tmp_path):
    archive = _make_archive(tmp_path, legacy_summary=True)

    receipt = importer.import_archive(archive, tmp_path / "accepted-legacy")

    assert receipt["latest_resumable"]["completed_epoch"] == 30
    assert receipt["latest_resumable"]["has_optimizer"] is True
