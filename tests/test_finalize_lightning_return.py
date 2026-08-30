import json

import pytest

from scripts import finalize_lightning_return as finalizer


def test_export_status_reports_exact_required_matrix():
    manifest = {
        "exports": {
            "onnx": {"status": "complete"},
            "openvino": {"status": "smoke_failed"},
            "mnn": {"status": "complete"},
            "ncnn": {"status": "complete"},
        }
    }

    assert finalizer.export_status(manifest) == {
        "onnx": "complete",
        "openvino": "smoke_failed",
        "mnn": "complete",
        "ncnn": "complete",
        "litert": "missing",
    }


def test_validate_history_requires_contiguous_exact_final_epoch():
    finalizer.validate_history(
        {
            "accepted_epoch_start": 1,
            "accepted_epoch_end": 30,
            "accepted_epoch_count": 30,
        },
        30,
    )
    with pytest.raises(ValueError, match="end at"):
        finalizer.validate_history(
            {
                "accepted_epoch_start": 1,
                "accepted_epoch_end": 29,
                "accepted_epoch_count": 29,
            },
            30,
        )


def test_validate_comparison_binds_full_split_and_both_hashes():
    comparison = {
        "split": "test",
        "fraction": 1.0,
        "imgsz": 640,
        "evaluated_image_count": 1515,
        "models": {
            "v0-pytorch-fp32": {"sha256": "baseline"},
            "cane-v1-pytorch-fp32": {"sha256": "fine"},
        },
    }
    finalizer.validate_comparison(comparison, "baseline", "fine")
    comparison["evaluated_image_count"] = 1514
    with pytest.raises(ValueError, match="1,515"):
        finalizer.validate_comparison(comparison, "baseline", "fine")


def test_validate_import_rejects_changed_model(tmp_path):
    archive = tmp_path / "return.zip"
    archive.write_bytes(b"archive")
    imported = tmp_path / "accepted"
    imported.mkdir()
    records = {}
    for key, filename in (
        ("best", "best.pt"),
        ("last", "last.pt"),
        ("latest_resumable", "latest-resumable.pt"),
    ):
        model = imported / filename
        model.write_bytes(key.encode())
        records[key] = {"sha256": finalizer.sha256(model)}
    receipt = {
        "archive": {"sha256": finalizer.sha256(archive)},
        **records,
    }
    (imported / "local-import-receipt.json").write_text(json.dumps(receipt))

    assert finalizer.validate_import(archive, imported) == receipt
    (imported / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="best.pt checksum changed"):
        finalizer.validate_import(archive, imported)
