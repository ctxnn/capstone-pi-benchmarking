import json
from pathlib import Path

import pytest

from scripts import build_pi_model_registry as registry


def test_resolves_transferred_export_by_name_and_verifies_tree(tmp_path):
    artifact_root = tmp_path / "artifacts/models/cane-v1"
    artifact_root.mkdir(parents=True)
    transferred = artifact_root / "best.onnx"
    transferred.write_bytes(b"portable-model")
    entry = {
        "path": "/old-host/private/export/best.onnx",
        "tree_sha256": registry.tree_sha256(transferred),
    }

    assert registry.resolve_export_path(entry, artifact_root) == transferred.resolve()


def test_rejects_transferred_export_with_wrong_tree_hash(tmp_path):
    artifact_root = tmp_path / "exports"
    artifact_root.mkdir()
    (artifact_root / "best_ncnn_model").mkdir()
    (artifact_root / "best_ncnn_model/model.ncnn.bin").write_bytes(b"changed")
    entry = {
        "path": "/old-host/best_ncnn_model",
        "tree_sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="tree checksum mismatch"):
        registry.resolve_export_path(entry, artifact_root)

def _input_size_manifest(tmp_path, source_sha="a" * 64, quality_imgsz=416):
    root = tmp_path / "artifacts/models/cane-v1-fast"
    export = root / "ov-fp32-416/best_openvino_model"
    export.mkdir(parents=True)
    (export / "best.bin").write_bytes(b"weights")
    manifest = root / "export-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_model": {"sha256": source_sha},
                "exports": {
                    "openvino-416": {
                        "status": "complete",
                        "path": "ov-fp32-416/best_openvino_model",
                        "tree_sha256": registry.tree_sha256(export),
                        "imgsz": 416,
                        "quality_metrics": {"split": "test", "fraction": 1.0, "imgsz": quality_imgsz, "map50": 0.62},
                    }
                },
            }
        )
    )
    return manifest


def test_registers_input_size_export_with_its_own_quality(tmp_path):
    manifest = _input_size_manifest(tmp_path)

    entries = registry.input_size_entries(manifest, "a" * 64, tmp_path)

    entry = entries["cane-v1-openvino-fp32-416"]
    assert entry["path"] == "artifacts/models/cane-v1-fast/ov-fp32-416/best_openvino_model"
    assert entry["quality_metrics"]["imgsz"] == 416


def test_rejects_input_size_export_from_another_checkpoint(tmp_path):
    manifest = _input_size_manifest(tmp_path, source_sha="b" * 64)

    with pytest.raises(ValueError, match="selected final best.pt"):
        registry.input_size_entries(manifest, "a" * 64, tmp_path)


def test_rejects_quality_measured_at_another_input_size(tmp_path):
    manifest = _input_size_manifest(tmp_path, quality_imgsz=640)

    with pytest.raises(ValueError, match="export size"):
        registry.input_size_entries(manifest, "a" * 64, tmp_path)
