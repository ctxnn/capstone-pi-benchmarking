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
