from types import SimpleNamespace

import pytest
import torch

from scripts import prepare_lightning_handoff as handoff


def test_materialize_is_idempotent_and_refuses_mismatch(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"verified")
    destination = tmp_path / "upload" / "source.bin"

    assert handoff.materialize(source, destination) in {"hardlink", "copy"}
    assert handoff.materialize(source, destination) == "already-present"
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"changed")
    replacement.replace(destination)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        handoff.materialize(source, destination)


def test_validate_source_inputs_checks_epoch_optimizer_and_taxonomy(
    tmp_path, monkeypatch
):
    archive = tmp_path / "dataset.tar"
    archive.write_bytes(b"dataset")
    checkpoint = tmp_path / "last.pt"
    torch.save(
        {
            "epoch": 21,
            "optimizer": {"state": {}},
            "ema": SimpleNamespace(names=handoff.EXPECTED_NAMES),
        },
        checkpoint,
    )
    monkeypatch.setattr(handoff, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(handoff, "ARCHIVE_SHA256", handoff.sha256(archive))
    monkeypatch.setattr(handoff, "CHECKPOINT_SHA256", handoff.sha256(checkpoint))

    record = handoff.validate_source_inputs(archive, checkpoint)

    assert record["checkpoint"]["completed_epoch"] == 22
    assert record["checkpoint"]["optimizer_present"] is True


def test_validate_scaffold_rejects_stale_canonical_copy(tmp_path):
    root = tmp_path / "root"
    output = root / "lightning-ai-upload"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    output.mkdir()
    for name in handoff.SCAFFOLD:
        (output / name).write_text(name, encoding="utf-8")
    (root / "scripts/lightning_train_resume.py").write_text(
        "different", encoding="utf-8"
    )
    (root / "docs/lightning-ai-training.md").write_text(
        "LIGHTNING-INSTRUCTIONS.md", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="stale"):
        handoff.validate_scaffold(output, root)
