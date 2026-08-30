import json
import tarfile
import zipfile
from types import SimpleNamespace

import torch

from scripts import lightning_train_resume as lightning


def _checkpoint(path, completed_epoch, names=None, optimizer=True):
    torch.save(
        {
            "epoch": completed_epoch - 1,
            "optimizer": {"state": {}} if optimizer else None,
            "ema": SimpleNamespace(
                names={
                    index: name
                    for index, name in enumerate(names or lightning.EXPECTED_NAMES)
                }
            ),
            "best_fitness": 0.4,
        },
        path,
    )


def test_safe_extract_rewrites_dataset_root(tmp_path):
    source = tmp_path / "source"
    root = source / "dataset"
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )
    archive = tmp_path / "dataset.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(root, arcname="dataset")

    data_yaml = lightning.safe_extract(archive, tmp_path / "extracted")

    assert data_yaml.is_file()
    assert f"path: {data_yaml.parent.resolve()}" in data_yaml.read_text(encoding="utf-8")


def test_select_resume_checkpoint_uses_newest_valid_snapshot(tmp_path):
    original = tmp_path / "input.pt"
    _checkpoint(original, 22)
    output = tmp_path / "output"
    snapshots = output / "checkpoint-snapshots"
    snapshots.mkdir(parents=True)
    _checkpoint(snapshots / "last-epoch-023.pt", 23)
    _checkpoint(snapshots / "last-epoch-024.pt", 24)

    result = lightning.select_resume_checkpoint(original, output, torch)

    assert result["selected"]["completed_epoch"] == 24
    assert result["selected"]["path"].endswith("last-epoch-024.pt")


def test_select_resume_checkpoint_rejects_non_resumable_and_wrong_taxonomy(tmp_path):
    original = tmp_path / "input.pt"
    _checkpoint(original, 22)
    output = tmp_path / "output"
    snapshots = output / "checkpoint-snapshots"
    snapshots.mkdir(parents=True)
    _checkpoint(snapshots / "last-epoch-023.pt", 23, optimizer=False)
    _checkpoint(snapshots / "last-epoch-024.pt", 24, names=["wrong"])

    result = lightning.select_resume_checkpoint(original, output, torch)

    assert result["selected"]["completed_epoch"] == 22
    assert len(result["rejected"]) == 2


def test_existing_completion_requires_matching_best_hash(tmp_path):
    output = tmp_path / "output"
    best = output / "deliverables" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"weights")
    receipt = {
        "deliverables": {
            "best": {"path": str(best), "sha256": lightning.sha256(best)}
        }
    }
    lightning.atomic_json(output / "LIGHTNING_COMPLETE.json", receipt)

    assert lightning.existing_completion(output) == receipt
    best.write_bytes(b"changed")
    try:
        lightning.existing_completion(output)
    except RuntimeError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("mismatched completion receipt was accepted")


def test_protect_resumable_checkpoint_writes_downloadable_bundle(tmp_path):
    source = tmp_path / "last.pt"
    _checkpoint(source, 24)
    output = tmp_path / "output"

    result = lightning.protect_resumable_checkpoint(
        source, 24, output, torch, reason="test"
    )

    snapshot = output / "checkpoint-snapshots/last-epoch-024.pt"
    archive = output / lightning.LATEST_RECOVERY_ZIP
    assert snapshot.is_file()
    assert result["checkpoint"]["optimizer_present"] is True
    assert result["recovery_bundle"]["completed_epoch"] == 24
    assert result["recovery_bundle"]["archive"]["sha256"] == lightning.sha256(
        archive
    )
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == {
            "checkpoint-snapshots/last-epoch-024.pt",
            "LATEST_RESUMABLE.json",
        }
        bundle.extractall(tmp_path / "restored")
    restored = tmp_path / "restored/checkpoint-snapshots/last-epoch-024.pt"
    assert lightning.checkpoint_metadata(restored, torch)["completed_epoch"] == 24


def test_protect_resumable_checkpoint_rejects_epoch_mismatch(tmp_path):
    source = tmp_path / "last.pt"
    _checkpoint(source, 23)
    output = tmp_path / "output"
    existing = output / "checkpoint-snapshots/last-epoch-024.pt"
    existing.parent.mkdir(parents=True)
    _checkpoint(existing, 24)
    existing_hash = lightning.sha256(existing)

    try:
        lightning.protect_resumable_checkpoint(
            source, 24, output, torch, reason="test"
        )
    except RuntimeError as exc:
        assert "epoch mismatch" in str(exc)
    else:
        raise AssertionError("epoch-mismatched recovery checkpoint was accepted")
    assert lightning.sha256(existing) == existing_hash
