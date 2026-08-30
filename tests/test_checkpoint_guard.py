import argparse
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch


def load_guard_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "colab_checkpoint_guard.py"
    spec = importlib.util.spec_from_file_location("colab_checkpoint_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_once_reports_a_current_last_checkpoint_refresh(tmp_path, monkeypatch):
    guard = load_guard_module()

    def fake_download(_session, remote, destination):
        if remote.endswith("yolo-pi-colab-artifacts.zip"):
            return "final archive is not ready"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(remote.encode("utf-8"))
        return None

    def fake_checkpoint_info(path):
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": guard.sha256(path),
            "epoch_zero_based": 14,
            "completed_epoch": 15,
            "best_fitness": 0.41313,
            "has_optimizer": True,
            "class_names": guard.EXPECTED_NAMES,
        }

    monkeypatch.setattr(guard, "download", fake_download)
    monkeypatch.setattr(guard, "checkpoint_info", fake_checkpoint_info)
    state = {}
    finished, refreshed_last = guard.guard_once("session", tmp_path, state)

    assert not finished
    assert refreshed_last
    assert state["status"] == "guarding"
    assert state["errors"] == {}
    assert state["latest_last"]["completed_epoch"] == 15
    assert Path(state["latest_last"]["snapshot"]).exists()


def test_repeated_failed_current_polls_stop_with_existing_snapshot(tmp_path, monkeypatch):
    guard = load_guard_module()
    (tmp_path / "state.json").write_text(
        json.dumps({"latest_last": {"completed_epoch": 15}}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(guard, "guard_once", lambda *_args: (False, False))
    monkeypatch.setattr(guard.time, "sleep", lambda _seconds: None)
    args = argparse.Namespace(
        output=tmp_path,
        session="missing-session",
        once=False,
        interval_seconds=30,
        max_consecutive_failures=2,
    )

    assert guard.run_guard(args) == 1
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["consecutive_failures"] == 2
    assert state["status"] == "stopped_after_repeated_failures_with_latest_local_snapshot"


def test_verified_final_archive_supplies_its_own_validation_summary(tmp_path):
    guard = load_guard_module()
    archive = tmp_path / "artifacts.zip"
    expected_summary = b'{"status":"complete"}\n'
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("run/validation-summary.json", expected_summary)
        handle.writestr("run/weights/best.pt", b"best")
        handle.writestr("run/weights/last.pt", b"last")

    verification = guard.verify_zip(archive)
    member = verification["required_members"]["validation-summary.json"]
    destination = tmp_path / "validation-summary.json"
    guard.extract_verified_zip_member(archive, member, destination)

    assert verification["zip_test"] == "passed"
    assert destination.read_bytes() == expected_summary


def test_final_archive_promotes_hash_matched_best_and_last(tmp_path, monkeypatch):
    guard = load_guard_module()
    source_checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "epoch": 29,
            "ema": None,
            "model": SimpleNamespace(names=guard.EXPECTED_NAMES),
            "optimizer": None,
            "best_fitness": 0.5,
        },
        source_checkpoint,
    )
    summary = json.dumps(
        {"best_weights": {"sha256": guard.sha256(source_checkpoint)}}
    ).encode("utf-8")
    source_archive = tmp_path / "source-artifacts.zip"
    with zipfile.ZipFile(source_archive, "w") as handle:
        handle.writestr("run/validation-summary.json", summary)
        handle.write(source_checkpoint, "run/weights/best.pt")
        handle.write(source_checkpoint, "run/weights/last.pt")

    def fake_download(_session, remote, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if remote.endswith("yolo-pi-colab-artifacts.zip"):
            shutil.copy2(source_archive, destination)
        elif remote.endswith("weights/best.pt") or remote.endswith("weights/last.pt"):
            shutil.copy2(source_checkpoint, destination)
        else:
            destination.write_text("sidecar\n", encoding="utf-8")
        return None

    monkeypatch.setattr(guard, "download", fake_download)
    state = {}
    finished, refreshed_last = guard.guard_once("session", tmp_path / "guard", state)

    assert finished and refreshed_last
    assert state["status"] == "final_archive_downloaded"
    assert state["final_best"]["sha256"] == guard.sha256(source_checkpoint)
    assert state["final_best"]["completed_epoch"] == 30
    assert (tmp_path / "guard" / "final" / "best.pt").exists()
    assert (tmp_path / "guard" / "final" / "last.pt").exists()


def test_checkpoint_info_recovers_completed_epoch_from_stripped_history(tmp_path):
    guard = load_guard_module()
    checkpoint = tmp_path / "stripped.pt"
    torch.save(
        {
            "epoch": -1,
            "ema": None,
            "model": SimpleNamespace(names=guard.EXPECTED_NAMES),
            "optimizer": None,
            "best_fitness": 0.5,
            "train_results": {"epoch": [28, 29, 30]},
        },
        checkpoint,
    )

    info = guard.checkpoint_info(checkpoint)
    assert info["epoch_zero_based"] == -1
    assert info["completed_epoch"] == 30
    assert not info["has_optimizer"]
