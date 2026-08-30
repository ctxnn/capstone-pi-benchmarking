#!/usr/bin/env python3
"""Persist Colab training checkpoints locally until final artifacts are safe."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional


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


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def download(session: str, remote: str, destination: Path) -> Optional[str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    incoming = destination.with_suffix(destination.suffix + ".download")
    incoming.unlink(missing_ok=True)
    completed = subprocess.run(
        ["colab", "download", "-s", session, remote, str(incoming)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0 or not incoming.exists():
        incoming.unlink(missing_ok=True)
        return (completed.stderr + "\n" + completed.stdout).strip()[-2000:]
    incoming.replace(destination)
    return None


def checkpoint_info(path: Path) -> Dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch_zero_based = int(checkpoint.get("epoch", -1))
    completed_epoch = epoch_zero_based + 1
    if completed_epoch <= 0:
        history_epochs = checkpoint.get("train_results", {}).get("epoch", [])
        if history_epochs:
            completed_epoch = int(history_epochs[-1])
    model = checkpoint.get("ema") or checkpoint.get("model")
    names = {int(index): str(name) for index, name in getattr(model, "names", {}).items()}
    if names != EXPECTED_NAMES:
        raise ValueError(f"unexpected checkpoint class taxonomy: {names}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "epoch_zero_based": epoch_zero_based,
        "completed_epoch": completed_epoch,
        "best_fitness": float(checkpoint.get("best_fitness", 0.0)),
        "has_optimizer": checkpoint.get("optimizer") is not None,
        "class_names": names,
    }


def snapshot_checkpoint(latest: Path, label: str, info: Dict[str, Any], snapshots: Path) -> Path:
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / (
        f"{label}-epoch-{info['completed_epoch']:03d}-{info['sha256'][:12]}.pt"
    )
    if not snapshot.exists():
        os.link(latest, snapshot)
    return snapshot


def verify_zip(path: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"corrupt ZIP member: {bad_member}")
        names = archive.namelist()
    required_markers = ("validation-summary.json", "weights/best.pt", "weights/last.pt")
    required_members = {
        marker: next((name for name in names if name.endswith(marker)), None)
        for marker in required_markers
    }
    missing = [marker for marker, member in required_members.items() if member is None]
    if missing:
        raise ValueError(f"final ZIP is missing required members: {missing}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "members": len(names),
        "zip_test": "passed",
        "required_members": required_members,
    }


def extract_verified_zip_member(archive_path: Path, member: str, destination: Path) -> None:
    """Atomically extract one already-verified member without trusting its path."""

    incoming = destination.with_suffix(destination.suffix + ".download")
    incoming.unlink(missing_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        incoming.write_bytes(archive.read(member))
    incoming.replace(destination)


def guard_once(session: str, output: Path, state: Dict[str, Any]) -> tuple[bool, bool]:
    remote_run = "/content/yolo-pi-runs/cane-v1-yolo26n-e30-img416"
    checkpoints = output / "checkpoints"
    snapshots = checkpoints / "snapshots"
    errors: Dict[str, str] = {}
    refreshed_last = False
    for label in ("best", "last"):
        latest = checkpoints / f"latest-{label}.pt"
        error = download(session, f"{remote_run}/weights/{label}.pt", latest)
        if error:
            errors[label] = error
            continue
        info = checkpoint_info(latest)
        info["snapshot"] = str(snapshot_checkpoint(latest, label, info, snapshots))
        state[f"latest_{label}"] = info
        if label == "last":
            refreshed_last = True

    sidecars = {
        "training_log": ("/content/cane-v1-resume-training-pinned.log", output / "training.log"),
        "results_csv": (f"{remote_run}/results.csv", output / "results.csv"),
        "args_yaml": (f"{remote_run}/args.yaml", output / "args.yaml"),
    }
    for name, (remote, local) in sidecars.items():
        error = download(session, remote, local)
        if error:
            errors[name] = error
        else:
            state[name] = {
                "path": str(local),
                "bytes": local.stat().st_size,
                "sha256": sha256(local),
            }

    final_zip = output / "yolo-pi-colab-artifacts.zip"
    final_error = download(session, "/content/yolo-pi-colab-artifacts.zip", final_zip)
    finished = False
    if final_error is None:
        final_archive = verify_zip(final_zip)
        state["final_archive"] = final_archive
        summary = output / "validation-summary.json"
        summary_member = str(final_archive["required_members"]["validation-summary.json"])
        extract_verified_zip_member(final_zip, summary_member, summary)
        summary_payload = json.loads(summary.read_text(encoding="utf-8"))
        state["validation_summary"] = {
            "path": str(summary),
            "sha256": sha256(summary),
            "source": "verified final archive",
            "archive_member": summary_member,
        }
        final_directory = output / "final"
        final_directory.mkdir(parents=True, exist_ok=True)
        for label in ("best", "last"):
            marker = f"weights/{label}.pt"
            member = str(final_archive["required_members"][marker])
            destination = final_directory / f"{label}.pt"
            extract_verified_zip_member(final_zip, member, destination)
            info = checkpoint_info(destination)
            info["source"] = "verified final archive"
            info["archive_member"] = member
            state[f"final_{label}"] = info
        expected_best_sha = summary_payload.get("best_weights", {}).get("sha256")
        if not expected_best_sha:
            raise ValueError("validation summary is missing best_weights.sha256")
        if state["final_best"]["sha256"] != expected_best_sha:
            raise ValueError("final best.pt does not match validation summary SHA-256")
        finished = True
    state.update(
        {
            "schema_version": 1,
            "session": session,
            "updated_at_epoch_ns": time.time_ns(),
            "status": "final_archive_downloaded" if finished else "guarding",
            "errors": errors,
        }
    )
    atomic_json(output / "state.json", state)
    return finished, refreshed_last


def run_guard(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_handle = (output / "guard.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"checkpoint guard already active for {output}", file=sys.stderr)
        return 2
    (output / "guard.pid").write_text(f"{os.getpid()}\n")
    state_path = output / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    consecutive_failures = 0
    while True:
        refreshed_last = False
        try:
            finished, refreshed_last = guard_once(args.session, output, state)
            consecutive_failures = 0 if refreshed_last else consecutive_failures + 1
            state["consecutive_failures"] = consecutive_failures
            atomic_json(state_path, state)
            if finished:
                return 0
        except Exception as exc:
            consecutive_failures += 1
            state.update(
                {
                    "schema_version": 1,
                    "session": args.session,
                    "updated_at_epoch_ns": time.time_ns(),
                    "status": "guard_error",
                    "guard_error_type": type(exc).__name__,
                    "guard_error": str(exc),
                    "consecutive_failures": consecutive_failures,
                }
            )
            atomic_json(state_path, state)
        if args.once:
            return 0 if refreshed_last else 1
        if consecutive_failures >= args.max_consecutive_failures:
            state["status"] = "stopped_after_repeated_failures_with_latest_local_snapshot"
            atomic_json(state_path, state)
            return 1
        time.sleep(args.interval_seconds)


def daemonize(log_path: Path) -> bool:
    """Fork once; return True in the parent and continue detached in the child."""

    pid = os.fork()
    if pid:
        print(f"checkpoint guard daemon pid={pid}")
        return True
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdin = open(os.devnull, "r")
    log = log_path.open("a", buffering=1)
    os.dup2(stdin.fileno(), sys.stdin.fileno())
    os.dup2(log.fileno(), sys.stdout.fileno())
    os.dup2(log.fileno(), sys.stderr.fileno())
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=120)
    parser.add_argument("--max-consecutive-failures", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 30:
        raise ValueError("interval must be at least 30 seconds")
    if args.max_consecutive_failures <= 0:
        raise ValueError("max consecutive failures must be positive")
    if args.daemon and daemonize(args.output.resolve() / "guard.log"):
        return 0
    return run_guard(args)


if __name__ == "__main__":
    raise SystemExit(main())
