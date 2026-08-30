"""Checksum-gated Colab launcher that resumes Cane V1 from a protected last.pt."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


MANIFEST_PATH = Path("/content/cane-v1-resume-manifest.json")
RUN_NAME = "cane-v1-yolo26n-e30-img416"
LAST_PATH = Path(f"/content/yolo-pi-runs/{RUN_NAME}/weights/last.pt")
LOG_PATH = Path("/content/cane-v1-resume-training-pinned.log")
PID_PATH = Path("/content/cane-v1-resume.pid")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if not MANIFEST_PATH.exists():
    raise FileNotFoundError(MANIFEST_PATH)
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
CHECKPOINT = Path(str(manifest["checkpoint"]))
EXPECTED_SHA256 = str(manifest["sha256"])
COMPLETED_EPOCH = int(manifest["completed_epoch"])
if COMPLETED_EPOCH <= 0:
    raise ValueError("completed_epoch must be positive")

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)
actual_sha = sha256(CHECKPOINT)
if actual_sha != EXPECTED_SHA256:
    raise RuntimeError(f"resume checkpoint checksum mismatch: {actual_sha}")

LAST_PATH.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(CHECKPOINT, LAST_PATH)

command = [
    sys.executable,
    "/content/colab_train_export.py",
    "--model",
    "yolo26n.pt",
    "--dataset-archive",
    "/content/cane-v1-training-640.tar",
    "--epochs",
    "30",
    "--patience",
    "8",
    "--imgsz",
    "416",
    "--batch",
    "32",
    "--workers",
    "4",
    "--cache",
    "false",
    "--name",
    RUN_NAME,
    "--export-backends",
    "onnx,ncnn",
    "--resume",
]
environment = dict(os.environ)
environment["PYTHONUNBUFFERED"] = "1"
log_handle = LOG_PATH.open("a", encoding="utf-8", buffering=1)
log_handle.write(f"RESUME_CHECKPOINT_SHA256={actual_sha}\n")
log_handle.write(f"LAUNCH={command!r}\n")
log_handle.flush()
process = subprocess.Popen(
    command,
    stdout=log_handle,
    stderr=subprocess.STDOUT,
    env=environment,
    start_new_session=True,
)
PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
print(
    {
        "resume_checkpoint": str(CHECKPOINT),
        "sha256": actual_sha,
        "completed_epoch": COMPLETED_EPOCH,
        "installed_as": str(LAST_PATH),
        "pid": process.pid,
        "log": str(LOG_PATH),
        "launch": command,
        "detached": True,
    },
    flush=True,
)
