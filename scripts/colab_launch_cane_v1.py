"""Colab-side launcher for the reviewed full Cane V1 fine-tune."""

import os
import subprocess
import sys
from pathlib import Path


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
    "cane-v1-yolo26n-e30-img416",
    "--export-backends",
    "onnx,ncnn",
]
print({"launch": command}, flush=True)
log_path = Path("/content/cane-v1-training.log")
environment = dict(os.environ)
environment["PYTHONUNBUFFERED"] = "1"
with log_path.open("a", encoding="utf-8", buffering=1) as log:
    log.write(f"LAUNCH={command!r}\n")
    completed = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
    )
if completed.returncode != 0:
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
    print(tail, flush=True)
    raise subprocess.CalledProcessError(completed.returncode, command)
