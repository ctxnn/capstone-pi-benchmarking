"""Colab-side Linux smoke for a checksum-bound OpenVINO export bundle."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path


subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "ultralytics==8.4.132",
        "openvino==2026.3.1",
    ]
)

archive = Path("/content/openvino-preflight.tar.gz")
destination = Path("/content/openvino-preflight")
destination.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive) as handle:
    root = destination.resolve()
    for member in handle.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"unsafe archive member: {member.name}")
    handle.extractall(destination, filter="data")

subprocess.check_call(
    [
        sys.executable,
        "/content/smoke_export_artifact.py",
        "--artifact",
        "/content/openvino-preflight/best_openvino_model",
        "--sample",
        "/content/bus.jpg",
        "--output",
        "/content/openvino-preflight-smoke.json",
        "--format",
        "openvino",
        "--imgsz",
        "640",
        "--precision",
        "FP32",
    ]
)
