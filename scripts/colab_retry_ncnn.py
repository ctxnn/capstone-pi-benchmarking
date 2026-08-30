#!/usr/bin/env python3
"""Retry NCNN export from a live Colab checkpoint with explicit YOLO26 fallback."""

import json
import shutil
from pathlib import Path

from ultralytics import YOLO


checkpoint = Path("/content/yolo-pi-runs/gpu-smoke/weights/best.pt")
if not checkpoint.exists():
    raise FileNotFoundError(checkpoint)

fp16_checkpoint = checkpoint.with_name("best-fp16.pt")
shutil.copy2(checkpoint, fp16_checkpoint)
model = YOLO(str(fp16_checkpoint))
exported = Path(
    model.export(
        format="ncnn",
        imgsz=320,
        batch=1,
        device="cpu",
        end2end=False,
        quantize=16,
    )
)
if not exported.exists():
    raise FileNotFoundError(exported)
print(
    "NCNN_RETRY_JSON="
    + json.dumps({"path": str(exported), "end2end": False, "quantize": 16})
)
