#!/usr/bin/env python3
"""Show live IMX219 detections on the Raspberry Pi desktop."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO

from yolo_pi.picamera2_source import Picamera2LatestFrameSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/models/cane-v1/best_openvino_model"),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-num", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"model not found: {args.model}")

    model = YOLO(str(args.model), task="detect")
    source = Picamera2LatestFrameSource(
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        camera_num=args.camera_num,
        pixel_format="RGB888",
    )
    window_name = "Cane V1 live detection - Q or Esc to quit"
    smoothed_fps: float | None = None

    try:
        source.start()
        while True:
            frame = source.next_frame(timeout_s=10.0)
            started = perf_counter()
            result = model.predict(
                source=frame.payload,
                imgsz=args.imgsz,
                conf=args.confidence,
                device=args.device,
                verbose=False,
            )[0]
            elapsed = perf_counter() - started
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            smoothed_fps = (
                current_fps
                if smoothed_fps is None
                else 0.85 * smoothed_fps + 0.15 * current_fps
            )

            annotated = result.plot()
            cv2.putText(
                annotated,
                f"Inference: {elapsed * 1000:.1f} ms | {smoothed_fps:.2f} FPS",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
