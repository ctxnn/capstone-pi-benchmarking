#!/usr/bin/env python3
"""Live IMX219 detection preview with optional headless latency reporting.

The camera runs on its own thread and always hands over the newest frame, so
inference never works through a stale backlog. ``--no-display`` skips drawing
and windowing entirely, which is the deployment mode for the cane and the mode
to measure when the screen is not the subject of the test.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from time import perf_counter_ns

from yolo_pi.pi_benchmark import configure_threads
from yolo_pi.picamera2_source import Picamera2LatestFrameSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/models/cane-v1-fast/ov-fp32-416/best_openvino_model"),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-num", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=0, help="override the model's own input size")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-display", action="store_true", help="headless: no window, no drawing")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = until Q)")
    parser.add_argument("--warmup-frames", type=int, default=5, help="excluded from the summary")
    return parser.parse_args()


def model_imgsz(model: Path) -> int:
    import yaml

    metadata = model / "metadata.yaml"
    if metadata.is_file():
        loaded = yaml.safe_load(metadata.read_text())
        if loaded and "imgsz" in loaded:
            return int(list(loaded["imgsz"])[0])
    raise FileNotFoundError(f"cannot determine input size for {model}; pass --imgsz")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def summarize(name: str, values: list[float], digits: int = 1) -> str:
    return (
        f"{name}: mean {statistics.mean(values):.{digits}f} ms  "
        f"p50 {percentile(values, 0.50):.{digits}f}  "
        f"p95 {percentile(values, 0.95):.{digits}f}  "
        f"max {max(values):.{digits}f}"
    )


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"model not found: {args.model}")

    configure_threads(args.threads)
    from ultralytics import YOLO

    imgsz = args.imgsz or model_imgsz(args.model)
    model = YOLO(str(args.model), task="detect")
    source = Picamera2LatestFrameSource(
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        camera_num=args.camera_num,
        pixel_format="RGB888",
    )

    import cv2

    window_name = "Cane live detection - Q or Esc to quit"
    display = not args.no_display
    inference_ms: list[float] = []
    pipeline_ms: list[float] = []
    age_ms: list[float] = []
    camera_to_result_ms: list[float] = []
    dropped_total = 0
    measured = 0
    started_ns = perf_counter_ns()

    try:
        source.start()
        while True:
            frame = source.next_frame(timeout_s=10.0)
            dropped_total += frame.dropped_since_last
            available_ns = perf_counter_ns()
            result = model.predict(
                source=frame.payload,
                imgsz=imgsz,
                conf=args.confidence,
                verbose=False,
            )[0]
            finished_ns = perf_counter_ns()

            measured += 1
            if measured > args.warmup_frames:
                inference_ms.append(float(result.speed["inference"]))
                pipeline_ms.append((finished_ns - available_ns) / 1e6)
                if frame.captured_ns is not None:
                    age_ms.append((available_ns - frame.captured_ns) / 1e6)
                    camera_to_result_ms.append((finished_ns - frame.captured_ns) / 1e6)

            if display:
                annotated = result.plot()
                cv2.putText(
                    annotated,
                    f"{result.speed['inference']:.1f} ms  {(finished_ns - available_ns) / 1e6:.1f} ms pipeline",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            elif measured % 10 == 0 and len(pipeline_ms) >= 10:
                recent = pipeline_ms[-10:]
                print(
                    f"frames {measured}  pipeline {statistics.mean(recent):.1f} ms  "
                    f"{1000 / statistics.mean(recent):.1f} FPS  detections {len(result.boxes)}",
                    flush=True,
                )

            if args.duration and (perf_counter_ns() - started_ns) / 1e9 >= args.duration:
                break
    finally:
        source.close()
        if display:
            cv2.destroyAllWindows()

    elapsed_s = (perf_counter_ns() - started_ns) / 1e9
    print(f"\n{args.model}  input {imgsz}x{imgsz}  threads {args.threads}")
    print(f"frames {measured}  elapsed {elapsed_s:.1f} s  dropped {dropped_total}")
    if inference_ms:
        print(summarize("inference ", inference_ms))
        print(summarize("pipeline  ", pipeline_ms))
        if age_ms:
            print(summarize("frame age ", age_ms))
            print(summarize("cam->result", camera_to_result_ms))
        print(f"throughput: {1000 / statistics.mean(pipeline_ms):.2f} FPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
