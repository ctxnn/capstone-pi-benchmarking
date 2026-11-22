#!/usr/bin/env python3
"""Quick Raspberry Pi speed comparison for exported Cane models on the fixed images."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from pathlib import Path

import cv2

from yolo_pi.pi_benchmark import configure_threads, load_fixed_inputs


def export_imgsz(model: Path) -> list[int]:
    import yaml

    for candidate in (model / "metadata.yaml", model.parent / "metadata.yaml"):
        if candidate.is_file():
            return list(yaml.safe_load(candidate.read_text())["imgsz"])
    raise FileNotFoundError(f"no metadata.yaml next to {model}")


def pi_state() -> str:
    try:
        temp = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True).stdout.strip()
        throttled = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
        return f"{temp} {throttled}"
    except FileNotFoundError:
        return "vcgencmd unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--inputs", type=Path, default=Path("artifacts/benchmarks/pi-runtime-inputs/manifest.json"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--rest", type=float, default=20.0, help="seconds to idle between models")
    args = parser.parse_args()

    configure_threads(args.threads)
    from ultralytics import YOLO

    frames = [cv2.imread(str(path)) for path in load_fixed_inputs(args.inputs, Path.cwd())]
    rows = []
    for index, model_path in enumerate(args.models):
        if index:
            time.sleep(args.rest)
        imgsz = export_imgsz(model_path)
        model = YOLO(str(model_path), task="detect")
        before = pi_state()
        totals, inference = [], []
        for run in range(args.warmup + args.runs):
            started = time.perf_counter()
            result = model.predict(frames[run % len(frames)], imgsz=imgsz, conf=0.25, verbose=False)[0]
            if run >= args.warmup:
                totals.append((time.perf_counter() - started) * 1000)
                inference.append(result.speed["inference"])
        totals.sort()
        row = {
            "model": str(model_path),
            "imgsz": "x".join(map(str, imgsz)),
            "inference_ms": statistics.mean(inference),
            "total_ms": statistics.mean(totals),
            "p95_ms": totals[int(0.95 * (len(totals) - 1))],
            "fps": 1000 / statistics.mean(totals),
            "before": before,
            "after": pi_state(),
        }
        rows.append(row)
        print(f"{row['model']}  {row['imgsz']}  inference {row['inference_ms']:.1f} ms  total {row['total_ms']:.1f} ms  "
              f"p95 {row['p95_ms']:.1f} ms  {row['fps']:.2f} FPS  [{row['before']} -> {row['after']}]", flush=True)

    print("\n| Model | Input | Inference (ms) | Total (ms) | P95 (ms) | FPS |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(f"| {Path(row['model']).parent.name}/{Path(row['model']).name} | {row['imgsz']} | {row['inference_ms']:.1f} | "
              f"{row['total_ms']:.1f} | {row['p95_ms']:.1f} | {row['fps']:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
