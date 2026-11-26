#!/usr/bin/env python3
"""Live preview of what the cane's vision stage sees.

Draws the detector's boxes with class, confidence and the cane's own monocular
distance estimate, the three sector bands, and the decision the priority engine
would publish for that frame. It needs only the IMX219 and the model, so it works
with the sensor harness unplugged and while the mux is down.

Run it on the Pi desktop (Raspberry Pi Connect screen sharing shows the window):

    PYTHONPATH=src .venv/bin/python scripts/cane_preview.py

Q or Esc closes the window. ``--no-display`` prints decisions instead, which is
useful over a shell.

Drawing needs OpenCV's HighGUI. If ``import cv2`` fails:

    sudo apt install -y python3-opencv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

SECTOR_COLORS = {
    "left": (255, 190, 0),
    "center": (0, 210, 90),
    "right": (60, 80, 255),
    "unknown": (200, 200, 200),
}
LEVEL_COLORS = {
    "clear": (90, 90, 90),
    "advisory": (0, 190, 255),
    "warning": (0, 140, 255),
    "critical": (40, 40, 255),
    "sensor_fault": (120, 120, 120),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cane-hardware.json"))
    parser.add_argument("--model", type=str, default=None, help="override the configured OpenVINO model")
    parser.add_argument("--confidence", type=float, default=None, help="override the configured confidence")
    parser.add_argument("--threads", type=int, default=None, help="override the configured thread count")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = until Q)")
    parser.add_argument("--no-display", action="store_true", help="print decisions instead of opening a window")
    parser.add_argument("--window-scale", type=float, default=1.5, help="window size multiplier")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from dataclasses import replace

    from yolo_pi.cane.config import load_config
    from yolo_pi.cane.priority import PriorityFusion, estimate_distance_m
    from yolo_pi.cane.vision import create_camera, create_detector

    config = load_config(args.config)
    vision = config.vision
    if args.model:
        vision = replace(vision, model=args.model)
    if args.confidence is not None:
        vision = replace(vision, confidence=args.confidence)
    if args.threads is not None:
        vision = replace(vision, threads=args.threads)

    camera = create_camera(vision)
    detector = create_detector(vision)
    fusion = PriorityFusion(config.fusion, config.priority)

    cv2 = None
    window = "Cane vision - Q or Esc to quit"
    if not args.no_display:
        try:
            import cv2  # noqa: F811 - optional display dependency
        except ImportError:
            print("OpenCV is missing; install it with: sudo apt install -y python3-opencv", flush=True)
            return 2

    print(f"model {vision.model}  input {vision.imgsz or 'from metadata'}  threads {vision.threads}", flush=True)
    started = time.monotonic()
    frames = 0
    last_print = 0.0
    try:
        camera.start()
        while True:
            packet = camera.next_frame(timeout_s=10.0)
            result = detector.detect(
                packet.payload,
                packet.frame_id,
                captured_ns=packet.captured_ns,
                source=packet.source,
            )
            fusion.update_vision(result)
            decision = fusion.evaluate(time.perf_counter_ns())
            frames += 1

            if args.no_display:
                now = time.monotonic()
                if now - last_print >= 1.0:
                    last_print = now
                    labels = [det.class_name for det in result.detections]
                    print(
                        f"{decision.level.name.lower():7s} source={decision.source:8s} "
                        f"sectors={[s.value for s in decision.sectors]} "
                        f"nearest={decision.nearest_distance_m if decision.nearest_distance_m is None else round(decision.nearest_distance_m, 2)} "
                        f"detections={labels} frame={frames}",
                        flush=True,
                    )
            else:
                image = packet.payload.copy()
                if image.ndim == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                height, width = image.shape[:2]
                band = width * config.vision.center_fraction
                left_edge = int((width - band) / 2.0)
                right_edge = int((width + band) / 2.0)
                for x, text in ((left_edge, "LEFT"), (right_edge, "RIGHT")):
                    cv2.line(image, (x, 0), (x, height), (70, 70, 70), 1)
                    cv2.putText(
                        image, text, (x + 6, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (140, 140, 140), 1, cv2.LINE_AA,
                    )
                for det in result.detections:
                    distance = estimate_distance_m(det, width, height, config.priority)
                    color = SECTOR_COLORS.get(det.sector.value, SECTOR_COLORS["unknown"])
                    box = det.box
                    cv2.rectangle(
                        image,
                        (int(box.x1), int(box.y1)),
                        (int(box.x2), int(box.y2)),
                        color,
                        2,
                    )
                    label = f"{det.class_name} {det.confidence:.2f}"
                    if distance is not None:
                        label += f"  {distance:.1f} m"
                    label += f"  {det.sector.value}"
                    cv2.putText(
                        image, label, (int(box.x1), max(18, int(box.y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
                    )
                banner = (
                    f"{decision.level.name}  {decision.source}  "
                    f"nearest {decision.nearest_distance_m if decision.nearest_distance_m is None else round(decision.nearest_distance_m, 2)} m  "
                    f"frame {frames}  {result.model_call_ms:.0f} ms"
                    if hasattr(result, "model_call_ms")
                    else f"{decision.level.name}  {decision.source}  frame {frames}"
                )
                cv2.rectangle(image, (0, 0), (width, 34), (0, 0, 0), -1)
                cv2.putText(
                    image, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    LEVEL_COLORS.get(decision.level.name.lower(), (255, 255, 255)), 2, cv2.LINE_AA,
                )
                if args.window_scale != 1.0:
                    image = cv2.resize(
                        image, None,
                        fx=args.window_scale, fy=args.window_scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                cv2.imshow(window, image)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            if args.duration and time.monotonic() - started >= args.duration:
                break
    finally:
        camera.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    print(f"frames {frames} in {time.monotonic() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
