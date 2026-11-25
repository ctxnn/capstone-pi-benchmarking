#!/usr/bin/env python3
"""Run the live Smart Cane runtime on a Raspberry Pi 5 or in simulation.

Examples::

    python scripts/run_smart_cane.py --mode simulate --no-camera --duration 5
    python scripts/run_smart_cane.py --mode hardware --haptic-mode rtp
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from yolo_pi.fusion_v2 import ExtendedHazardLevel  # noqa: E402
from yolo_pi.hardware import HardwareConfigError, load_hardware_config  # noqa: E402
from yolo_pi.runtime import (  # noqa: E402
    RuntimeSummary,
    SmartCaneRuntime,
    build_hardware_runtime,
    build_simulated_runtime,
)
from yolo_pi.types import Sector  # noqa: E402

_SECTORS = (Sector.LEFT, Sector.CENTER, Sector.RIGHT)


def _resolve_config(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    candidate = REPO_ROOT / path
    return candidate if candidate.exists() else path


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the live multi-threaded Smart Cane runtime."
    )
    parser.add_argument(
        "--mode",
        choices=("hardware", "simulate"),
        default="hardware",
        help="real Pi 5 drivers or the in-memory simulation (default: hardware)",
    )
    parser.add_argument(
        "--config",
        default="configs/smart-cane-pi5.json",
        help="hardware profile JSON (default: configs/smart-cane-pi5.json)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="detector model path (default: the profile's camera.model_path)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="inference threads; 2 keeps the Pi 5 inside its thermal budget",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="run sensor, IMU, and haptic fusion without vision",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to run; 0 runs until Ctrl-C (default: 0)",
    )
    parser.add_argument(
        "--haptic-mode",
        choices=("rtp", "library"),
        default="rtp",
        help="DRV2605L drive mode (default: rtp)",
    )
    parser.add_argument(
        "--log-jsonl",
        default=None,
        help="optional path for a per-tick JSONL decision log",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=1.0,
        help="seconds between live status lines (default: 1.0)",
    )
    return parser.parse_args(argv)


def _distance(value: Optional[float]) -> str:
    if value is None:
        return "  --  "
    return f"{value:5.2f}m"


def _format_status(runtime: SmartCaneRuntime) -> str:
    decision = runtime.latest_decision
    frame = runtime.latest_haptic_frame
    detections = runtime.latest_detections
    if decision is None:
        return "[warming up] no decision yet"

    parts = []
    for sector, tag in ((Sector.LEFT, "L"), (Sector.CENTER, "C"), (Sector.RIGHT, "R")):
        state = decision.sectors[sector]
        parts.append(f"{tag} {_distance(state.fused_distance_m)} {state.level.name.lower()}")
    line = " | ".join(parts)

    motors = []
    for sector, tag in ((Sector.LEFT, "L"), (Sector.CENTER, "C"), (Sector.RIGHT, "R")):
        command = frame.motors.get(sector) if frame is not None else None
        intensity = int(round(100.0 * command.intensity)) if command is not None else 0
        motors.append(f"{tag} {intensity:3d}%")

    labels = ""
    if detections is not None and detections.detections:
        nearest = sorted(detections.detections, key=lambda det: det.box.area, reverse=True)[0]
        labels = f"{nearest.class_name} ({nearest.confidence:.2f})"
    else:
        labels = "none"

    p95 = runtime.metrics.snapshot().sensor_to_haptic_ms.p95_ms
    age = runtime.metrics.vision_age_ms
    vision_age = "  --  " if age == float("inf") else f"{age:4.1f}ms"
    return (
        f"{line} | motors {' '.join(motors)} | vision {labels} "
        f"| vision_age {vision_age} | p95 {p95:.1f}ms"
    )


def _format_summary(summary: RuntimeSummary) -> str:
    payload = summary.as_dict()
    counts = payload.pop("sector_hazard_counts")
    lines = [
        "smart-cane summary",
        f"  duration          : {payload['duration_s']:.2f} s "
        f"({payload['tick_rate_hz']:.1f} Hz, {payload['ticks']} ticks)",
        f"  sensor->haptic    : p50 {payload['sensor_to_haptic_ms']['p50_ms']:.2f} ms "
        f"p95 {payload['sensor_to_haptic_ms']['p95_ms']:.2f} ms "
        f"p99 {payload['sensor_to_haptic_ms']['p99_ms']:.2f} ms",
        f"  fusion eval       : p50 {payload['fusion_eval_ms']['p50_ms']:.2f} ms "
        f"p95 {payload['fusion_eval_ms']['p95_ms']:.2f} ms",
        f"  loop jitter       : p95 {payload['loop_jitter_ms']['p95_ms']:.2f} ms",
        f"  vision            : {payload['vision_frames']} frames, "
        f"{payload['frame_drops']} dropped, "
        f"p95 {payload['vision_call_ms']['p95_ms']:.2f} ms, "
        + (
            "age n/a"
            if payload["vision_age_ms"] == float("inf")
            else f"age {payload['vision_age_ms']:.1f} ms"
        ),
        f"  degraded events   : {payload['degraded_events']}",
        "  hazard ticks      : "
        + ", ".join(f"{name} {count}" for name, count in counts.items()),
    ]
    return "\n".join(lines)


def _build_runtime(args: argparse.Namespace) -> SmartCaneRuntime:
    config_path = _resolve_config(args.config)
    try:
        config = load_hardware_config(config_path)
    except HardwareConfigError as exc:
        raise SystemExit(f"hardware config error: {exc}") from exc

    if args.mode == "simulate":
        detector = None
        if not args.no_camera and args.model:
            from yolo_pi.runtime import _build_detector

            detector = _build_detector(args.model, args.threads)
        return build_simulated_runtime(
            config,
            detector=detector,
            enable_camera=not args.no_camera,
            log_jsonl_path=args.log_jsonl,
        )
    return build_hardware_runtime(
        config,
        enable_camera=not args.no_camera,
        model_path=args.model,
        num_threads=args.threads,
        log_jsonl_path=args.log_jsonl,
        haptic_mode=args.haptic_mode,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.print_interval <= 0.0:
        raise SystemExit("--print-interval must be positive")
    if args.duration < 0.0:
        raise SystemExit("--duration must not be negative")
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")

    runtime = _build_runtime(args)
    runtime.install_signal_handlers()
    started = time.monotonic()
    exit_code = 0
    try:
        runtime.start()
    except BaseException as exc:  # noqa: BLE001 - report a clean startup failure
        print(f"failed to start: {exc!r}", file=sys.stderr, flush=True)
        return 2
    print(
        f"smart-cane {args.mode} mode running at {runtime.control_loop_hz:.0f} Hz"
        f"{' without camera' if args.no_camera else ''}; Ctrl-C to stop",
        flush=True,
    )
    try:
        next_print = started + args.print_interval
        while not runtime.stop_requested:
            now = time.monotonic()
            if args.duration > 0.0 and now - started >= args.duration:
                break
            if runtime.thread_errors:
                exit_code = 1
                break
            if now >= next_print:
                print(f"[{now - started:6.1f}s] {_format_status(runtime)}", flush=True)
                next_print = now + args.print_interval
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()

    for error in runtime.thread_errors:
        print(f"runtime error: {error!r}", file=sys.stderr, flush=True)
    if runtime.vision_errors:
        print(
            f"vision warnings: {len(runtime.vision_errors)} "
            f"(last: {runtime.vision_errors[-1]!r})",
            file=sys.stderr,
            flush=True,
        )
    summary = runtime.summary()
    print(_format_summary(summary), flush=True)
    if runtime.latest_decision is not None:
        level = runtime.latest_decision.overall_level
        if level == ExtendedHazardLevel.SENSOR_FAULT:
            print("final state: sensor fault", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
