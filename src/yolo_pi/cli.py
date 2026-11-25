"""Command-line entry points for laptop smoke testing."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .benchmark import BenchmarkInput, run_benchmark
from .detector import UltralyticsDetector
from .fusion import FusionEngine
from .mount import load_mount_profile
from .replay import load_sensor_jsonl, replay_readings, write_decisions_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO Pi laptop preparation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-model", help="Run one image through an Ultralytics-compatible model"
    )
    inspect_parser.add_argument("--model", default="yolo26n.pt")
    inspect_parser.add_argument("--input", required=True)
    inspect_parser.add_argument("--imgsz", type=int, default=416)
    inspect_parser.add_argument("--confidence", type=float, default=0.35)
    inspect_parser.add_argument("--device", default=None)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run a fixed-input laptop functional benchmark"
    )
    benchmark_parser.add_argument("--model", default="yolo26n.pt")
    benchmark_parser.add_argument("--inputs", nargs="+", required=True)
    benchmark_parser.add_argument("--imgsz", type=int, default=416)
    benchmark_parser.add_argument("--confidence", type=float, default=0.35)
    benchmark_parser.add_argument("--device", default=None)
    benchmark_parser.add_argument("--warmup", type=int, default=5)
    benchmark_parser.add_argument("--repetitions", type=int, default=10)
    benchmark_parser.add_argument("--output", type=Path, required=True)

    replay_parser = subparsers.add_parser(
        "replay-sensors", help="Replay timestamped sensor JSONL through fusion"
    )
    replay_parser.add_argument("--input", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)

    mount_parser = subparsers.add_parser(
        "validate-mount", help="Validate and print a camera-mount profile"
    )
    mount_parser.add_argument("--profile", type=Path, required=True)

    sim_parser = subparsers.add_parser(
        "simulate-fusion", help="Run the 14-scenario physics-based fusion simulation suite"
    )
    sim_parser.add_argument("--seed", type=int, default=42)
    sim_parser.add_argument("--output", type=Path, default=None)

    cane_parser = subparsers.add_parser(
        "run-cane", help="Run the live multi-threaded smart cane runtime"
    )
    cane_parser.add_argument("--mode", choices=("hardware", "simulate"), default="hardware")
    cane_parser.add_argument("--config", type=Path, default=Path("configs/smart-cane-pi5.json"))
    cane_parser.add_argument("--model", default=None)
    cane_parser.add_argument("--threads", type=int, default=2)
    cane_parser.add_argument("--no-camera", action="store_true")
    cane_parser.add_argument("--duration", type=float, default=0.0)
    cane_parser.add_argument("--haptic-mode", choices=("rtp", "library"), default="rtp")
    cane_parser.add_argument("--log-jsonl", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-model":
        detector = UltralyticsDetector(
            model_path=args.model,
            imgsz=args.imgsz,
            confidence=args.confidence,
            device=args.device,
        )
        result = detector.detect(args.input, frame_id=0, source=args.input)
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark":
        detector = UltralyticsDetector(
            model_path=args.model,
            imgsz=args.imgsz,
            confidence=args.confidence,
            device=args.device,
        )
        report = run_benchmark(
            detector,
            [BenchmarkInput(path, path) for path in args.inputs],
            warmup_runs=args.warmup,
            repetitions=args.repetitions,
        )
        report.write_json(args.output)
        print(json.dumps(report.as_dict()["model_call_ms"], indent=2, sort_keys=True))
        return 0
    if args.command == "replay-sensors":
        decisions = replay_readings(load_sensor_jsonl(args.input), FusionEngine())
        write_decisions_jsonl(decisions, args.output)
        print(f"wrote {len(decisions)} decisions to {args.output}")
        return 0
    if args.command == "validate-mount":
        profile = load_mount_profile(args.profile)
        print(json.dumps(profile.as_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "simulate-fusion":
        from .simulator import run_all_scenarios

        results = run_all_scenarios(seed=args.seed)
        payload = [result.as_dict() for result in results]
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        passed = sum(1 for result in results if result.passed)
        print(f"simulate-fusion: {passed}/{len(results)} scenarios passed (seed={args.seed})")
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.name}: {result.summary}")
        return 0 if passed == len(results) else 1
    if args.command == "run-cane":
        from .hardware import load_hardware_config
        from .runtime import _build_detector, build_hardware_runtime, build_simulated_runtime

        config = load_hardware_config(args.config)
        if args.mode == "simulate":
            detector = (
                _build_detector(args.model, args.threads)
                if (not args.no_camera and args.model)
                else None
            )
            runtime = build_simulated_runtime(
                config,
                detector=detector,
                enable_camera=not args.no_camera,
                log_jsonl_path=args.log_jsonl,
            )
        else:
            runtime = build_hardware_runtime(
                config,
                enable_camera=not args.no_camera,
                model_path=args.model,
                num_threads=args.threads,
                log_jsonl_path=args.log_jsonl,
                haptic_mode=args.haptic_mode,
            )
        runtime.install_signal_handlers()
        summary = runtime.run_for(args.duration if args.duration > 0 else 3600.0)
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
