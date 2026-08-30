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
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
