"""Run the fourteen-scenario SmartCane v2 fusion stress suite.

Every scenario drives the production fusion engine and haptic mapper through a
deterministic physics simulation, so a red row points at real pipeline
behaviour rather than at a mock. The process exits non-zero unless all fourteen
scenarios pass their assertions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from yolo_pi.simulator import SimulationResult, run_all_scenarios

_NAME_WIDTH = 36


def _row(
    number: str,
    name: str,
    verdict: str,
    ticks: str,
    latency: str,
    chatter: str,
    false_critical: str,
    false_warning: str,
    p95: str,
) -> str:
    return (
        f"{number:>2} {name:<{_NAME_WIDTH}} {verdict:<6} {ticks:>5} {latency:>9} "
        f"{chatter:>7} {false_critical:>6} {false_warning:>6} {p95:>7}"
    )


def format_table(results: Sequence[SimulationResult]) -> str:
    lines = [
        _row("#", "scenario", "result", "ticks", "latency", "chatter", "falseC", "falseW", "p95 ms"),
        "-" * 3
        + "-" * (_NAME_WIDTH + 1)
        + "-" * 7
        + "-" * 6
        + "-" * 10
        + "-" * 8
        + "-" * 7
        + "-" * 7
        + "-" * 8,
    ]
    for index, result in enumerate(results, start=1):
        latency = "-" if result.reaction_latency_ms is None else "{:.1f}".format(
            result.reaction_latency_ms
        )
        lines.append(
            _row(
                str(index),
                result.name[:_NAME_WIDTH],
                "PASS" if result.passed else "FAIL",
                str(result.ticks),
                latency,
                str(result.chatter_transitions),
                str(result.false_critical_count),
                str(result.false_warning_count),
                "{:.3f}".format(result.p95_tick_eval_ms),
            )
        )
    passed = sum(1 for result in results if result.passed)
    lines.append("")
    lines.append("{}/{} scenarios passed".format(passed, len(results)))
    for result in results:
        if result.passed:
            continue
        failed = [name for name, ok in result.checks.items() if not ok]
        lines.append("  FAIL {}: {}".format(result.name, ", ".join(failed)))
        lines.append("       {}".format(result.summary))
    return "\n".join(lines)


def build_report(seed: int, results: Sequence[SimulationResult]) -> dict:
    return {
        "seed": seed,
        "total": len(results),
        "passed": sum(1 for result in results if result.passed),
        "results": [result.as_dict() for result in results],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulate the SmartCane perception, fusion, IMU and haptic pipeline"
    )
    parser.add_argument("--seed", type=int, default=42, help="deterministic noise seed")
    parser.add_argument("--output", type=Path, help="write the full scorecard as JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress the table")
    args = parser.parse_args(argv)

    results = run_all_scenarios(seed=args.seed)
    report = build_report(args.seed, results)
    if not args.quiet:
        print(format_table(results))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not args.quiet:
            print("report written to {}".format(args.output))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
