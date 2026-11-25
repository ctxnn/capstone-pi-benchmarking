"""``yolo-pi-cane``: run, bench-test and inspect the whole cane.

    yolo-pi-cane scan                 list I2C devices on the bus and on every mux channel
    yolo-pi-cane selftest             read every sensor and play every vibration pattern once
    yolo-pi-cane buzz --pattern left  play one vibration pattern
    yolo-pi-cane run                  the real cane (what the systemd service starts)
    yolo-pi-cane run --simulate       the same software on a laptop, with scripted obstacles
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from .config import CaneConfig, ConfigError, load_config

DEFAULT_CONFIG = Path("configs/cane-hardware.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yolo-pi-cane", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None,
                        help=f"hardware configuration (default: {DEFAULT_CONFIG} if present)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the cane")
    run.add_argument("--simulate", action="store_true", help="no hardware: scripted obstacles")
    run.add_argument("--no-vision", action="store_true", help="range sensors and motors only")
    run.add_argument("--no-motors", action="store_true", help="log haptic commands instead of buzzing")
    run.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = until Ctrl+C)")

    selftest = sub.add_parser("selftest", help="bench bring-up checks")
    selftest.add_argument("--no-buzz", action="store_true", help="skip the motor buzz")
    selftest.add_argument("--seconds", type=float, default=3.0, help="ToF sampling time")
    selftest.add_argument("--simulate", action="store_true", help="exercise the checks on fake devices")

    soak = sub.add_parser("soak", help="fault-injection soak test on fake hardware (no Pi needed)")
    soak.add_argument("--seconds", type=float, default=60.0)
    soak.add_argument("--seed", type=int, default=1)

    sub.add_parser("scan", help="list I2C addresses on the main bus and each mux channel")

    buzz = sub.add_parser("buzz", help="play one vibration pattern on the haptic driver")
    buzz.add_argument("--pattern", required=True,
                      choices=("left", "center", "right", "overhead", "fault", "jolt"))
    buzz.add_argument("--urgency", type=float, default=0.5, help="0 = far/slow, 1 = very near/rapid")

    sub.add_parser("show-config", help="print the wiring this configuration describes")
    return parser


def _load(path: Optional[Path]) -> CaneConfig:
    if path is None and DEFAULT_CONFIG.is_file():
        path = DEFAULT_CONFIG
    return load_config(path)


def _cmd_run(config: CaneConfig, args: argparse.Namespace) -> int:
    from .app import build_system, run_until_signal

    system = build_system(
        config,
        simulate=args.simulate,
        with_vision=not args.no_vision,
        with_motors=not args.no_motors,
    )
    return run_until_signal(system, args.duration)


def _cmd_selftest(config: CaneConfig, args: argparse.Namespace) -> int:
    from .selftest import run_selftest

    if args.simulate:
        from .sim import SimClock, build_simulated_hardware

        hardware = build_simulated_hardware(config, clock=SimClock())
    else:
        from .hardware import build_real_hardware

        hardware = build_real_hardware(config)
    try:
        checks = run_selftest(config, hardware, duration_s=args.seconds, buzz=not args.no_buzz)
    finally:
        hardware.close()
    return 1 if any(check.status == "FAIL" for check in checks) else 0


def _cmd_scan(config: CaneConfig) -> int:
    from .bus import BlinkaBus

    bus = BlinkaBus(config.i2c.mux_address)
    try:
        found = bus.scan()
    finally:
        bus.close()
    print(f"main bus     : {[hex(a) for a in found.get(None, [])]}")
    expected = {}
    for spec in config.tof.sensors:
        expected.setdefault(spec.mux_channel, []).append(f"{hex(config.i2c.tof_address)} ({spec.sensor_id})")
    expected.setdefault(config.haptics.driver_mux_channel, []).append(
        f"{hex(config.i2c.haptic_address)} (haptic driver)")
    if config.imu.enabled and config.imu.mux_channel is not None:
        expected.setdefault(config.imu.mux_channel, []).append(f"{hex(config.i2c.imu_address)} (imu)")
    for channel in range(8):
        seen = [hex(a) for a in found.get(channel, [])]
        print(f"mux channel {channel}: {seen}   expected: {expected.get(channel, [])}")
    return 0


def _cmd_buzz(config: CaneConfig, args: argparse.Namespace) -> int:
    from ..fusion import HazardLevel
    from ..types import Sector
    from .hardware import build_real_hardware
    from .haptics import amplitude_to_rtp, play_pattern_once
    from .priority import CaneCommand

    hardware = build_real_hardware(config, with_ranging=False)
    try:
        if hardware.haptic is None:
            print("the haptic driver is not available", file=sys.stderr)
            return 2
        if args.pattern == "jolt":
            value = amplitude_to_rtp(config.imu.jolt_intensity, config.haptics.amplitude_floor)
            hardware.haptic.set_rtp(value)
            time.sleep(config.imu.jolt_pulse_ms / 1000.0)
            hardware.haptic.set_rtp(0)
        else:
            sector = {"left": Sector.LEFT, "center": Sector.CENTER, "right": Sector.RIGHT,
                      "overhead": Sector.CENTER, "fault": Sector.UNKNOWN}[args.pattern]
            source = args.pattern if args.pattern in ("overhead", "fault") else "range"
            urgency = max(0.0, min(1.0, args.urgency))
            intensity = 0.35 if source == "fault" else 0.65 + 0.35 * urgency
            command = CaneCommand(0, HazardLevel.WARNING, (sector,), intensity, None, "buzz", source, urgency)
            for _ in range(3):
                play_pattern_once(hardware.haptic, command, config.haptics)
                time.sleep(0.6)
        print(f"played the {args.pattern} pattern")
    finally:
        hardware.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    try:
        config = _load(args.config)
    except (ConfigError, OSError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    from .bus import HardwareUnavailable

    try:
        if args.command == "run":
            return _cmd_run(config, args)
        if args.command == "selftest":
            return _cmd_selftest(config, args)
        if args.command == "soak":
            from .chaos import run_everything

            ok, text = run_everything(config, args.seconds, args.seed, str(args.config) if args.config else None)
            print(text)
            return 0 if ok else 1
        if args.command == "scan":
            return _cmd_scan(config)
        if args.command == "buzz":
            return _cmd_buzz(config, args)
        if args.command == "show-config":
            print(json.dumps(config.describe(), indent=2, sort_keys=True))
            return 0
    except HardwareUnavailable as exc:
        print(f"hardware unavailable: {exc}", file=sys.stderr)
        return 3
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
