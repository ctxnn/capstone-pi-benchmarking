"""Bench bring-up checks for a freshly wired cane.

Run ``yolo-pi-cane selftest`` after every wiring change. It reads each sensor,
checks the I2C map against the configuration, calibrates the IMU, and plays every
vibration pattern once (so the person holding the cane learns what each feels
like), so a swapped mux channel or a loose ECHO wire shows up here and not on
the street.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .config import CaneConfig
from .hardware import HardwareSet
from ..fusion import HazardLevel
from ..types import Sector
from .haptics import play_pattern_once
from .imu import calibrate_gyro
from .priority import CaneCommand


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # PASS, WARN, FAIL or SKIP
    detail: str


def _check_i2c_map(config: CaneConfig, hardware: HardwareSet) -> List[Check]:
    if hardware.bus is None:
        return [Check("i2c map", "SKIP", "no I2C bus (simulation)")]
    found = hardware.bus.scan()
    checks: List[Check] = []
    main = found.get(None, [])
    checks.append(
        Check(
            "i2c mux",
            "PASS" if config.i2c.mux_address in main else "FAIL",
            f"TCA9548A expected at {hex(config.i2c.mux_address)}; main bus shows "
            f"{[hex(a) for a in main]}",
        )
    )
    for spec in config.tof.sensors:
        ok = config.i2c.tof_address in found.get(spec.mux_channel, [])
        checks.append(Check(f"i2c {spec.sensor_id}", "PASS" if ok else "FAIL",
                            f"{hex(config.i2c.tof_address)} on mux channel {spec.mux_channel}"))
    where = config.haptics.driver_mux_channel
    ok = config.i2c.haptic_address in found.get(where, [])
    checks.append(Check("i2c haptic driver", "PASS" if ok else "FAIL",
                        f"{hex(config.i2c.haptic_address)} on "
                        f"{'the main bus' if where is None else f'mux channel {where}'}"))
    if config.imu.enabled:
        where = config.imu.mux_channel
        ok = config.i2c.imu_address in found.get(where, [])
        checks.append(Check("i2c imu", "PASS" if ok else "FAIL",
                            f"{hex(config.i2c.imu_address)} on "
                            f"{'the main bus' if where is None else f'mux channel {where}'}"))
    return checks


def _check_tof(config: CaneConfig, hardware: HardwareSet, duration_s: float,
               sleep: Callable[[float], None]) -> List[Check]:
    stats: Dict[str, List[float]] = {name: [] for name in hardware.tof}
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        for name, device in hardware.tof.items():
            try:
                sample = device.poll()
            except Exception as exc:
                return [Check(f"tof {name}", "FAIL", f"read error: {exc}")]
            if sample is not None and sample.distance_m is not None:
                stats[name].append(sample.distance_m)
        sleep(0.005)
    checks = []
    required = max(3, int(duration_s * 10))  # a 33 ms budget gives ~30 per second; ask for a third
    for spec in config.tof.sensors:
        values = stats.get(spec.sensor_id)
        if values is None:
            checks.append(Check(f"tof {spec.sensor_id}", "FAIL", "device did not start"))
        elif len(values) < required:
            checks.append(Check(f"tof {spec.sensor_id}", "FAIL",
                                f"only {len(values)} readings in {duration_s:.1f} s "
                                f"(expected at least {required}, i.e. 10 per second)"))
        else:
            checks.append(Check(
                f"tof {spec.sensor_id}", "PASS",
                f"{len(values)} readings, {min(values):.2f}-{max(values):.2f} m "
                f"(sector {spec.sector.value})"))
    return checks


def _check_ultrasonic(config: CaneConfig, hardware: HardwareSet, rounds: int,
                      sleep: Callable[[float], None]) -> List[Check]:
    checks = []
    for spec in config.ultrasonic.sensors:
        device = hardware.ultrasonic.get(spec.sensor_id)
        if device is None:
            checks.append(Check(f"ultrasonic {spec.sensor_id}", "FAIL", "device did not start"))
            continue
        distances = []
        try:
            for _ in range(rounds):
                sample = device.measure()
                if sample.raw is not None and not sample.out_of_range and sample.distance_m:
                    distances.append(sample.distance_m)
                sleep(config.ultrasonic.spacing_ms / 1000.0)
        except Exception as exc:
            checks.append(Check(f"ultrasonic {spec.sensor_id}", "FAIL", f"measure error: {exc}"))
            continue
        if distances:
            checks.append(Check(f"ultrasonic {spec.sensor_id}", "PASS",
                                f"{len(distances)}/{rounds} echoes, {min(distances):.2f}-{max(distances):.2f} m"))
        else:
            checks.append(Check(
                f"ultrasonic {spec.sensor_id}", "WARN",
                f"no echo in {rounds} tries. Hold a hand 20-50 cm in front. If it stays silent, "
                "check TRIG/ECHO wiring and the 5 V to 3.3 V level shifter on ECHO"))
    return checks


def _check_imu(config: CaneConfig, hardware: HardwareSet) -> List[Check]:
    if hardware.imu is None:
        return [Check("imu", "SKIP" if not config.imu.enabled else "FAIL",
                      "disabled" if not config.imu.enabled else "did not start")]
    try:
        sample = hardware.imu.read()
        gravity = sum(v * v for v in sample.accel_g) ** 0.5
        calibration = calibrate_gyro(hardware.imu, config.imu)
    except Exception as exc:
        return [Check("imu", "FAIL", f"read error: {exc}")]
    checks = [Check(
        "imu", "PASS" if abs(gravity - 1.0) < 0.15 else "WARN",
        f"accel magnitude {gravity:.2f} g (expect ~1.00 at rest), die temperature {sample.temperature_c:.1f} C")]
    checks.append(Check(
        "imu gyro bias", "PASS" if calibration.calibrated else "WARN",
        "bias " + ", ".join(f"{v:.2f}" for v in calibration.bias_dps) + " deg/s"
        if calibration.calibrated else "cane was moving; keep it still and rerun"))
    return checks


def _check_haptic(config: CaneConfig, hardware: HardwareSet, sleep: Callable[[float], None]) -> List[Check]:
    """Play every pattern once so the person holding the cane learns what each one feels like."""

    if hardware.haptic is None:
        return [Check("haptic driver", "FAIL", "driver did not start")]
    plan = (
        ("left", "1 tap", (Sector.LEFT,), "range", 0.5),
        ("center", "2 taps", (Sector.CENTER,), "range", 0.5),
        ("right", "3 taps", (Sector.RIGHT,), "range", 0.5),
        ("center, very near", "2 taps, rapid", (Sector.CENTER,), "range", 1.0),
        ("overhead", "2 long pulses", (Sector.CENTER,), "overhead", 0.5),
        ("sensor fault", "1 long weak pulse", (Sector.UNKNOWN,), "fault", 0.0),
    )
    checks = []
    for name, feel, sectors, source, urgency in plan:
        intensity = 0.35 if source == "fault" else 0.65 + 0.35 * urgency
        command = CaneCommand(0, HazardLevel.WARNING, sectors, intensity, None, "selftest", source, urgency)
        try:
            play_pattern_once(hardware.haptic, command, config.haptics, sleep)
            sleep(0.5)
        except Exception as exc:
            checks.append(Check(f"haptic {name}", "FAIL", f"write error: {exc}"))
            continue
        checks.append(Check(f"haptic {name}", "PASS",
                            f"played ({feel}). You should have felt it; "
                            "a silent PASS means the motor wiring, not the driver, is at fault"))
    return checks


def run_selftest(
    config: CaneConfig,
    hardware: HardwareSet,
    *,
    duration_s: float = 3.0,
    ultrasonic_rounds: int = 20,
    buzz: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    out: Optional[Callable[[str], None]] = print,
) -> List[Check]:
    checks: List[Check] = [Check("startup", "FAIL", message) for message in hardware.errors]
    checks += _check_i2c_map(config, hardware)
    checks += _check_tof(config, hardware, duration_s, sleep)
    checks += _check_ultrasonic(config, hardware, ultrasonic_rounds, sleep)
    checks += _check_imu(config, hardware)
    if buzz:
        checks += _check_haptic(config, hardware, sleep)
    if out is not None:
        for check in checks:
            out(f"{check.status:<4}  {check.name:<24} {check.detail}")
        failed = sum(1 for check in checks if check.status == "FAIL")
        warned = sum(1 for check in checks if check.status == "WARN")
        out(f"\n{len(checks)} checks: {failed} failed, {warned} warnings")
    return checks
