"""Live multi-threading tests for the Smart Cane runtime.

Everything here runs the real four-thread runtime against the in-memory
simulation, so threading, latency, telemetry, JSONL logging, and teardown are
exercised on any machine rather than only on a Pi 5.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from yolo_pi.fusion_v2 import ExtendedHazardLevel
from yolo_pi.runtime import (
    SimulatedToFArray,
    build_simulated_runtime,
)
from yolo_pi.types import Sector

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_smart_cane.py"


def test_simulated_soak_run_produces_directional_haptics(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    runtime = build_simulated_runtime(
        enable_camera=True, log_jsonl_path=log_path, control_loop_hz=50.0
    )
    summary = runtime.run_for(0.6)
    sink = runtime.haptic_sink

    assert summary.ticks >= 25, summary.as_dict()
    assert summary.tick_rate_hz >= 40.0, summary.as_dict()
    assert summary.vision_frames >= 5, summary.as_dict()
    assert summary.sensor_to_haptic_ms.p95_ms < 20.0, summary.as_dict()
    assert summary.sensor_to_haptic_ms.count == summary.ticks

    center_frames = [
        frame
        for frame in sink.frames
        if frame.motors[Sector.CENTER].intensity > 0.5
    ]
    assert center_frames, "an approaching center obstacle must actuate the center motor"
    assert any(
        frame.motors[Sector.CENTER].level == ExtendedHazardLevel.CRITICAL
        for frame in sink.frames
    )
    assert all(
        frame.motors[Sector.LEFT].intensity == 0.0 for frame in sink.frames
    ), "a clear left sector must stay silent"

    assert runtime.thread_errors == ()
    assert runtime.vision_errors == ()
    assert runtime.latest_detections is not None
    assert runtime.latest_detections.detections

    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(lines) == summary.ticks
    for record in lines:
        assert record["decision"]["sectors"].keys() == {"left", "center", "right"}
        assert "haptics" in record

    assert sink.stopped
    assert sink.all_motors_zeroed
    assert not runtime.running


def test_slow_ultrasonic_thread_never_stalls_fast_loop() -> None:
    runtime = build_simulated_runtime(
        enable_camera=False,
        ultrasonic_read_delay_s=0.06,
        control_loop_hz=50.0,
    )
    summary = runtime.run_for(0.6)
    ultrasonic = runtime.ultrasonic_driver
    assert ultrasonic is not None

    assert summary.ticks >= 25, summary.as_dict()
    assert summary.tick_rate_hz >= 40.0, summary.as_dict()
    assert summary.sensor_to_haptic_ms.p95_ms < 20.0, summary.as_dict()
    assert ultrasonic.reads <= 12, "60 ms pings cannot outpace a 20 ms period"
    assert summary.ticks > ultrasonic.reads * 2
    assert runtime.thread_errors == ()


class _ExplodingToFArray:
    """Wraps the simulation and raises on the third poll."""

    def __init__(self, inner: SimulatedToFArray, fail_after: int = 3) -> None:
        self._inner = inner
        self._fail_after = fail_after
        self._calls = 0

    def read_all(self, now_ns=None):
        self._calls += 1
        if self._calls >= self._fail_after:
            raise RuntimeError("simulated I2C wedge")
        return self._inner.read_all(now_ns)

    def close(self) -> None:
        self._inner.close()


def test_loop_exception_still_stops_haptics() -> None:
    runtime = build_simulated_runtime(enable_camera=False)
    runtime.tof_array = _ExplodingToFArray(runtime.tof_array)
    sink = runtime.haptic_sink

    with pytest.raises(RuntimeError, match="runtime thread failed"):
        runtime.run_for(0.5)

    assert runtime.thread_errors
    assert isinstance(runtime.thread_errors[0], RuntimeError)
    assert sink.stopped
    assert sink.all_motors_zeroed
    assert not runtime.running


def test_cli_simulate_mode_exits_cleanly() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "simulate",
            "--no-camera",
            "--duration",
            "0.4",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "smart-cane summary" in result.stdout
    assert "sensor->haptic" in result.stdout
    assert "runtime error" not in result.stderr


def test_runtime_context_manager_tarears_down() -> None:
    runtime = build_simulated_runtime(enable_camera=False)
    with runtime:
        assert runtime.running
    assert not runtime.running
    assert runtime.haptic_sink.stopped
    assert runtime.haptic_sink.all_motors_zeroed
    with pytest.raises(ValueError, match="must not be negative"):
        runtime.run_for(-1.0)
