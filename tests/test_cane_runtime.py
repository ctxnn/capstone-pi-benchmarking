import json
import signal
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from yolo_pi.cane.app import build_system, run_until_signal
from yolo_pi.cane.config import CaneConfig
from yolo_pi.cane.haptics import HapticDriver
from yolo_pi.cane.runtime import CaneRuntime, EventLog, HazardSmoother, LockedFusion
from yolo_pi.cane.sensors import ToFPoller, UltrasonicPoller
from yolo_pi.cane.sim import build_simulated_hardware
from yolo_pi.fusion import (
    HazardDecision,
    HazardLevel,
    RecordingHapticSink,
    SensorKind,
    SensorReading,
)
from yolo_pi.types import Sector

MS = 1_000_000


def decision(level, ts, sectors=(Sector.CENTER,), nearest=0.4):
    return HazardDecision(ts, level, tuple(sectors), nearest, tuple(), True, 2, "test")


class SmootherTests(unittest.TestCase):
    def test_rising_severity_is_never_delayed(self) -> None:
        smoother = HazardSmoother(250)
        self.assertEqual(smoother.apply(decision(HazardLevel.CLEAR, 0)).level, HazardLevel.CLEAR)
        self.assertEqual(smoother.apply(decision(HazardLevel.WARNING, 10 * MS)).level, HazardLevel.WARNING)
        self.assertEqual(smoother.apply(decision(HazardLevel.CRITICAL, 20 * MS)).level, HazardLevel.CRITICAL)

    def test_critical_is_held_briefly_then_released_stepwise(self) -> None:
        smoother = HazardSmoother(250)
        smoother.apply(decision(HazardLevel.CRITICAL, 0))
        held = smoother.apply(decision(HazardLevel.WARNING, 100 * MS))
        self.assertEqual(held.level, HazardLevel.CRITICAL)
        self.assertEqual(held.timestamp_ns, 100 * MS)  # re-stamped so it stays fresh
        self.assertEqual(smoother.apply(decision(HazardLevel.WARNING, 300 * MS)).level, HazardLevel.WARNING)
        self.assertEqual(smoother.apply(decision(HazardLevel.CLEAR, 400 * MS)).level, HazardLevel.WARNING)
        self.assertEqual(smoother.apply(decision(HazardLevel.CLEAR, 600 * MS)).level, HazardLevel.CLEAR)

    def test_a_boundary_flicker_does_not_flip_the_pattern(self) -> None:
        smoother = HazardSmoother(250)
        levels = []
        for i in range(10):
            raw = HazardLevel.WARNING if i % 2 == 0 else HazardLevel.CLEAR
            levels.append(smoother.apply(decision(raw, i * 30 * MS)).level)
        self.assertTrue(all(level == HazardLevel.WARNING for level in levels))

    def test_a_sensor_fault_is_shown_immediately(self) -> None:
        smoother = HazardSmoother(250)
        smoother.apply(decision(HazardLevel.CRITICAL, 0))
        fault = smoother.apply(decision(HazardLevel.SENSOR_FAULT, 10 * MS, (Sector.UNKNOWN,), None))
        self.assertEqual(fault.level, HazardLevel.SENSOR_FAULT)

    def test_seeing_the_same_level_again_refreshes_the_hold(self) -> None:
        smoother = HazardSmoother(250)
        smoother.apply(decision(HazardLevel.CRITICAL, 0))
        smoother.apply(decision(HazardLevel.CRITICAL, 200 * MS))
        self.assertEqual(smoother.apply(decision(HazardLevel.CLEAR, 400 * MS)).level, HazardLevel.CRITICAL)


def tap_groups(events, gap_s=0.12):
    """Sizes of the tap groups in a motor's (time, value) write log."""

    taps = []
    for stamp, value in events:
        if value > 0 and (not taps or taps[-1][1] is not None):
            taps.append([stamp, None])
        elif value == 0 and taps and taps[-1][1] is None:
            taps[-1][1] = stamp
    groups, previous_end = [], None
    for start, end in taps:
        if previous_end is None or start - previous_end > gap_s:
            groups.append(0)
        groups[-1] += 1
        previous_end = end if end is not None else start
    return groups


def reading(sensor_id, sector, distance, ts, kind=SensorKind.TOF):
    return SensorReading(sensor_id, kind, ts, sector, distance, True)


class RuntimeTickTests(unittest.TestCase):
    def make(self, sink=None):
        config = replace(CaneConfig(), runtime=replace(CaneConfig().runtime, event_log=None))
        fusion = LockedFusion(config.fusion)
        sink = sink or RecordingHapticSink()
        return CaneRuntime(config, fusion, sink, clock_ns=lambda: 0), fusion, sink

    def test_tick_turns_readings_into_a_haptic_command(self) -> None:
        runtime, fusion, sink = self.make()
        fusion.update_sensor(reading("tof-left", Sector.LEFT, 0.4, 100 * MS))
        command = runtime.tick(110 * MS)
        self.assertEqual(command.level, HazardLevel.CRITICAL)
        self.assertEqual(command.sectors, (Sector.LEFT,))
        self.assertEqual((command.source, command.urgency), ("range", 0.75))  # 0.4 m is well inside 1.0 m
        self.assertGreater(command.intensity, 0.65)
        self.assertEqual(sink.commands[-1], command)

    def test_no_readings_is_the_fault_pattern(self) -> None:
        runtime, _, _ = self.make()
        command = runtime.tick(1_000 * MS)
        self.assertEqual(command.level, HazardLevel.SENSOR_FAULT)
        self.assertEqual(command.pulse_period_ms, 1000)

    def test_decision_changes_are_logged_once_not_every_tick(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "events.jsonl"
            config = replace(CaneConfig(), runtime=replace(CaneConfig().runtime, event_log=str(log_path)))
            fusion = LockedFusion(config.fusion)
            runtime = CaneRuntime(config, fusion, RecordingHapticSink(), clock_ns=lambda: 0)
            fusion.update_sensor(reading("t", Sector.CENTER, 0.3, 100 * MS))
            for ms in (101, 102, 103):
                runtime.tick(ms * MS)
            runtime._events.close()
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual([r["kind"] for r in records], ["decision"])
            self.assertEqual(records[0]["level"], "critical")

    def test_event_log_failure_never_stops_the_cane(self) -> None:
        class BrokenHandle:
            def write(self, _):
                raise OSError("disk full")

            def close(self):
                pass

        log = EventLog(None)
        log._handle = BrokenHandle()
        log.write("decision", {"x": 1})  # must not raise
        log.write("decision", {"x": 2})

    def test_run_loop_stops_on_event_and_reports_crashed_threads(self) -> None:
        runtime, _, _ = self.make()
        stop = threading.Event()
        runtime._clock_ns = time.perf_counter_ns
        runtime._sleep = lambda s: stop.set()  # stop after the first pass
        runtime.run(stop)
        self.assertGreaterEqual(runtime.stats.iterations, 1)

        broken, _, _ = self.make()
        broken._hardware_ok = lambda: False
        with self.assertRaisesRegex(RuntimeError, "crashed"):
            broken.run(threading.Event())


class ScriptWorld:
    """Obstacle distances per sector, changed by the test."""

    def __init__(self) -> None:
        self.state = {Sector.LEFT: None, Sector.CENTER: None, Sector.RIGHT: None}

    def __call__(self, _t):
        return dict(self.state)


class FullChainTests(unittest.TestCase):
    """Sim sensors -> pollers -> fusion -> smoother -> haptic routing, stepped by hand."""

    def test_a_left_hazard_plays_one_tap_and_a_right_hazard_three(self) -> None:
        config = replace(CaneConfig(), runtime=replace(CaneConfig().runtime, event_log=None))
        world = ScriptWorld()
        now = [10.0]
        hardware = build_simulated_hardware(config, world, clock=lambda: now[0])
        fusion = LockedFusion.from_config(config)
        clock_ns = lambda: int(now[0] * 1e9)
        tof = ToFPoller(hardware.tof, {s.sensor_id: s.sector for s in config.tof.sensors},
                        fusion, config.tof, clock_ns=clock_ns)
        us = UltrasonicPoller(hardware.ultrasonic, fusion, config.ultrasonic, clock_ns=clock_ns)
        haptics = HapticDriver(hardware.haptic, config.haptics, clock=lambda: now[0])
        runtime = CaneRuntime(config, fusion, haptics, clock_ns=clock_ns)

        def advance(steps, seconds=0.02):
            for _ in range(steps):
                now[0] += seconds
                tof.step()
                us.step()
                runtime.tick(clock_ns())
                haptics._apply(haptics.target(now[0]))

        advance(10)
        self.assertEqual(runtime.last_decision.level, HazardLevel.CLEAR)
        self.assertFalse(hardware.haptic.events)  # nothing to say, nothing buzzed

        world.state[Sector.LEFT] = 0.35
        advance(60)  # 1.2 s
        self.assertEqual(runtime.last_decision.level, HazardLevel.CRITICAL)
        self.assertEqual(set(tap_groups(hardware.haptic.events)[:-1]), {1})

        world.state[Sector.LEFT] = None
        advance(20)  # past the 250 ms release hold
        mark = len(hardware.haptic.events)
        world.state[Sector.RIGHT] = 0.35
        advance(60)
        self.assertEqual(set(tap_groups(hardware.haptic.events[mark:])[:-1]), {3})

        world.state[Sector.RIGHT] = None
        advance(30)
        self.assertEqual(runtime.last_decision.level, HazardLevel.CLEAR)
        self.assertEqual(hardware.haptic.events[-1][1], 0)


class SystemTests(unittest.TestCase):
    def make_config(self):
        base = CaneConfig()
        return replace(
            base,
            haptics=replace(base.haptics, startup_sweep=False),
            runtime=replace(base.runtime, event_log=None, status_interval_s=0.3),
        )

    def test_threaded_simulated_system_runs_and_parks_the_motors(self) -> None:
        config = self.make_config()
        world = ScriptWorld()
        clock = time.monotonic
        hardware = build_simulated_hardware(config, world, clock)
        system = build_system(config, hardware=hardware, with_vision=False)
        stop = threading.Event()
        system.start()
        worker = threading.Thread(target=system.run, args=(stop,))
        worker.start()
        try:
            time.sleep(0.3)
            world.state[Sector.RIGHT] = 0.3
            time.sleep(0.6)
            self.assertEqual(system.runtime.last_decision.level, HazardLevel.CRITICAL)
            self.assertEqual(system.runtime.last_decision.sectors, (Sector.RIGHT,))
        finally:
            stop.set()
            worker.join(timeout=3)
            system.close()
        events = hardware.haptic.events
        self.assertEqual(max(tap_groups(events)), 3)  # right = three taps, never more
        self.assertEqual(events[-1][1], 0)  # parked

    def test_run_until_signal_honours_duration_and_returns_zero(self) -> None:
        config = self.make_config()
        system = build_system(config, simulate=True, with_vision=False)
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            self.assertEqual(run_until_signal(system, duration_s=0.5), 0)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
        self.assertGreater(system.runtime.stats.iterations, 5)

    def test_simulated_system_can_include_the_vision_worker(self) -> None:
        config = self.make_config()
        system = build_system(config, simulate=True, with_vision=True)
        stop = threading.Event()
        system.start()
        worker = threading.Thread(target=system.run, args=(stop,))
        worker.start()
        time.sleep(0.5)
        stop.set()
        worker.join(timeout=3)
        system.close()
        self.assertGreater(system.vision.frames, 0)


if __name__ == "__main__":
    unittest.main()
