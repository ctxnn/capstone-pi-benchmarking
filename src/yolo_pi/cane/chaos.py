"""Fault-injection soak test and fuzzing for the whole cane, on fake hardware.

``run_soak`` runs the real threaded system (pollers, fusion, smoother, haptic
driver, shutdown) against simulated devices that misbehave on purpose: random
bus errors, NaN and negative readings, slow reads, a sensor class going dead
for seconds, garbage camera frames, failing motor writes. It then checks the
properties a cane must keep, and returns a report; any violation is a failure.

Run it with ``yolo-pi-cane soak``. It needs no hardware.
"""

from __future__ import annotations

import logging
import math
import os
import random
import resource
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..fusion import SensorKind, SensorReading
from ..types import BoundingBox, Detection, FrameDetections, Sector
from .app import build_system
from .config import CaneConfig
from .imu import ImuSample, JoltDetector
from .priority import PriorityFusion
from .sim import (
    OVERHEAD,
    FAR,
    SimClock,
    SimImu,
    SimMotorDriver,
    WorldState,
    build_simulated_hardware,
)
from .tof import RangeSample

LOGGER = logging.getLogger("yolo_pi.cane.chaos")

BAD_FLOATS = (float("nan"), float("inf"), float("-inf"), -1.0, 0.0, 1e9, -1e9)
# Longest legitimate stretch: the 0.6 s fault pulse plus a 0.3 s jolt burst that lands on its end
# (seed 2 produced 0.88 s this way). A stuck motor would run past the 1 s watchdog instead.
MAX_ON_STRETCH_S = 1.0
PHASE_S = 6.0


# -- a hostile world -------------------------------------------------------------


def chaos_world(seed: int, outage_start_s: float, outage_len_s: float):
    """Obstacles that jump around every quarter second, plus one total sensor outage."""

    def world(t: float) -> Optional[WorldState]:
        if outage_start_s <= t < outage_start_s + outage_len_s:
            return None
        rng = random.Random(seed * 100003 + int(t * 4))
        state: WorldState = {}
        for sector in (Sector.LEFT, Sector.CENTER, Sector.RIGHT):
            state[sector] = rng.choice([None, None, rng.uniform(0.05, 2.5)])
            if rng.random() < 0.3:
                state[(FAR, sector)] = rng.uniform(1.0, 4.5)
        if rng.random() < 0.25:
            state[OVERHEAD] = rng.uniform(0.1, 2.0)
        return state

    return world


# -- misbehaving devices (each wraps a simulated one) ------------------------------


class Chaos:
    """Shared fault dial. ``rate`` is the chance that any one call misbehaves."""

    def __init__(self, seed: int, rate: float) -> None:
        self.rng = random.Random(seed)
        self.rate = rate
        self.enabled = True
        self._lock = threading.Lock()
        self.injected = 0

    def roll(self) -> float:
        with self._lock:
            value = self.rng.random()
            choice = self.rng.random()
        if self.enabled and value < self.rate:
            self.injected += 1
            return choice
        return -1.0


class ChaosTof:
    def __init__(self, inner: Any, chaos: Chaos) -> None:
        self._inner, self._chaos = inner, chaos
        self.sector = inner.sector

    def start(self) -> None:
        self._inner.start()

    def restart(self) -> None:
        if self._chaos.roll() >= 0 and self._chaos.rng.random() < 0.5:
            raise OSError("chaos: restart failed")
        self._inner.restart()

    def close(self) -> None:
        self._inner.close()

    def poll(self) -> Optional[RangeSample]:
        c = self._chaos.roll()
        if c < 0:
            return self._inner.poll()
        if c < 0.4:
            raise OSError("chaos: I2C error")
        if c < 0.55:
            time.sleep(0.05)  # a slow read
            return self._inner.poll()
        value = self._chaos.rng.choice(BAD_FLOATS)
        return RangeSample(value, self._chaos.rng.random() < 0.5, False, value)


class ChaosUs:
    def __init__(self, inner: Any, chaos: Chaos) -> None:
        self._inner, self._chaos = inner, chaos
        self.spec = inner.spec

    def start(self) -> None:
        self._inner.start()

    def close(self) -> None:
        self._inner.close()

    def measure(self) -> RangeSample:
        c = self._chaos.roll()
        if c < 0:
            return self._inner.measure()
        if c < 0.5:
            raise OSError("chaos: GPIO error")
        value = self._chaos.rng.choice(BAD_FLOATS)
        return RangeSample(value, self._chaos.rng.random() < 0.5, False, value)


class ChaosImu:
    def __init__(self, inner: SimImu, chaos: Chaos) -> None:
        self._inner, self._chaos = inner, chaos
        self.who_am_i = inner.who_am_i

    def read(self) -> ImuSample:
        c = self._chaos.roll()
        if c < 0:
            return self._inner.read()
        if c < 0.4:
            raise OSError("chaos: IMU I2C error")
        bad = self._chaos.rng.choice(BAD_FLOATS)
        picks = [self._chaos.rng.choice((0.0, 1.0, bad)) for _ in range(6)]
        return ImuSample(tuple(picks[:3]), tuple(picks[3:]), 30.0)  # type: ignore[arg-type]


class ChaosMotor:
    """Wraps a SimMotorDriver; failed writes raise and are NOT recorded as writes."""

    def __init__(self, inner: SimMotorDriver, chaos: Chaos) -> None:
        self.inner, self._chaos = inner, chaos

    def configure(self) -> None:
        if self._chaos.roll() >= 0 and self._chaos.rng.random() < 0.3:
            raise OSError("chaos: configure failed")
        self.inner.configure()

    def set_rtp(self, value: int) -> None:
        if self._chaos.roll() >= 0:
            raise OSError("chaos: motor write failed")
        self.inner.set_rtp(value)

    def stop(self) -> None:
        self.set_rtp(0)

    def standby(self) -> None:
        self.inner.standby()


class ChaosVision(threading.Thread):
    """Publishes hostile camera frames: huge counts, NaN boxes, odd classes and times."""

    CLASSES = ("person", "pole", "tree", "unicorn", "", "overhead_obstacle", "curb")

    def __init__(self, sink: Any, seed: int) -> None:
        super().__init__(name="chaos-vision", daemon=True)
        self._sink, self._rng = sink, random.Random(seed)
        self._halt = threading.Event()
        self.errors: List[str] = []
        self.frames = 0
        self.ready = threading.Event()

    def stats(self) -> Dict[str, object]:
        return {"frames": self.frames, "chaos": True}

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        self.ready.set()
        frame_id = 0
        while not self._halt.wait(self._rng.uniform(0.01, 0.08)):
            try:
                self._sink.update_vision(self._frame(frame_id))
            except Exception as exc:  # the sink must never raise on any frame
                self.errors.append(f"{type(exc).__name__}: {exc}")
            frame_id += 1
            self.frames += 1

    def _frame(self, frame_id: int) -> FrameDetections:
        rng = self._rng
        now = time.perf_counter_ns() + rng.choice((0, 0, 0, -10**9, 10**6))  # sometimes stale or early
        detections = []
        for _ in range(rng.choice((0, 1, 3, 40))):
            cx, cy = rng.uniform(-50, 700), rng.uniform(-50, 530)
            w, h = rng.choice((0.0, 1.0, 60.0, 900.0)), rng.choice((0.0, 1.0, 120.0, 900.0, float("nan")))
            try:
                box = BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                sector = rng.choice(list(Sector))
                detections.append(Detection(rng.randrange(15), rng.choice(self.CLASSES),
                                            rng.random(), box, sector))
            except ValueError:
                continue
        return FrameDetections(frame_id, now, now, now, rng.choice((640, 640, 1, 4000)),
                               rng.choice((480, 480, 1, 4000)), detections)


# -- the report ---------------------------------------------------------------------


@dataclass
class SoakReport:
    seconds: float
    seed: int
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations

    def describe(self) -> str:
        lines = [f"soak: {self.seconds:.0f} s, seed {self.seed}"]
        lines += [f"  {key}: {value}" for key, value in sorted(self.metrics.items())]
        if self.violations:
            lines += ["FAILED:"] + [f"  - {v}" for v in self.violations]
        else:
            lines.append("PASSED: no violations")
        return "\n".join(lines)


def _stretches(events: List[Tuple[float, int]]) -> Tuple[float, int]:
    """(longest time a motor stayed non-zero between two successful writes, #writes)."""

    longest, on_since = 0.0, None
    for stamp, value in events:
        if value > 0 and on_since is None:
            on_since = stamp
        elif value == 0 and on_since is not None:
            longest = max(longest, stamp - on_since)
            on_since = None
    return longest, len(events)


def _dump_stretch(events: List[Tuple[float, int]]) -> None:
    on_since, begin = None, 0
    for index, (stamp, value) in enumerate(events):
        if value > 0 and on_since is None:
            on_since, begin = stamp, index
        elif value == 0 and on_since is not None:
            if stamp - on_since > MAX_ON_STRETCH_S:
                for t, v in events[max(0, begin - 2): index + 2]:
                    print(f"  debug write t={t - on_since:+.3f} value={v}")
            on_since = None


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_soak(config: CaneConfig, seconds: float = 60.0, seed: int = 1, fault_rate: float = 0.08) -> SoakReport:
    """Run the full threaded system under fault injection and check the invariants."""

    report = SoakReport(seconds, seed)
    violations = report.violations
    threads_before = threading.active_count()
    clock = SimClock()
    chaos = Chaos(seed, fault_rate)
    world = chaos_world(seed, outage_start_s=seconds * 0.55, outage_len_s=min(4.0, seconds * 0.1))
    hardware = build_simulated_hardware(config, world, clock)
    hardware.tof = {k: ChaosTof(d, chaos) for k, d in hardware.tof.items()}  # type: ignore[assignment]
    hardware.ultrasonic = {k: ChaosUs(d, chaos) for k, d in hardware.ultrasonic.items()}  # type: ignore[assignment]
    sim_motors: Dict[str, SimMotorDriver] = {"haptic": hardware.haptic}  # type: ignore[dict-item]
    hardware.haptic = ChaosMotor(hardware.haptic, chaos)  # type: ignore[assignment,arg-type]
    if hardware.imu is not None:
        hardware.imu = ChaosImu(hardware.imu, chaos)  # type: ignore[assignment,arg-type]

    vision_holder: List[ChaosVision] = []

    def make_vision(fusion: Any) -> ChaosVision:
        worker = ChaosVision(fusion, seed)
        vision_holder.append(worker)
        return worker

    system = build_system(config, simulate=False, hardware=hardware, vision_factory=make_vision)
    seen_levels: Dict[str, int] = {}
    original_tick = system.runtime.tick

    def counting_tick(now_ns: Optional[int] = None):
        command = original_tick(now_ns)
        decision = system.runtime.last_decision
        key = f"{decision.level.name}/{getattr(decision, 'source', '?')}"
        seen_levels[key] = seen_levels.get(key, 0) + 1
        return command

    system.runtime.tick = counting_tick  # type: ignore[method-assign]

    stop = threading.Event()
    runtime_error: List[BaseException] = []
    rss_samples: List[float] = []

    def watch() -> None:
        started = time.monotonic()
        while not stop.wait(1.0):
            if time.monotonic() - started > min(10.0, seconds / 3):
                rss_samples.append(_rss_mb())

    watcher = threading.Thread(target=watch, daemon=True)
    started = time.monotonic()
    timer: Optional[threading.Timer] = None
    try:
        system.start()
        started = time.monotonic()
        timer = threading.Timer(seconds, stop.set)
        timer.daemon = True
        timer.start()
        watcher.start()
        system.run(stop)
    except BaseException as exc:  # noqa: BLE001 - any escape from the runtime is a failure
        runtime_error.append(exc)
    finally:
        if timer is not None:
            timer.cancel()
        stop.set()
        chaos.enabled = False  # a bus that fails during shutdown cannot be tested away
        try:
            system.close()
        except BaseException as exc:  # noqa: BLE001
            violations.append(f"shutdown raised {type(exc).__name__}: {exc}")
    elapsed = time.monotonic() - started
    watcher.join(timeout=2.0)

    if runtime_error:
        violations.append(f"runtime stopped with {type(runtime_error[0]).__name__}: {runtime_error[0]}")
    crashed = system.hub.crashed()
    if crashed:
        violations.append(f"sensor threads crashed: {crashed}")
    for worker in vision_holder:
        if worker.errors:
            violations.append(f"fusion rejected a camera frame: {worker.errors[0]}")

    for name, motor in sim_motors.items():
        longest, writes = _stretches(motor.events)
        report.metrics[f"{name} writes"] = writes
        report.metrics[f"{name} longest on (s)"] = round(longest, 3)
        if longest > MAX_ON_STRETCH_S:
            violations.append(f"{name} stayed on for {longest:.2f} s without a new write")
            if os.environ.get("CANE_SOAK_DEBUG"):
                _dump_stretch(motor.events)
        if motor.events and motor.events[-1][1] != 0:
            violations.append(f"{name} was left running at shutdown (last write {motor.events[-1][1]})")
        if not motor.events:
            violations.append(f"{name} never received a write")

    stats = system.runtime.stats
    report.metrics["loop iterations"] = stats.iterations
    report.metrics["loop max ms"] = round(stats.max_loop_ms, 2)
    report.metrics["loop overruns"] = stats.overruns
    report.metrics["faults injected"] = chaos.injected
    report.metrics["jolt pulses"] = stats.jolts
    report.metrics["decisions seen"] = dict(sorted(seen_levels.items()))
    expected = seconds * config.runtime.loop_hz
    if stats.iterations < 0.8 * expected:
        violations.append(f"loop ran {stats.iterations} times, expected about {expected:.0f}")
    if stats.max_loop_ms > 100.0:
        violations.append(f"one loop pass took {stats.max_loop_ms:.0f} ms (limit 100)")
    if stats.overruns > 0.02 * max(1, stats.iterations):
        violations.append(f"{stats.overruns} loop overruns")
    if elapsed > seconds + 5.0:
        violations.append(f"shutdown took {elapsed - seconds:.1f} s")

    time.sleep(0.2)
    leaked = threading.active_count() - threads_before
    report.metrics["leaked threads"] = leaked
    if leaked > 1:  # the one-shot timer may still be winding down
        violations.append(f"{leaked} threads still alive after shutdown")
    if len(rss_samples) >= 2:
        growth = rss_samples[-1] - rss_samples[0]
        report.metrics["memory growth MB"] = round(growth, 1)
        if growth > 40.0:
            violations.append(f"memory grew {growth:.0f} MB during the run")
    for needed in ("SENSOR_FAULT/fault", "WARNING/vision"):
        if needed not in seen_levels:
            report.metrics.setdefault("coverage warnings", []).append(f"never saw {needed}")
    return report


# -- scenario: nobody is steering -----------------------------------------------------


def run_watchdog_check(config: CaneConfig) -> List[str]:
    """If the runtime stops sending commands, the motors must go quiet by themselves."""

    from ..fusion import HapticCommand, HazardLevel
    from .haptics import HapticDriver

    problems: List[str] = []
    clock = SimClock()
    motor = SimMotorDriver("haptic", clock)
    driver = HapticDriver(motor, config.haptics)
    driver.start()
    try:
        driver.emit(HapticCommand(0, HazardLevel.CRITICAL, (Sector.CENTER,), 1.0, 100, "t"))
        time.sleep(0.5)
        if not any(v > 0 for _, v in motor.events):
            problems.append("watchdog check: the motors never started")
        time.sleep(config.haptics.command_timeout_s + 0.5)  # the "runtime" has frozen
        marker = clock()
        time.sleep(0.8)
        if [v for t, v in motor.events if t > marker and v > 0]:
            problems.append(
                f"watchdog check: still pulsing {config.haptics.command_timeout_s + 0.5:.1f} s after commands stopped")
    finally:
        driver.close()
    return problems


# -- scenario: the last "off" write is lost -------------------------------------------


def run_shutdown_check(config: CaneConfig) -> List[str]:
    """A lost 'motors off' write at shutdown must be retried, not left buzzing."""

    from ..fusion import HapticCommand, HazardLevel
    from .haptics import HapticDriver

    class LossyMotor:
        def __init__(self) -> None:
            self.value, self.fail_stops = 0, 0

        def configure(self) -> None:
            return None

        def set_rtp(self, value: int) -> None:
            self.value = value

        def stop(self) -> None:
            if self.fail_stops > 0:
                self.fail_stops -= 1
                raise OSError("bus error")
            self.value = 0

    motor = LossyMotor()
    driver = HapticDriver(motor, config.haptics)  # type: ignore[arg-type]
    driver.start()
    driver.emit(HapticCommand(0, HazardLevel.CRITICAL, (Sector.UNKNOWN,), 1.0, 100, "t"))
    time.sleep(0.15)
    motor.value, motor.fail_stops = 99, 2  # buzzing, and the first two off-writes are lost
    driver.close()
    return [] if motor.value == 0 else [f"shutdown check: the motors were left at {motor.value}"]


# -- scenario: a sensor thread keeps raising -------------------------------------------


def run_poller_error_check(config: CaneConfig) -> List[str]:
    """A sensor thread that keeps raising must restart in place; the cane must keep running."""

    from .app import run_until_signal

    problems: List[str] = []
    clock = SimClock()
    hardware = build_simulated_hardware(config, chaos_world(3, 1e9, 0), clock)
    sim_motors = {"haptic": hardware.haptic}
    system = build_system(config, simulate=False, with_vision=False, hardware=hardware)
    poller = system.hub.tof
    assert poller is not None

    class Fatal(BaseException):
        """Something a poller's own error handling does not catch."""

    calls = [0]
    original = poller.step

    def doomed() -> float:
        calls[0] += 1
        if calls[0] > 20:
            raise Fatal("simulated thread death")
        return original()

    poller.step = doomed  # type: ignore[method-assign]
    started = time.monotonic()
    code = run_until_signal(system, duration_s=2.5)
    took = time.monotonic() - started
    if code != 0:
        problems.append(f"poller check: the cane exited with code {code} instead of carrying on")
    if took < 2.0:
        problems.append(f"poller check: the cane stopped after {took:.1f} s")
    if poller.crashes < 2:
        problems.append("poller check: the thread died on its first error instead of restarting")
    if system.runtime.stats.iterations < 10:
        problems.append(
            f"poller check: the control loop only ran {system.runtime.stats.iterations} times"
        )
    for name, motor in sim_motors.items():
        if motor.events and motor.events[-1][1] != 0:  # type: ignore[attr-defined]
            problems.append(f"poller check: {name} left running")
    return problems


# -- scenario: signals --------------------------------------------------------------------


def run_signal_check(config_path: Optional[str], which: int = signal.SIGTERM) -> List[str]:
    """Start the real command line, stop it with a signal, expect a prompt clean exit."""

    command = [sys.executable, "-m", "yolo_pi.cane"]
    if config_path:
        command += ["--config", config_path]
    command += ["run", "--simulate"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.join(os.getcwd(), "src"), env.get("PYTHONPATH")]))
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, text=True)
    time.sleep(5.0)
    if process.poll() is not None:
        return [f"signal check: process exited early with {process.returncode}"]
    started = time.monotonic()
    process.send_signal(which)
    try:
        _, err = process.communicate(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        return [f"signal check: no exit within 10 s of {signal.Signals(which).name}"]
    problems = []
    if process.returncode != 0:
        problems.append(f"signal check: exit code {process.returncode} after {signal.Signals(which).name}")
    if "Traceback" in (err or ""):
        problems.append("signal check: a traceback was printed during shutdown")
    if time.monotonic() - started > 5.0:
        problems.append("signal check: shutdown took more than 5 s")
    return problems


# -- fuzzing the decision logic ---------------------------------------------------------------


def run_fuzz(config: CaneConfig, iterations: int = 100_000, seed: int = 1) -> List[str]:
    """Throw hostile numbers at fusion and the jolt detector; neither may raise or lock up."""

    rng = random.Random(seed)
    problems: List[str] = []
    ids = [s.sensor_id for s in config.tof.sensors] + [s.sensor_id for s in config.ultrasonic.sensors]
    overhead = [s.sensor_id for s in config.ultrasonic.sensors if s.role == "overhead"]
    fusion = PriorityFusion(config.fusion, config.priority, overhead)
    sectors = [Sector.LEFT, Sector.CENTER, Sector.RIGHT, Sector.UNKNOWN]
    now = 10**9
    valid_levels = {0, 1, 2, 3}
    for i in range(iterations):
        now += rng.choice((0, 1, 10**6, 5 * 10**7, -10**6))
        try:
            kind = rng.random()
            if kind < 0.6:
                fusion.update_sensor(SensorReading(
                    rng.choice(ids), rng.choice(list(SensorKind)), now + rng.choice((0, -3 * 10**8, 10**8)),
                    rng.choice(sectors), rng.choice((rng.uniform(0, 5), rng.choice(BAD_FLOATS), None)),
                    rng.random() < 0.9))
            elif kind < 0.7:
                w, h = rng.choice((640, 1, 4000)), rng.choice((480, 1, 4000))
                dets = []
                for _ in range(rng.choice((0, 2, 30))):
                    x, y = rng.uniform(-100, 800), rng.uniform(-100, 600)
                    bw, bh = rng.choice((0.0, 5.0, 100.0, 1e6)), rng.choice((0.0, 5.0, 200.0, float("inf"), float("nan")))
                    try:
                        dets.append(Detection(0, rng.choice(("person", "x", "")), rng.random(),
                                              BoundingBox(x, y, x + bw, y + bh), rng.choice(sectors)))
                    except ValueError:
                        pass
                fusion.update_vision(FrameDetections(i, now, now, now + rng.choice((0, 5)), w, h, dets))
            decision = fusion.evaluate(now)
            if decision.level.value not in valid_levels:
                problems.append(f"fuzz: invalid level {decision.level}")
                break
            if decision.nearest_distance_m is not None and not math.isfinite(decision.nearest_distance_m):
                problems.append(f"fuzz: non-finite distance in decision {decision.reason}")
                break
        except Exception as exc:  # noqa: BLE001
            problems.append(f"fuzz: fusion raised {type(exc).__name__}: {exc} at step {i}")
            break

    detector = JoltDetector(config.imu)
    t = 0
    for i in range(iterations):
        t += rng.choice((10**7, 10**7, 0, 3 * 10**8, -10**6))
        values = [rng.choice((0.0, 0.5, 1.0, -1.0, 300.0, rng.choice(BAD_FLOATS))) for _ in range(6)]
        try:
            detector.update(ImuSample(tuple(values[:3]), tuple(values[3:]), 30.0), t)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            problems.append(f"fuzz: jolt detector raised {type(exc).__name__}: {exc} at step {i}")
            break
    # After any amount of garbage it must still recognise a real knock.
    clean = JoltDetector(config.imu)
    for i in range(50):
        clean.update(ImuSample((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 30.0), (i + 1) * 10**7)
    for sample_value in BAD_FLOATS:
        clean.update(ImuSample((sample_value, 0.0, 1.0), (sample_value, 0.0, 0.0), 30.0), 6 * 10**8)
    knocked = 0
    base = 7 * 10**8
    for step in range(1, 16):
        theta = math.radians(50.0 * step / 15)
        sample = ImuSample((0.0, math.sin(theta), math.cos(theta)), (50.0 / 0.15, 0.0, 0.0), 30.0)
        if clean.update(sample, base + step * 10**7):
            knocked += 1
    if knocked != 1:
        problems.append("fuzz: the jolt detector stopped working after receiving non-finite readings")
    return problems


def run_everything(config: CaneConfig, seconds: float, seed: int, config_path: Optional[str] = None) -> Tuple[bool, str]:
    sections: List[str] = []
    ok = True

    problems = run_fuzz(config, seed=seed)
    sections.append("fuzz (100k hostile inputs): " + ("PASSED" if not problems else "FAILED\n  - " + "\n  - ".join(problems)))
    ok &= not problems

    problems = run_watchdog_check(config)
    sections.append("watchdog (runtime freezes): " + ("PASSED" if not problems else "FAILED\n  - " + "\n  - ".join(problems)))
    ok &= not problems

    for title, check in (("shutdown (lost 'off' write)", run_shutdown_check),
                         ("poller errors survived", run_poller_error_check)):
        problems = check(config)
        sections.append(f"{title}: " + ("PASSED" if not problems else "FAILED\n  - " + "\n  - ".join(problems)))
        ok &= not problems

    report = run_soak(config, seconds=seconds, seed=seed)
    sections.append(report.describe())
    ok &= report.ok

    for which in (signal.SIGTERM, signal.SIGINT):
        problems = run_signal_check(config_path, which)
        sections.append(f"{signal.Signals(which).name} shutdown: " + ("PASSED" if not problems else "FAILED\n  - " + "\n  - ".join(problems)))
        ok &= not problems
    return ok, "\n\n".join(sections)
