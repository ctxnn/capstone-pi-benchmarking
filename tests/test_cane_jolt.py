import math
import unittest

from yolo_pi.cane.config import ImuConfig
from yolo_pi.cane.imu import ImuSample, JoltDetector, MotionMonitor
from yolo_pi.cane.sensors import ImuPoller, SensorHub

MS = 1_000_000
STEP_MS = 10


def tilt_sample(theta_deg, rate_dps, axis="x"):
    th = math.radians(theta_deg)
    if axis == "x":
        return ImuSample((0.0, math.sin(th), math.cos(th)), (rate_dps, 0.0, 0.0), 30.0)
    return ImuSample((math.sin(th), 0.0, math.cos(th)), (0.0, -rate_dps, 0.0), 30.0)


def drive(detector, total_deg, over_s, hold_s=0.5, axis="x", start_ms=0):
    """Tilt by total_deg over over_s seconds, then hold. Returns the events produced."""

    events, now, theta = [], start_ms * MS, 0.0
    steps = max(1, int(over_s * 1000 / STEP_MS))
    rate = total_deg / over_s
    for _ in range(steps):
        theta += total_deg / steps
        now += STEP_MS * MS
        event = detector.update(tilt_sample(theta, rate, axis), now)
        if event:
            events.append(event)
    for _ in range(int(hold_s * 1000 / STEP_MS)):
        now += STEP_MS * MS
        event = detector.update(tilt_sample(theta, 0.0, axis), now)
        if event:
            events.append(event)
    return events


def settled(detector, seconds=0.3):
    now = 0
    for _ in range(int(seconds * 1000 / STEP_MS)):
        now += STEP_MS * MS
        detector.update(tilt_sample(0.0, 0.0), now)
    return now


class JoltDetectorTests(unittest.TestCase):
    def test_a_sudden_fifty_degree_tilt_is_a_jolt(self) -> None:
        events = drive(JoltDetector(ImuConfig()), 50, 0.15)
        self.assertEqual(len(events), 1)
        self.assertGreaterEqual(events[0].angle_deg, 25.0)

    def test_the_same_tilt_done_slowly_is_not(self) -> None:
        self.assertEqual(drive(JoltDetector(ImuConfig()), 50, 2.0), [])

    def test_a_small_quick_tilt_is_not(self) -> None:
        self.assertEqual(drive(JoltDetector(ImuConfig()), 10, 0.1), [])

    def test_it_does_not_matter_which_way_the_chip_is_mounted(self) -> None:
        self.assertEqual(len(drive(JoltDetector(ImuConfig()), 50, 0.15, axis="y")), 1)

    def test_a_still_cane_never_fires(self) -> None:
        detector = JoltDetector(ImuConfig())
        settled(detector, 2.0)
        self.assertEqual(detector.last_angle_deg < 1.0, True)

    def test_turning_about_the_vertical_is_not_a_tilt(self) -> None:
        detector, now, events = JoltDetector(ImuConfig()), 0, []
        for _ in range(100):
            now += STEP_MS * MS
            event = detector.update(ImuSample((0.0, 0.0, 1.0), (0.0, 0.0, 400.0), 30.0), now)
            if event:
                events.append(event)
        self.assertEqual(events, [])

    def test_an_impact_spike_on_the_accelerometer_alone_is_not_a_rotation(self) -> None:
        detector = JoltDetector(ImuConfig())
        now = settled(detector)
        events = []
        for accel in ((3.0, 0.0, 0.2), (0.0, 3.5, 0.1), (-2.5, 0.0, 0.0)):
            now += STEP_MS * MS
            event = detector.update(ImuSample(accel, (0.0, 0.0, 0.0), 30.0), now)
            if event:
                events.append(event)
        self.assertEqual(events, [])

    def test_the_cooldown_stops_a_shake_from_firing_repeatedly(self) -> None:
        detector = JoltDetector(ImuConfig(jolt_cooldown_s=1.0))
        first = drive(detector, 50, 0.15, hold_s=0.1)
        second = drive(detector, -50, 0.15, hold_s=0.1, start_ms=300)
        self.assertEqual((len(first), len(second)), (1, 0))

    def test_a_jolt_fires_again_after_the_cooldown(self) -> None:
        detector = JoltDetector(ImuConfig(jolt_cooldown_s=0.5))
        self.assertEqual(len(drive(detector, 50, 0.15, hold_s=0.8)), 1)
        self.assertEqual(len(drive(detector, -50, 0.15, hold_s=0.2, start_ms=2000)), 1)


class FakeImu:
    def __init__(self, samples) -> None:
        self.samples = list(samples)

    def read(self):
        return self.samples.pop(0)


class ImuPollerJoltTests(unittest.TestCase):
    def test_a_jolt_is_queued_once_and_only_a_jolt(self) -> None:
        now = [0]
        samples, theta = [], 0.0
        for _ in range(30):
            samples.append(tilt_sample(0.0, 0.0))
        for _ in range(15):
            theta += 50 / 15
            samples.append(tilt_sample(theta, 333.0))
        poller = ImuPoller(FakeImu(samples), MotionMonitor(ImuConfig()), 10, lambda: now[0],
                           jolt=JoltDetector(ImuConfig()))
        found = []
        for _ in samples:
            now[0] += STEP_MS * MS
            poller.step()
            event = poller.pop_jolt()
            if event:
                found.append(event)
        self.assertEqual(len(found), 1)
        self.assertIsNone(SensorHub([poller], imu=poller).pop_jolt())


class RuntimeJoltTests(unittest.TestCase):
    def test_a_jolt_bursts_every_motor_once_and_is_counted(self) -> None:
        from yolo_pi.cane.config import CaneConfig, RuntimeConfig
        from yolo_pi.cane.imu import JoltEvent
        from yolo_pi.cane.runtime import CaneRuntime, LockedFusion
        from yolo_pi.fusion import RecordingHapticSink

        class Sink(RecordingHapticSink):
            def __init__(self):
                super().__init__()
                self.bursts = []

            def burst(self, intensity, duration_ms):
                self.bursts.append((intensity, duration_ms))

        config = CaneConfig(runtime=RuntimeConfig(event_log=None))
        pending = [JoltEvent(0, 40.0)]
        sink = Sink()
        runtime = CaneRuntime(config, LockedFusion.from_config(config), sink, clock_ns=lambda: 0,
                              imu_events=lambda: pending.pop() if pending else None)
        runtime.tick(0)
        runtime.tick(20 * MS)
        self.assertEqual(sink.bursts, [(config.imu.jolt_intensity, config.imu.jolt_pulse_ms)])
        self.assertEqual(runtime.status(0)["loop"]["jolts"], 1)

    def test_no_jolt_means_no_burst_whatever_else_the_imu_sees(self) -> None:
        from yolo_pi.cane.config import CaneConfig, RuntimeConfig
        from yolo_pi.cane.runtime import CaneRuntime, LockedFusion
        from yolo_pi.fusion import RecordingHapticSink

        class Sink(RecordingHapticSink):
            bursts = 0

            def burst(self, *args):
                Sink.bursts += 1

        config = CaneConfig(runtime=RuntimeConfig(event_log=None))
        sink = Sink()
        runtime = CaneRuntime(config, LockedFusion.from_config(config), sink, clock_ns=lambda: 0)
        for i in range(10):
            runtime.tick(i * 20 * MS)
        self.assertEqual(Sink.bursts, 0)
