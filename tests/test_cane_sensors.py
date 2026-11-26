import time
import unittest

from yolo_pi.cane.config import ImuConfig, TofConfig, UltrasonicConfig, UltrasonicSensorSpec
from yolo_pi.cane.imu import ImuSample, MotionMonitor
from yolo_pi.cane.runtime import LockedFusion
from yolo_pi.cane.sensors import ImuPoller, SensorHub, ToFPoller, UltrasonicPoller
from yolo_pi.cane.tof import RangeSample
from yolo_pi.fusion import HazardLevel, SensorKind
from yolo_pi.types import Sector

MS = 1_000_000


class Clock:
    def __init__(self, start=1_000 * MS) -> None:
        self.now = start

    def __call__(self) -> int:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += int(ms * MS)


class Sink:
    def __init__(self) -> None:
        self.readings = []

    def update_sensor(self, reading) -> None:
        self.readings.append(reading)


class ScriptedTof:
    def __init__(self, script) -> None:
        self.script = list(script)
        self.restarts = 0

    def poll(self):
        item = self.script.pop(0) if self.script else None
        if isinstance(item, Exception):
            raise item
        return item

    def restart(self):
        self.restarts += 1


def sample(distance, valid=True):
    return RangeSample(distance, valid, False, None if distance is None else distance * 1000)


class ToFPollerTests(unittest.TestCase):
    def make(self, devices, sink=None, clock=None, **cfg):
        sink = sink or Sink()
        clock = clock or Clock()
        sectors = {"tof-left": Sector.LEFT, "tof-center": Sector.CENTER}
        return ToFPoller(devices, sectors, sink, TofConfig(**cfg), clock_ns=clock), sink, clock

    def test_publishes_timestamped_readings_with_sector(self) -> None:
        poller, sink, clock = self.make({"tof-left": ScriptedTof([sample(0.8)])})
        poller.step()
        reading = sink.readings[0]
        self.assertEqual(
            (reading.sensor_id, reading.kind, reading.sector, reading.distance_m, reading.valid),
            ("tof-left", SensorKind.TOF, Sector.LEFT, 0.8, True),
        )
        self.assertEqual(reading.timestamp_ns, clock.now)

    def test_no_new_sample_publishes_nothing(self) -> None:
        poller, sink, _ = self.make({"tof-left": ScriptedTof([None])})
        poller.step()
        self.assertEqual(sink.readings, [])
        self.assertEqual(poller.health["tof-left"].error_count, 0)

    def test_an_invalid_sample_is_published_as_invalid(self) -> None:
        poller, sink, _ = self.make({"tof-left": ScriptedTof([sample(None, valid=False)])})
        poller.step()
        self.assertFalse(sink.readings[0].valid)

    def test_one_failing_sensor_does_not_stop_the_others(self) -> None:
        devices = {
            "tof-left": ScriptedTof([OSError("bus"), OSError("bus")]),
            "tof-center": ScriptedTof([sample(0.6), sample(0.6)]),
        }
        poller, sink, _ = self.make(devices)
        poller.step()
        poller.step()
        self.assertEqual([r.sensor_id for r in sink.readings], ["tof-center", "tof-center"])
        self.assertEqual(poller.health["tof-left"].error_count, 2)
        self.assertIn("OSError", poller.health["tof-left"].last_error)

    def test_repeated_errors_restart_the_sensor_with_backoff(self) -> None:
        device = ScriptedTof([OSError("bus")] * 5)
        poller, _, clock = self.make({"tof-left": device}, reinit_after_errors=5, reinit_backoff_s=1.0)
        for _ in range(5):
            poller.step()
        self.assertEqual(device.restarts, 0)
        poller.step()  # the sixth step restarts it
        self.assertEqual(device.restarts, 1)
        self.assertEqual(poller.health["tof-left"].restarts, 1)
        self.assertEqual(poller.health["tof-left"].consecutive_errors, 0)

    def test_failed_restart_waits_before_trying_again(self) -> None:
        class Broken(ScriptedTof):
            def restart(self):
                self.restarts += 1
                raise OSError("still dead")

        device = Broken([OSError("bus")] * 20)
        poller, _, clock = self.make({"tof-left": device}, reinit_after_errors=2, reinit_backoff_s=1.0)
        for _ in range(8):
            poller.step()
        self.assertEqual(device.restarts, 1)  # not once per step
        clock.advance(1500)
        poller.step()
        poller.step()
        self.assertEqual(device.restarts, 2)


class FakeHcsr:
    def __init__(self, sensor_id, sector, results) -> None:
        self.spec = UltrasonicSensorSpec(sensor_id, 1, 2, sector)
        self.results = list(results)
        self.order = []

    def measure(self):
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class UltrasonicPollerTests(unittest.TestCase):
    def test_sensors_are_fired_one_at_a_time_in_turn(self) -> None:
        a = FakeHcsr("us-1", Sector.CENTER, [sample(1.0), sample(1.1)])
        b = FakeHcsr("us-2", Sector.CENTER, [sample(2.0), sample(2.1)])
        sink, clock = Sink(), Clock()
        poller = UltrasonicPoller({"us-1": a, "us-2": b}, sink, UltrasonicConfig(spacing_ms=60), clock)
        delays = [poller.step() for _ in range(4)]
        self.assertEqual([r.sensor_id for r in sink.readings], ["us-1", "us-2", "us-1", "us-2"])
        self.assertTrue(all(abs(d - 0.06) < 1e-9 for d in delays))
        self.assertEqual(sink.readings[0].kind, SensorKind.ULTRASONIC)

    def test_a_failure_is_counted_and_skipped(self) -> None:
        a = FakeHcsr("us-1", Sector.CENTER, [RuntimeError("gpio")])
        sink = Sink()
        poller = UltrasonicPoller({"us-1": a}, sink, UltrasonicConfig(), Clock())
        poller.step()
        self.assertEqual(sink.readings, [])
        self.assertEqual(poller.health["us-1"].error_count, 1)


class FakeImu:
    def __init__(self, samples) -> None:
        self.samples = list(samples)

    def read(self):
        item = self.samples.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ImuPollerTests(unittest.TestCase):
    def test_stillness_is_tracked_and_an_imu_failure_fails_towards_moving(self) -> None:
        still = ImuSample((0, 0, 1.0), (0, 0, 0), 30.0)
        clock = Clock()
        monitor = MotionMonitor(ImuConfig(still_after_s=0.0))
        poller = ImuPoller(FakeImu([still, OSError("bus")]), monitor, 20, clock)
        poller.step()
        self.assertTrue(monitor.is_still)
        poller.step()
        self.assertFalse(monitor.is_still)  # detector must not idle on a dead IMU


class FusionIntegrationTests(unittest.TestCase):
    """The behaviour the whole safety story rests on."""

    def run_fusion(self, devices):
        fusion, clock = LockedFusion(), Clock()
        sectors = {"tof-left": Sector.LEFT, "tof-center": Sector.CENTER}
        poller = ToFPoller(devices, sectors, fusion, TofConfig(), clock_ns=clock)
        poller.step()
        clock.advance(10)
        return fusion.evaluate(clock.now)

    def test_empty_path_reads_as_clear_not_as_a_sensor_fault(self) -> None:
        clear = RangeSample(2.0, True, True, 8190.0)
        decision = self.run_fusion({"tof-left": ScriptedTof([clear]), "tof-center": ScriptedTof([clear])})
        self.assertEqual(decision.level, HazardLevel.CLEAR)

    def test_close_obstacle_on_the_left_is_critical_in_the_left_sector(self) -> None:
        decision = self.run_fusion({
            "tof-left": ScriptedTof([sample(0.3)]),
            "tof-center": ScriptedTof([sample(1.8)]),
        })
        self.assertEqual(decision.level, HazardLevel.CRITICAL)
        self.assertEqual(decision.sectors, (Sector.LEFT,))

    def test_silent_sensors_become_a_fault_once_readings_go_stale(self) -> None:
        fusion, clock = LockedFusion(), Clock()
        poller = ToFPoller({"tof-left": ScriptedTof([sample(1.5)])}, {"tof-left": Sector.LEFT},
                           fusion, TofConfig(), clock_ns=clock)
        poller.step()
        self.assertEqual(fusion.evaluate(clock.now + 100 * MS).level, HazardLevel.CLEAR)
        self.assertEqual(fusion.evaluate(clock.now + 400 * MS).level, HazardLevel.SENSOR_FAULT)


class HubTests(unittest.TestCase):
    def test_health_report_and_crash_detection(self) -> None:
        clock = Clock()
        poller = ToFPoller({"tof-left": ScriptedTof([sample(0.9)])}, {"tof-left": Sector.LEFT},
                           Sink(), TofConfig(), clock_ns=clock)
        poller.step()
        hub = SensorHub([poller], tof=poller)
        report = hub.health(clock.now + 20 * MS)
        self.assertEqual(report["tof-left"]["ok"], 1)
        self.assertEqual(report["tof-left"]["age_ms"], 20.0)
        self.assertEqual(hub.crashed(), [])

    def test_a_poller_bug_is_restarted_not_fatal(self) -> None:
        class Boom(ToFPoller):
            def step(self):
                raise ValueError("boom")

        poller = Boom({}, {}, Sink(), TofConfig())
        poller.start()
        self.assertTrue(poller.is_alive(), "a poller error must not kill the thread")
        deadline = time.monotonic() + 2.0
        while poller.crashes < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertGreaterEqual(poller.crashes, 2, "the poller should keep retrying")
        self.assertIn("boom", poller.last_error or "")
        self.assertEqual(SensorHub([poller], tof=poller).crashed(), [])
        poller.stop()
        poller.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
