import unittest

from yolo_pi.cane.config import HapticsConfig
from yolo_pi.cane.haptics import (
    Drv2605Erm,
    HapticDriver,
    amplitude_to_rtp,
    od_clamp_for,
    play_pattern_once,
)
from yolo_pi.cane.patterns import (
    amplitude_at,
    build_cycle,
    cycle_length,
    one_shot,
    urgency_for,
)
from yolo_pi.cane.priority import CaneCommand
from yolo_pi.fusion import HazardLevel
from yolo_pi.types import Sector

CONFIG = HapticsConfig()


class FakeRegs:
    address = 0x5A

    def __init__(self, status=0xC0, fail=False) -> None:
        self.regs = {0x00: status, 0x1A: 0xB6, 0x1D: 0x88}
        self.writes = []
        self.fail = fail

    def write_reg(self, reg, value):
        if self.fail:
            raise OSError("I2C error")
        self.writes.append((reg, value))
        self.regs[reg] = value

    def read_reg(self, reg):
        return self.regs.get(reg, 0)

    def read_regs(self, reg, length):
        return bytes(self.regs.get(reg + i, 0) for i in range(length))


class FakeMotor:
    def __init__(self, fail_times=0) -> None:
        self.values = []
        self.fail_times = fail_times
        self.configured = 0

    def configure(self):
        self.configured += 1

    def set_rtp(self, value):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError("bus error")
        self.values.append(value)

    def stop(self):
        self.values.append(0)


def make_driver(clock=lambda: 0.0, **motor_kwargs):
    motor = FakeMotor(**motor_kwargs)
    return HapticDriver(motor, CONFIG, clock=clock, sleep=lambda _s: None), motor


def cmd(sectors, intensity=1.0, urgency=1.0, source="range", level=HazardLevel.CRITICAL):
    return CaneCommand(0, level, tuple(sectors), intensity, None, "test", source, urgency)


def on_durations(cycle):
    return [seconds for amplitude, seconds in cycle if amplitude > 0]


def taps(cycle):
    return len(on_durations(cycle))


class MappingTests(unittest.TestCase):
    def test_od_clamp_for_a_three_volt_motor(self) -> None:
        self.assertEqual(od_clamp_for(3.0), 142)
        self.assertEqual(od_clamp_for(100.0), 255)

    def test_zero_intensity_is_off_and_others_respect_the_floor(self) -> None:
        self.assertEqual(amplitude_to_rtp(0.0, 0.3), 0)
        self.assertEqual(amplitude_to_rtp(1.0, 0.3), 127)
        weak, mid, strong = (amplitude_to_rtp(i, 0.3) for i in (0.35, 0.65, 1.0))
        self.assertLess(weak, mid)
        self.assertLess(mid, strong)
        self.assertGreaterEqual(amplitude_to_rtp(0.01, 0.3), int(0.3 * 127))


class Drv2605Tests(unittest.TestCase):
    def test_configure_selects_open_loop_erm_rtp(self) -> None:
        regs = FakeRegs()
        driver = Drv2605Erm(regs, 3.0, sleep=lambda _s: None)
        driver.configure()
        self.assertEqual(regs.regs[0x17], 142)  # OD_CLAMP for 3 V
        self.assertEqual(regs.regs[0x1A] & 0x80, 0)  # ERM, not LRA
        self.assertEqual(regs.regs[0x1D] & 0x20, 0x20)  # open loop
        self.assertEqual(regs.regs[0x1D] & 0x08, 0)  # signed RTP, full scale 127
        self.assertEqual(regs.regs[0x01], 0x05)  # RTP mode
        self.assertEqual(regs.regs[0x02], 0)  # starts silent
        self.assertEqual(driver.device_id, 6)  # DRV2605L
        self.assertLess(regs.writes.index((0x01, 0x80)), regs.writes.index((0x01, 0x05)))

    def test_set_rtp_is_clamped_to_full_scale(self) -> None:
        regs = FakeRegs()
        driver = Drv2605Erm(regs, 3.0, sleep=lambda _s: None)
        driver.set_rtp(500)
        driver.set_rtp(-4)
        self.assertEqual([v for r, v in regs.writes if r == 0x02], [127, 0])

    def test_standby_silences_then_sleeps(self) -> None:
        regs = FakeRegs()
        Drv2605Erm(regs, 3.0, sleep=lambda _s: None).standby()
        self.assertEqual(regs.writes[-2:], [(0x02, 0), (0x01, 0x40)])


class PatternLanguageTests(unittest.TestCase):
    """What each hazard feels like on the one shared channel."""

    def test_direction_is_the_number_of_taps(self) -> None:
        counts = {s: taps(build_cycle(cmd([s]), CONFIG)) for s in (Sector.LEFT, Sector.CENTER, Sector.RIGHT)}
        self.assertEqual(counts, {Sector.LEFT: 1, Sector.CENTER: 2, Sector.RIGHT: 3})

    def test_two_sides_at_once_play_both_groups_with_a_longer_pause_between(self) -> None:
        cycle = build_cycle(cmd([Sector.LEFT, Sector.RIGHT]), CONFIG)
        self.assertEqual(taps(cycle), 4)  # 1 + 3
        pauses = [seconds for amplitude, seconds in cycle if amplitude == 0]
        self.assertGreater(max(pauses), 2 * min(pauses))

    def test_nearer_is_faster_and_stronger(self) -> None:
        far = build_cycle(cmd([Sector.CENTER], 0.65, 0.0), CONFIG)
        near = build_cycle(cmd([Sector.CENTER], 1.0, 1.0), CONFIG)
        self.assertLess(cycle_length(near), 0.6 * cycle_length(far))
        self.assertLess(max(on_durations(near)), max(on_durations(far)))
        self.assertEqual(taps(near), taps(far))  # speed never changes the direction code

    def test_tempo_changes_smoothly_with_urgency(self) -> None:
        lengths = [cycle_length(build_cycle(cmd([Sector.RIGHT], 0.8, u), CONFIG)) for u in (0, .25, .5, .75, 1)]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_the_gap_between_groups_always_stays_clearly_longer_than_the_gap_between_taps(self) -> None:
        for u in (0.0, 0.5, 1.0):
            cycle = build_cycle(cmd([Sector.RIGHT], 1.0, u), CONFIG)
            pauses = [seconds for amplitude, seconds in cycle if amplitude == 0]
            self.assertGreaterEqual(pauses[-1], 1.8 * pauses[0], msg=f"urgency {u}")

    def test_overhead_is_two_long_pulses_and_fault_is_one_long_weak_pulse(self) -> None:
        overhead = build_cycle(cmd([Sector.CENTER], 0.8, 0.5, "overhead"), CONFIG)
        fault = build_cycle(cmd([Sector.UNKNOWN], 0.35, 0.0, "fault", HazardLevel.SENSOR_FAULT), CONFIG)
        self.assertEqual(on_durations(overhead), [0.25, 0.25])
        self.assertEqual(taps(fault), 1)
        self.assertGreater(on_durations(fault)[0], 0.5)
        tap = build_cycle(cmd([Sector.LEFT]), CONFIG)
        self.assertGreater(min(on_durations(overhead)), 2 * max(on_durations(tap)))

    def test_every_pattern_is_distinguishable_from_every_other(self) -> None:
        shapes = {
            "left": build_cycle(cmd([Sector.LEFT], 0.8, 0.5), CONFIG),
            "center": build_cycle(cmd([Sector.CENTER], 0.8, 0.5), CONFIG),
            "right": build_cycle(cmd([Sector.RIGHT], 0.8, 0.5), CONFIG),
            "overhead": build_cycle(cmd([Sector.CENTER], 0.8, 0.5, "overhead"), CONFIG),
            "fault": build_cycle(cmd([Sector.UNKNOWN], 0.35, 0.0, "fault", HazardLevel.SENSOR_FAULT), CONFIG),
        }
        signatures = {name: (taps(c), round(max(on_durations(c)), 2)) for name, c in shapes.items()}
        self.assertEqual(len(set(signatures.values())), 5, msg=str(signatures))

    def test_a_clear_command_is_silent(self) -> None:
        self.assertEqual(build_cycle(cmd([], 0.0, 0.0, level=HazardLevel.CLEAR), CONFIG), ())
        self.assertEqual(amplitude_at((), 1.0), 0.0)

    def test_amplitude_at_walks_through_the_cycle_and_repeats(self) -> None:
        cycle = ((1.0, 0.1), (0.0, 0.2))
        self.assertEqual([amplitude_at(cycle, t) for t in (0.0, 0.05, 0.15, 0.29, 0.31, 0.45)],
                         [1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    def test_urgency_runs_from_zero_at_the_warning_distance_to_one_when_very_near(self) -> None:
        self.assertEqual(urgency_for(1.0, 1.0, 0.15), 0.0)
        self.assertEqual(urgency_for(1.5, 1.0, 0.15), 0.0)
        self.assertEqual(urgency_for(0.1, 1.0, 0.15), 1.0)
        values = [urgency_for(d, 1.0, 0.15) for d in (0.95, 0.7, 0.5, 0.3, 0.15)]
        self.assertEqual(values, sorted(values))

    def test_one_shot_drops_the_closing_pause(self) -> None:
        cycle = build_cycle(cmd([Sector.LEFT]), CONFIG)
        self.assertEqual(one_shot(cycle)[-1][0], 1.0)


class DriverPlaybackTests(unittest.TestCase):
    def test_target_follows_the_pattern_over_time(self) -> None:
        now = [10.0]
        driver, _ = make_driver(clock=lambda: now[0])
        driver.emit(cmd([Sector.LEFT]))
        self.assertEqual(driver.target(10.01), 127)  # first tap is on
        self.assertEqual(driver.target(10.1), 0)  # then the pause
        self.assertEqual(driver.target(10.01 + cycle_length(build_cycle(cmd([Sector.LEFT]), CONFIG))), 127)

    def test_the_same_pattern_keeps_its_phase_and_a_new_direction_restarts_it_in_a_gap(self) -> None:
        now = [10.0]
        driver, _ = make_driver(clock=lambda: now[0])
        driver.emit(cmd([Sector.LEFT]))
        start = driver._start
        now[0] = 10.05
        driver.emit(cmd([Sector.LEFT]))
        self.assertEqual(driver._start, start)
        now[0] = 10.15  # in the pause after the tap
        driver.emit(cmd([Sector.RIGHT]))
        self.assertEqual(driver._start, 10.15)

    def test_a_flickering_direction_cannot_hold_the_motor_on(self) -> None:
        now = [10.0]
        driver, _ = make_driver(clock=lambda: now[0])
        sectors = [Sector.LEFT, Sector.CENTER, Sector.RIGHT]
        on_for, longest = 0.0, 0.0
        for i in range(400):  # a new direction every 25 ms for 10 s
            driver.emit(cmd([sectors[i % 3]], 1.0, 1.0))
            now[0] += 0.025
            if driver.target(now[0]) > 0:
                on_for += 0.025
                longest = max(longest, on_for)
            else:
                on_for = 0.0
        self.assertLess(longest, 0.4)

    def test_the_backstop_forces_a_rest_if_the_motor_would_stay_on_too_long(self) -> None:
        import time

        from dataclasses import replace

        motor = FakeMotor()
        driver = HapticDriver(motor, replace(CONFIG, command_timeout_s=60.0))
        driver.burst(1.0, 3000)  # a stuck "on" request, far longer than any real pattern
        driver.start()
        time.sleep(1.4)
        driver.close()
        self.assertGreaterEqual(motor.values.count(0), 2)  # rested mid-burst, then parked

    def test_a_faster_tempo_keeps_the_position_in_the_cycle(self) -> None:
        now = [10.0]
        driver, _ = make_driver(clock=lambda: now[0])
        slow = cmd([Sector.CENTER], 0.65, 0.0)
        fast = cmd([Sector.CENTER], 1.0, 1.0)
        driver.emit(slow)
        now[0] = 10.0 + 0.5 * cycle_length(build_cycle(slow, CONFIG))
        driver.emit(fast)
        fraction = ((now[0] - driver._start) % driver._cycle_len) / driver._cycle_len
        self.assertAlmostEqual(fraction, 0.5, places=6)

    def test_apply_writes_only_on_change(self) -> None:
        driver, motor = make_driver()
        for value in (100, 100, 0, 0):
            driver._apply(value)
        self.assertEqual(motor.values, [100, 0])

    def test_a_failing_driver_is_retried_and_never_raises(self) -> None:
        driver, motor = make_driver(fail_times=2)
        for _ in range(4):
            driver._apply(90)
        self.assertEqual(motor.values, [90])
        self.assertEqual(driver.error_count, 2)

    def test_repeated_failures_trigger_a_reconfigure_at_most_once_a_second(self) -> None:
        now = [100.0]
        driver, motor = make_driver(clock=lambda: now[0], fail_times=50)
        for _ in range(6):
            driver._apply(90)
        self.assertEqual(motor.configured, 1)
        now[0] = 102.0
        driver._apply(90)
        self.assertEqual(motor.configured, 2)

    def test_a_burst_overlays_the_pattern_then_ends_by_itself(self) -> None:
        now = [0.0]
        driver, _ = make_driver(clock=lambda: now[0])
        driver.burst(0.8, 300)
        expected = amplitude_to_rtp(0.8, CONFIG.amplitude_floor)
        self.assertEqual(driver.target(0.1), expected)
        self.assertEqual(driver.target(0.4), 0)

    def test_a_burst_does_not_cancel_a_stronger_alert(self) -> None:
        now = [0.0]
        driver, _ = make_driver(clock=lambda: now[0])
        driver.emit(cmd([Sector.LEFT], 1.0, 1.0))
        driver.burst(0.5, 300)
        self.assertEqual(driver.target(0.01), 127)

    def test_it_falls_silent_when_commands_stop_arriving(self) -> None:
        import time
        from dataclasses import replace

        motor = FakeMotor()
        driver = HapticDriver(motor, replace(CONFIG, command_timeout_s=0.15))
        driver.start()
        try:
            driver.emit(cmd([Sector.CENTER]))  # one command, then the "runtime" goes quiet
            time.sleep(0.6)
            settled = len(motor.values)
            time.sleep(0.4)
            self.assertEqual(len(motor.values), settled)
            self.assertEqual(motor.values[-1], 0)
            self.assertIn(127, motor.values)
        finally:
            driver.close()

    def test_the_threaded_driver_plays_the_taps_then_parks_on_close(self) -> None:
        import time

        driver, motor = make_driver(clock=time.monotonic)
        driver.start()
        driver.emit(cmd([Sector.RIGHT]))
        time.sleep(0.5)
        driver.close()
        self.assertGreaterEqual(motor.values.count(127), 3)  # three taps in the first 500 ms
        self.assertEqual(motor.values[-1], 0)

    def test_play_pattern_once_plays_exactly_the_group_and_stops(self) -> None:
        motor = FakeMotor()
        slept = []
        play_pattern_once(motor, cmd([Sector.CENTER], 1.0, 1.0), CONFIG, slept.append)
        self.assertEqual(motor.values, [127, 0, 127, 0])  # tap, gap, tap, stop
        self.assertEqual(len(slept), 3)

    def test_the_startup_sweep_plays_one_two_then_three_taps(self) -> None:
        driver, motor = make_driver()
        driver.sweep()
        self.assertEqual(motor.values.count(0) >= 3, True)
        self.assertEqual(sum(1 for v in motor.values if v > 0), 1 + 2 + 3)

    def test_close_without_start_still_silences_the_motors(self) -> None:
        driver, motor = make_driver()
        driver.close()
        self.assertEqual(motor.values, [0])


if __name__ == "__main__":
    unittest.main()
