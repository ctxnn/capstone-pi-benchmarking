"""End-to-end simulation tests for the SmartCane v2 pipeline.

Each test replays one scripted real-world situation through the production
fusion engine and asserts the decision metrics the Capstone report promises:
reaction latency, hysteresis chatter, false alarms against ground truth and the
per-tick evaluation budget.
"""

import pytest

from yolo_pi.simulator import SimulationResult, run_all_scenarios

SCENARIO_ORDER = [
    "approach_wall_head_on",
    "narrow_pole_center",
    "glass_door_center",
    "ultrasonic_multipath_spike",
    "multi_sector_simultaneous",
    "range_blind_stairs_ahead",
    "long_range_vehicle_and_flicker",
    "boundary_hover_chatter",
    "cane_sweep_imu_yaw_comp",
    "steep_downward_tilt_ground_echo",
    "partial_sensor_dropout",
    "total_range_blackout_and_recovery",
    "fast_approaching_ttc",
    "stationary_anti_fatigue",
]

MAX_P95_TICK_MS = 2.0
MAX_TICK_MS = 20.0


@pytest.fixture(scope="module")
def scorecards():
    results = run_all_scenarios(seed=42)
    return {result.name: result for result in results}


def _result(scorecards, name: str) -> SimulationResult:
    assert name in scorecards, f"scenario {name} did not run"
    result = scorecards[name]
    failed = [check for check, ok in result.checks.items() if not ok]
    assert result.passed, (
        f"{name} failed {failed}: {result.summary} "
        f"(chatter={result.chatter_transitions}, "
        f"false_critical={result.false_critical_count}, "
        f"false_warning={result.false_warning_count})"
    )
    assert result.p95_tick_eval_ms < MAX_P95_TICK_MS, (
        f"{name} p95 evaluation {result.p95_tick_eval_ms:.3f} ms exceeds {MAX_P95_TICK_MS} ms"
    )
    assert result.max_tick_eval_ms < MAX_TICK_MS, (
        f"{name} worst evaluation {result.max_tick_eval_ms:.3f} ms exceeds {MAX_TICK_MS} ms"
    )
    assert result.ticks > 40, f"{name} only produced {result.ticks} decisions"
    return result


def test_all_fourteen_scenarios_are_reported(scorecards) -> None:
    assert list(scorecards) == SCENARIO_ORDER


def test_simulation_is_deterministic(scorecards) -> None:
    repeat = {result.name: result for result in run_all_scenarios(seed=42)}
    for name, first in scorecards.items():
        second = repeat[name]
        assert (
            first.passed,
            first.checks,
            first.ticks,
            first.chatter_transitions,
            first.false_critical_count,
            first.false_warning_count,
            first.reaction_latency_ms,
        ) == (
            second.passed,
            second.checks,
            second.ticks,
            second.chatter_transitions,
            second.false_critical_count,
            second.false_warning_count,
            second.reaction_latency_ms,
        ), f"{name} is not reproducible for a fixed seed"


def test_every_tick_stays_inside_the_latency_budget(scorecards) -> None:
    for name, result in scorecards.items():
        assert result.p95_tick_eval_ms < MAX_P95_TICK_MS, name
        assert result.max_tick_eval_ms < MAX_TICK_MS, name


def test_approach_wall_head_on(scorecards) -> None:
    result = _result(scorecards, "approach_wall_head_on")
    assert result.checks["vision_advisory_before_1.80m"]
    assert result.checks["warning_within_120ms_of_1.00m"]
    assert result.checks["critical_within_120ms_of_0.50m"]
    assert result.reaction_latency_ms is not None and result.reaction_latency_ms <= 120.0
    assert result.chatter_transitions <= 6
    assert result.false_critical_count == 0
    assert result.false_warning_count == 0


def test_narrow_pole_center(scorecards) -> None:
    result = _result(scorecards, "narrow_pole_center")
    assert result.checks["center_warning_sustained"]
    assert result.checks["closer_return_trusted"]
    assert result.checks["ultrasonic_background_rejected"]
    assert result.false_critical_count == 0
    assert result.false_warning_count == 0


def test_glass_door_center(scorecards) -> None:
    result = _result(scorecards, "glass_door_center")
    assert result.checks["tof_alive_but_clear"]
    assert result.checks["center_warning_sustained"]
    assert result.checks["center_health_ok"]
    assert result.chatter_transitions == 0
    assert result.false_critical_count == 0


def test_ultrasonic_multipath_spike(scorecards) -> None:
    result = _result(scorecards, "ultrasonic_multipath_spike")
    assert result.false_critical_count == 0
    assert result.false_warning_count == 0
    assert result.chatter_transitions <= 4


def test_multi_sector_simultaneous(scorecards) -> None:
    result = _result(scorecards, "multi_sector_simultaneous")
    assert result.checks["left_motor_critical_1.0_100ms"]
    assert result.checks["right_motor_warning_0.6_0.88"]
    assert result.checks["center_motor_clear_0.0"]
    assert result.false_critical_count == 0
    assert result.chatter_transitions <= 2


def test_range_blind_stairs_ahead(scorecards) -> None:
    result = _result(scorecards, "range_blind_stairs_ahead")
    assert result.checks["vision_advisory_seen"]
    assert result.checks["vision_warning_seen"]
    assert result.checks["never_vision_critical"]
    assert result.checks["range_sensors_blind"]
    assert result.false_critical_count == 0


def test_long_range_vehicle_and_flicker(scorecards) -> None:
    result = _result(scorecards, "long_range_vehicle_and_flicker")
    assert result.checks["right_flicker_ignored"]
    assert result.checks["left_never_warning_or_critical"]
    assert result.reaction_latency_ms is not None and result.reaction_latency_ms <= 400.0
    assert result.false_critical_count == 0
    assert result.false_warning_count == 0


def test_boundary_hover_chatter(scorecards) -> None:
    result = _result(scorecards, "boundary_hover_chatter")
    assert result.checks["critical_held_while_hovering_0.50"]
    assert result.checks["warning_held_while_hovering_1.00"]
    assert result.chatter_transitions <= 3


def test_cane_sweep_imu_yaw_comp(scorecards) -> None:
    result = _result(scorecards, "cane_sweep_imu_yaw_comp")
    assert result.checks["yaw_lag_pushed_raw_box_sideways"]
    assert result.checks["compensation_keeps_every_frame_center"]
    assert result.checks["center_carries_person_label"]
    assert result.checks["no_side_alerts"]
    assert result.false_critical_count == 0


def test_steep_downward_tilt_ground_echo(scorecards) -> None:
    result = _result(scorecards, "steep_downward_tilt_ground_echo")
    assert result.checks["floor_return_suppressed"]
    assert result.checks["cane_pitch_tracked"]
    assert result.checks["real_obstacle_critical_within_150ms"]
    assert result.false_warning_count == 0
    assert result.false_critical_count == 0


def test_partial_sensor_dropout(scorecards) -> None:
    result = _result(scorecards, "partial_sensor_dropout")
    assert result.checks["left_flagged_fault"]
    assert result.checks["center_still_warning"]
    assert result.checks["right_still_detects"]
    assert result.checks["overall_not_sensor_fault"]
    assert result.false_critical_count == 0
    assert result.chatter_transitions <= 4


def test_total_range_blackout_and_recovery(scorecards) -> None:
    result = _result(scorecards, "total_range_blackout_and_recovery")
    assert result.checks["all_three_motors_fault_pattern"]
    assert result.checks["fault_clears_on_recovery"]
    assert result.reaction_latency_ms is not None and result.reaction_latency_ms <= 300.0
    assert result.chatter_transitions == 0


def test_fast_approaching_ttc(scorecards) -> None:
    result = _result(scorecards, "fast_approaching_ttc")
    assert result.checks["warning_before_1.0m"]
    assert result.checks["ttc_below_0.85s_at_escalation"]
    assert result.checks["closing_velocity_measured"]
    assert result.reaction_latency_ms is not None and result.reaction_latency_ms <= 250.0
    assert result.false_critical_count == 0


def test_stationary_anti_fatigue(scorecards) -> None:
    result = _result(scorecards, "stationary_anti_fatigue")
    assert result.checks["stationary_detected"]
    assert result.checks["intensity_softened"]
    assert result.checks["critical_motor_never_softened"]
    assert result.reaction_latency_ms is not None and result.reaction_latency_ms <= 150.0
    assert result.false_critical_count == 0
