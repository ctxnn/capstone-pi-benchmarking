"""The vibration language: what a hazard feels like on one shared motor channel.

One DRV2605L drives all three ERM motors together, so the motors cannot be told
apart and direction has to live in the rhythm:

* a group of taps says where: LEFT = 1 tap, CENTER = 2 taps, RIGHT = 3 taps
  (``haptics.taps_*``). Two sides at once play both groups, one after the other;
* the tempo says how close: the taps and the pause between groups speed up
  smoothly from the slow values at the warning distance to the fast values at
  ``haptics.urgent_distance_m``, so something very near is a rapid buzz;
* the amplitude grows with closeness too (0.65 up to 1.0);
* a gentle, slow version of the same groups means the camera sees something
  still far away;
* OVERHEAD (something at head height) is two long pulses;
* a SENSOR FAULT is one long, weak pulse every 1.5 s;
* the IMU's sudden-tilt pulse is one solid 300 ms buzz (see ``HapticDriver.burst``).

A pattern is a repeating cycle of ``(amplitude, seconds)`` segments, so it can
be inspected and tested without any hardware.
"""

from __future__ import annotations

from typing import Tuple

from ..fusion import HapticCommand
from ..types import Sector
from .config import HapticsConfig

Segment = Tuple[float, float]  # (amplitude 0..1, seconds)
Cycle = Tuple[Segment, ...]

SOURCE_FAULT = "fault"
SOURCE_OVERHEAD = "overhead"
URGENCY_STEPS = 4  # tempo changes in quarters so the rhythm does not wobble with every reading
FAULT_PAUSE_S = 0.9
OVERHEAD_GAP_S = 0.12
_ORDER = (Sector.LEFT, Sector.CENTER, Sector.RIGHT)


def urgency_for(distance_m: float, warning_m: float, urgent_m: float) -> float:
    """0 at the warning distance, 1 at ``urgent_m`` or closer, in quarter steps."""

    if warning_m <= urgent_m:
        return 1.0
    raw = (warning_m - distance_m) / (warning_m - urgent_m)
    raw = max(0.0, min(1.0, raw))
    return round(raw * URGENCY_STEPS) / URGENCY_STEPS


def cycle_length(cycle: Cycle) -> float:
    return sum(seconds for _, seconds in cycle)


def amplitude_at(cycle: Cycle, elapsed_s: float) -> float:
    """Amplitude at ``elapsed_s`` into the (repeating) cycle."""

    length = cycle_length(cycle)
    if not cycle or length <= 0.0:
        return 0.0
    position = elapsed_s % length
    for amplitude, seconds in cycle:
        if position < seconds:
            return amplitude
        position -= seconds
    return 0.0


def _lerp(slow: float, fast: float, urgency: float) -> float:
    return slow + (fast - slow) * urgency


def build_cycle(command: HapticCommand, config: HapticsConfig) -> Cycle:
    """The repeating pattern for a command; an empty cycle means silence."""

    amplitude = command.intensity
    if amplitude <= 0.0 or not command.sectors:
        return ()
    source = getattr(command, "source", "range")
    urgency = max(0.0, min(1.0, float(getattr(command, "urgency", 0.0))))
    long_s = config.long_pulse_ms / 1000.0
    sectors = [sector for sector in _ORDER if sector in command.sectors]

    if source == SOURCE_FAULT or not sectors:
        return ((amplitude, long_s * 2.4), (0.0, FAULT_PAUSE_S))

    on_s = _lerp(config.tap_on_ms, config.tap_on_fast_ms, urgency) / 1000.0
    off_s = _lerp(config.tap_off_ms, config.tap_off_fast_ms, urgency) / 1000.0
    gap_s = _lerp(config.group_gap_ms, config.group_gap_fast_ms, urgency) / 1000.0

    if source == SOURCE_OVERHEAD:
        return ((amplitude, long_s), (0.0, OVERHEAD_GAP_S), (amplitude, long_s), (0.0, gap_s))

    cycle = []
    for index, sector in enumerate(sectors):
        for tap in range(config.taps_for(sector)):
            cycle.append((amplitude, on_s))
            if tap < config.taps_for(sector) - 1:
                cycle.append((0.0, off_s))
        cycle.append((0.0, config.sector_gap_ms / 1000.0 if index < len(sectors) - 1 else gap_s))
    return tuple(cycle)


def one_shot(cycle: Cycle) -> Cycle:
    """The cycle without its closing pause, for playing a pattern exactly once."""

    trimmed = list(cycle)
    while trimmed and trimmed[-1][0] == 0.0:
        trimmed.pop()
    return tuple(trimmed)
