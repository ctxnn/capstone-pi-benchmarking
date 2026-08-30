"""JSONL sensor-replay loader for deterministic policy validation."""

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from .fusion import (
    FusionEngine,
    HazardDecision,
    SensorKind,
    SensorReading,
)
from .types import Sector


def reading_from_dict(payload: Dict[str, object]) -> SensorReading:
    required = {"sensor_id", "kind", "timestamp_ns", "sector"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Sensor reading missing fields: {', '.join(missing)}")
    distance = payload.get("distance_m")
    return SensorReading(
        sensor_id=str(payload["sensor_id"]),
        kind=SensorKind(str(payload["kind"])),
        timestamp_ns=int(payload["timestamp_ns"]),
        sector=Sector(str(payload["sector"])),
        distance_m=None if distance is None else float(distance),
        valid=bool(payload.get("valid", True)),
    )


def load_sensor_jsonl(path: Path) -> Iterator[SensorReading]:
    previous_timestamp = -1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
                reading = reading_from_dict(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sensor JSONL at line {line_number}: {exc}") from exc
            if reading.timestamp_ns < previous_timestamp:
                raise ValueError(
                    f"Sensor JSONL timestamps are not monotonic at line {line_number}"
                )
            previous_timestamp = reading.timestamp_ns
            yield reading


def replay_readings(
    readings: Iterable[SensorReading], engine: FusionEngine
) -> List[HazardDecision]:
    decisions: List[HazardDecision] = []
    for reading in readings:
        engine.update_sensor(reading)
        decisions.append(engine.evaluate(reading.timestamp_ns))
    return decisions


def write_decisions_jsonl(decisions: Iterable[HazardDecision], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision.as_dict(), sort_keys=True) + "\n")
