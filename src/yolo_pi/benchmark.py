"""Reproducible detector benchmark runner and structured result writer."""

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .detector import Detector


@dataclass(frozen=True)
class BenchmarkInput:
    """One fixed input with a traceable source label."""

    payload: Any
    source: str


@dataclass(frozen=True)
class BenchmarkObservation:
    iteration: int
    repetition: int
    source: str
    model_call_ms: float
    frame_age_at_model_finish_ms: float
    detection_count: int
    stage_ms: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "iteration": self.iteration,
            "repetition": self.repetition,
            "source": self.source,
            "model_call_ms": self.model_call_ms,
            "frame_age_at_model_finish_ms": self.frame_age_at_model_finish_ms,
            "detection_count": self.detection_count,
            "stage_ms": dict(self.stage_ms),
        }


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    mean = statistics.fmean(values)
    return {
        "count": float(len(values)),
        "mean": mean,
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "model_calls_per_second_from_mean": 1000.0 / mean if mean > 0 else 0.0,
    }


@dataclass(frozen=True)
class BenchmarkReport:
    warmup_runs: int
    repetitions: int
    backend: str
    model_id: str
    observations: List[BenchmarkObservation]

    def as_dict(self) -> Dict[str, object]:
        model_times = [observation.model_call_ms for observation in self.observations]
        ages = [
            observation.frame_age_at_model_finish_ms
            for observation in self.observations
        ]
        stage_names = sorted(
            {
                stage_name
                for observation in self.observations
                for stage_name in observation.stage_ms
            }
        )
        stage_summaries = {
            stage_name: summarize(
                [
                    observation.stage_ms[stage_name]
                    for observation in self.observations
                    if stage_name in observation.stage_ms
                ]
            )
            for stage_name in stage_names
        }
        return {
            "schema_version": 1,
            "warmup_runs_excluded": self.warmup_runs,
            "repetitions": self.repetitions,
            "backend": self.backend,
            "model_id": self.model_id,
            "scope": "model-call laptop measurement; not Raspberry Pi evidence",
            "model_call_ms": summarize(model_times),
            "frame_age_at_model_finish_ms": summarize(ages),
            "stage_ms": stage_summaries,
            "observations": [observation.as_dict() for observation in self.observations],
        }

    def write_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def run_benchmark(
    detector: Detector,
    inputs: Sequence[BenchmarkInput],
    warmup_runs: int = 5,
    repetitions: int = 10,
) -> BenchmarkReport:
    """Benchmark fixed inputs after explicit warm-up calls.

    Warm-up results are deliberately excluded. Each measured call receives a
    fresh capture timestamp so frame age remains meaningful.
    """

    if not inputs:
        raise ValueError("At least one benchmark input is required")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must not be negative")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    frame_id = 0
    for warmup_index in range(warmup_runs):
        item = inputs[warmup_index % len(inputs)]
        detector.detect(
            item.payload,
            frame_id=frame_id,
            captured_ns=perf_counter_ns(),
            source=item.source,
        )
        frame_id += 1

    observations: List[BenchmarkObservation] = []
    backend: Optional[str] = None
    model_id: Optional[str] = None
    iteration = 0
    for repetition in range(repetitions):
        for item in inputs:
            result = detector.detect(
                item.payload,
                frame_id=frame_id,
                captured_ns=perf_counter_ns(),
                source=item.source,
            )
            frame_id += 1
            if backend is None:
                backend = result.backend
                model_id = result.model_id
            elif result.backend != backend or result.model_id != model_id:
                raise ValueError("A benchmark report cannot mix backends or model identities")
            observations.append(
                BenchmarkObservation(
                    iteration=iteration,
                    repetition=repetition,
                    source=item.source,
                    model_call_ms=result.model_call_ms,
                    frame_age_at_model_finish_ms=result.frame_age_at_model_finish_ms,
                    detection_count=len(result.detections),
                    stage_ms=dict(result.stage_ms),
                )
            )
            iteration += 1

    return BenchmarkReport(
        warmup_runs=warmup_runs,
        repetitions=repetitions,
        backend=backend or "unknown",
        model_id=model_id or "unknown",
        observations=observations,
    )
