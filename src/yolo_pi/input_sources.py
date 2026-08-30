from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Optional, Protocol, Sequence


@dataclass(frozen=True)
class BenchmarkFrame:
    frame_id: int
    payload: Any
    source: str
    source_type: str
    captured_ns: Optional[int]
    capture_started_ns: Optional[int]
    available_ns: int
    dropped_since_last: int = 0

    @property
    def capture_ms(self) -> Optional[float]:
        if self.capture_started_ns is None:
            return None
        return (self.available_ns - self.capture_started_ns) / 1_000_000.0

    @property
    def frame_age_at_available_ms(self) -> Optional[float]:
        if self.captured_ns is None:
            return None
        return (self.available_ns - self.captured_ns) / 1_000_000.0


class BenchmarkInputSource(Protocol):
    def start(self) -> None: ...

    def next_frame(self, timeout_s: float = 10.0) -> BenchmarkFrame: ...

    def close(self) -> None: ...


class ImageSequenceSource:
    """Repeat a fixed saved-image sequence without pretending it is a camera."""

    def __init__(self, inputs: Sequence[Path]) -> None:
        if not inputs:
            raise ValueError("image source requires at least one input")
        self.inputs = list(inputs)
        self._next_index = 0
        self._frame_id = 0

    def start(self) -> None:
        return None

    def next_frame(self, timeout_s: float = 10.0) -> BenchmarkFrame:
        del timeout_s
        path = self.inputs[self._next_index % len(self.inputs)]
        self._next_index += 1
        available_ns = perf_counter_ns()
        frame = BenchmarkFrame(
            frame_id=self._frame_id,
            payload=path,
            source=str(path),
            source_type="images",
            captured_ns=None,
            capture_started_ns=None,
            available_ns=available_ns,
        )
        self._frame_id += 1
        return frame

    def close(self) -> None:
        return None

    def __enter__(self) -> "ImageSequenceSource":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def create_input_source(
    source_type: str,
    *,
    inputs: Sequence[Path],
    camera_config: Optional[dict] = None,
) -> BenchmarkInputSource:
    if source_type == "images":
        return ImageSequenceSource(inputs)
    if source_type == "camera":
        from .picamera2_source import Picamera2LatestFrameSource

        return Picamera2LatestFrameSource(**(camera_config or {}))
    raise ValueError(f"unsupported input source: {source_type}")
