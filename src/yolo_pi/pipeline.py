"""Bounded latest-frame transport used to avoid stale inference queues."""

from dataclasses import dataclass
from threading import Condition
from typing import Any, Optional


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    captured_ns: int
    payload: Any
    source: Optional[str] = None


class LatestFrameBuffer:
    """A thread-safe one-slot buffer that overwrites stale unconsumed frames."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._latest: Optional[FramePacket] = None
        self._closed = False
        self._published = 0
        self._overwritten = 0
        self._last_consumed_frame_id: Optional[int] = None

    @property
    def published_count(self) -> int:
        with self._condition:
            return self._published

    @property
    def overwritten_count(self) -> int:
        with self._condition:
            return self._overwritten

    def publish(self, packet: FramePacket) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot publish to a closed frame buffer")
            if (
                self._latest is not None
                and self._latest.frame_id != self._last_consumed_frame_id
            ):
                self._overwritten += 1
            self._latest = packet
            self._published += 1
            self._condition.notify_all()

    def wait_for_next(
        self, after_frame_id: Optional[int] = None, timeout_s: Optional[float] = None
    ) -> Optional[FramePacket]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed
                or (
                    self._latest is not None
                    and (after_frame_id is None or self._latest.frame_id > after_frame_id)
                ),
                timeout=timeout_s,
            )
            if self._latest is None:
                return None
            if after_frame_id is not None and self._latest.frame_id <= after_frame_id:
                return None
            self._last_consumed_frame_id = self._latest.frame_id
            return self._latest

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
