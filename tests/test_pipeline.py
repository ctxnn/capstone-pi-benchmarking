import unittest

from yolo_pi.pipeline import FramePacket, LatestFrameBuffer


class LatestFrameBufferTests(unittest.TestCase):
    def test_overwrites_unconsumed_frame_and_returns_latest(self) -> None:
        buffer = LatestFrameBuffer()
        buffer.publish(FramePacket(1, 10, "old"))
        buffer.publish(FramePacket(2, 20, "new"))
        packet = buffer.wait_for_next(timeout_s=0)
        self.assertIsNotNone(packet)
        self.assertEqual(packet.payload, "new")
        self.assertEqual(buffer.overwritten_count, 1)

    def test_wait_after_consumed_frame_times_out(self) -> None:
        buffer = LatestFrameBuffer()
        buffer.publish(FramePacket(1, 10, "only"))
        first = buffer.wait_for_next(timeout_s=0)
        self.assertEqual(first.frame_id, 1)
        self.assertIsNone(buffer.wait_for_next(after_frame_id=1, timeout_s=0))

    def test_rejects_publish_after_close(self) -> None:
        buffer = LatestFrameBuffer()
        buffer.close()
        with self.assertRaises(RuntimeError):
            buffer.publish(FramePacket(1, 10, "late"))


if __name__ == "__main__":
    unittest.main()
