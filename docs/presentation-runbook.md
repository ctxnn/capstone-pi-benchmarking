# Presentation runbook

Every command the demo needs, in the order it runs. Commands are from
`~/capstone-pi-benchmarking` with the service stopped.

## Before the room (5 minutes)

```bash
sudo systemctl stop yolo-pi-cane          # the service and a manual run cannot share the bus
i2cdetect -y 1                            # 70 must appear, or nothing behind the mux works
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane buzz --pattern left   # you must feel one tap
PYTHONPATH=src .venv/bin/python scripts/cane_preview.py --duration 15  # warms the model, proves the camera
```

Two terminals side by side: one for commands, one showing the live log.

```bash
tail -f logs/cane-events.jsonl
```

## 1. The vibration language (40 s)

One DRV2605L drives all three ERM motors, so direction is a rhythm, not a motor.
Say the meaning, then play it.

```bash
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane buzz --pattern left --urgency 0
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane buzz --pattern center --urgency 1
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane buzz --pattern right
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane buzz --pattern overhead
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane buzz --pattern fault
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane buzz --pattern jolt
```

| Pattern | Meaning |
| --- | --- |
| 1 tap | hazard on the left |
| 2 taps | hazard in the centre |
| 3 taps | hazard on the right |
| both groups | hazards on both sides |
| two long pulses | overhead obstacle (`us-2`) |
| one long weak pulse | sensor fault |
| one solid burst | sudden tilt (IMU) |

Tempo is distance: 90 ms taps with a 600 ms pause at 1.0 m, 60 ms taps with a
150 ms pause at 0.15 m.

## 2. Bench bring-up (40 s)

```bash
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane selftest
```

Every sensor prints PASS with a live reading, then every pattern plays once.

## 3. Range and haptics (90 s)

```bash
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane run --no-vision --duration 120
```

Hand at ~0.8 m in front of left / centre / right gives 1 / 2 / 3 taps, faster as
you close in. Hand above `us-2` gives two long pulses. Tilt quickly for the jolt
burst. Point at the 5 s status line: per-sensor health, loop overruns, throttled
flags.

## 4. Vision and fusion (120 s)

```bash
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane run --duration 150
```

Walk a person in from 4 m to 0.5 m: gentle slow taps from far vision, firm under
1 m, urgent at 0.5 m, tap count following the side. The log lines name the source
(`vision` or `range`), the sector and the distance.

## 5. The live vision overlay (60 s)

Works with the sensor harness completely unplugged, so it is also the fallback.
Save the code below as `scripts/cane_preview.py` on the Pi (or run the copy that
ships with the repository), then run it from the desktop session:

```bash
sudo apt install -y python3-opencv        # once; the venv sees it via system site-packages
PYTHONPATH=src .venv/bin/python scripts/cane_preview.py
```

The window shows boxes with class, confidence and the cane's monocular distance,
the LEFT / CENTRE / RIGHT bands, and the decision the priority engine would
publish for that frame. Q closes it. `--no-display` prints decisions instead.
From the remote shell, point it at the desktop you are sharing:

```bash
WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR=/run/user/$(id -u) PYTHONPATH=src .venv/bin/python scripts/cane_preview.py
# X11 instead: DISPLAY=:0 PYTHONPATH=src .venv/bin/python scripts/cane_preview.py
```

### `scripts/cane_preview.py`

```python
#!/usr/bin/env python3
"""Live preview of what the cane's vision stage sees.

Draws the detector's boxes with class, confidence and the cane's own monocular
distance estimate, the three sector bands, and the decision the priority engine
would publish for that frame. It needs only the IMX219 and the model, so it works
with the sensor harness unplugged and while the mux is down.

Run it on the Pi desktop (Raspberry Pi Connect screen sharing shows the window):

    PYTHONPATH=src .venv/bin/python scripts/cane_preview.py

Q or Esc closes the window. ``--no-display`` prints decisions instead, which is
useful over a shell.

Drawing needs OpenCV's HighGUI. If ``import cv2`` fails:

    sudo apt install -y python3-opencv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

SECTOR_COLORS = {
    "left": (255, 190, 0),
    "center": (0, 210, 90),
    "right": (60, 80, 255),
    "unknown": (200, 200, 200),
}
LEVEL_COLORS = {
    "clear": (90, 90, 90),
    "advisory": (0, 190, 255),
    "warning": (0, 140, 255),
    "critical": (40, 40, 255),
    "sensor_fault": (120, 120, 120),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cane-hardware.json"))
    parser.add_argument("--model", type=str, default=None, help="override the configured OpenVINO model")
    parser.add_argument("--confidence", type=float, default=None, help="override the configured confidence")
    parser.add_argument("--threads", type=int, default=None, help="override the configured thread count")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = until Q)")
    parser.add_argument("--no-display", action="store_true", help="print decisions instead of opening a window")
    parser.add_argument("--window-scale", type=float, default=1.5, help="window size multiplier")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from dataclasses import replace

    from yolo_pi.cane.config import load_config
    from yolo_pi.cane.priority import PriorityFusion, estimate_distance_m
    from yolo_pi.cane.vision import create_camera, create_detector

    config = load_config(args.config)
    vision = config.vision
    if args.model:
        vision = replace(vision, model=args.model)
    if args.confidence is not None:
        vision = replace(vision, confidence=args.confidence)
    if args.threads is not None:
        vision = replace(vision, threads=args.threads)

    camera = create_camera(vision)
    detector = create_detector(vision)
    fusion = PriorityFusion(config.fusion, config.priority)

    cv2 = None
    window = "Cane vision - Q or Esc to quit"
    if not args.no_display:
        try:
            import cv2  # noqa: F811 - optional display dependency
        except ImportError:
            print("OpenCV is missing; install it with: sudo apt install -y python3-opencv", flush=True)
            return 2

    print(f"model {vision.model}  input {vision.imgsz or 'from metadata'}  threads {vision.threads}", flush=True)
    started = time.monotonic()
    frames = 0
    last_print = 0.0
    try:
        camera.start()
        while True:
            packet = camera.next_frame(timeout_s=10.0)
            result = detector.detect(
                packet.payload,
                packet.frame_id,
                captured_ns=packet.captured_ns,
                source=packet.source,
            )
            fusion.update_vision(result)
            decision = fusion.evaluate(time.perf_counter_ns())
            frames += 1

            if args.no_display:
                now = time.monotonic()
                if now - last_print >= 1.0:
                    last_print = now
                    labels = [det.class_name for det in result.detections]
                    print(
                        f"{decision.level.name.lower():7s} source={decision.source:8s} "
                        f"sectors={[s.value for s in decision.sectors]} "
                        f"nearest={decision.nearest_distance_m if decision.nearest_distance_m is None else round(decision.nearest_distance_m, 2)} "
                        f"detections={labels} frame={frames}",
                        flush=True,
                    )
            else:
                image = packet.payload.copy()
                if image.ndim == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                height, width = image.shape[:2]
                band = width * config.vision.center_fraction
                left_edge = int((width - band) / 2.0)
                right_edge = int((width + band) / 2.0)
                for x, text in ((left_edge, "LEFT"), (right_edge, "RIGHT")):
                    cv2.line(image, (x, 0), (x, height), (70, 70, 70), 1)
                    cv2.putText(
                        image, text, (x + 6, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (140, 140, 140), 1, cv2.LINE_AA,
                    )
                for det in result.detections:
                    distance = estimate_distance_m(det, width, height, config.priority)
                    color = SECTOR_COLORS.get(det.sector.value, SECTOR_COLORS["unknown"])
                    box = det.box
                    cv2.rectangle(
                        image,
                        (int(box.x1), int(box.y1)),
                        (int(box.x2), int(box.y2)),
                        color,
                        2,
                    )
                    label = f"{det.class_name} {det.confidence:.2f}"
                    if distance is not None:
                        label += f"  {distance:.1f} m"
                    label += f"  {det.sector.value}"
                    cv2.putText(
                        image, label, (int(box.x1), max(18, int(box.y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
                    )
                banner = (
                    f"{decision.level.name}  {decision.source}  "
                    f"nearest {decision.nearest_distance_m if decision.nearest_distance_m is None else round(decision.nearest_distance_m, 2)} m  "
                    f"frame {frames}  {result.model_call_ms:.0f} ms"
                    if hasattr(result, "model_call_ms")
                    else f"{decision.level.name}  {decision.source}  frame {frames}"
                )
                cv2.rectangle(image, (0, 0), (width, 34), (0, 0, 0), -1)
                cv2.putText(
                    image, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    LEVEL_COLORS.get(decision.level.name.lower(), (255, 255, 255)), 2, cv2.LINE_AA,
                )
                if args.window_scale != 1.0:
                    image = cv2.resize(
                        image, None,
                        fx=args.window_scale, fy=args.window_scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                cv2.imshow(window, image)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            if args.duration and time.monotonic() - started >= args.duration:
                break
    finally:
        camera.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    print(f"frames {frames} in {time.monotonic() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 6. Fault tolerance (60 s)

Stage it before the talk: power off, unplug `tof-left` from the mux, boot.

```bash
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane run --no-vision --duration 120
```

The cane keeps warning on centre and right while the left sector reports a fault
in the status line, instead of stopping. Then power off, plug it back in, boot,
run again: within about 2 s the log says `ToF sensors recovered: tof-left`.

No rewiring? Show the injector instead:

```bash
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane soak --seconds 60 --seed 1
```

Fuzz, watchdog, shutdown, poller-error and soak sections all PASSED with
thousands of injected faults, zero loop overruns and no leaked threads.

## 7. Autoboot (30 s)

```bash
sudo reboot
```

No login, no shell: wave a hand and the cane buzzes. Then:

```bash
journalctl -u yolo-pi-cane -f
```

## After the demo

```bash
sudo systemctl start yolo-pi-cane
```

## If the mux is down

`i2cdetect -y 1` empty means the TCA9548A is not answering, and the cane cannot
run at all: all three ToF chips share address `0x29`, so the mux is mandatory, and
the haptic driver is required by design (no haptics, no cane). Demo the parts
that do not need it and say so honestly:

```bash
PYTHONPATH=src .venv/bin/python scripts/cane_preview.py        # vision on the shared screen
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane soak --seconds 60 --seed 1
.venv/bin/pytest tests/test_fault_tolerance.py -q              # 25 fault-injection tests
PYTHONPATH=src .venv/bin/python scripts/simulate_fusion.py     # 14/14 scenarios
```

## Numbers worth quoting

- 493 tests passing, 14/14 fusion scenarios, soak clean under thousands of injected faults.
- 50 Hz control loop, worst loop in the soak was a few milliseconds, zero overruns.
- Faults survived: dead ToF chip, wedged mux channel, failed ultrasonic ping, IMU
  dropout, detector error, lost haptic write, and a crashed fusion pass.
- Blind-sector vision: WARNING to 2.0 m, CRITICAL at 0.5 m only after three
  consecutive close, confident frames.

## Quick fixes

| Symptom | Fix |
| --- | --- |
| `haptic driver unavailable: DRV2605L (mux channel 3)` | Cold boot; check `0x5a` on channel 3 with `scan`; reseat channel-3 jumpers |
| `Errno 121` on every command | The mux is not answering: `i2cdetect -y 1` must show `70`; check its 3.3 V, GND, SDA pin 3, SCL pin 5, and tie RESET high |
| `scan` hangs | The bus is wedged; power-cycle and reseat before retrying |
| `ultrasonic us-1: no echo` | Hold a hand 20-50 cm in front; then check TRIG/ECHO and the 5 V level shifter |
| `IMU read failed` | Keep `"imu": {"enabled": false}` in `configs/cane-hardware.json`; the cane runs without it |
