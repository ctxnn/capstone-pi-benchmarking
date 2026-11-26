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

```bash
PYTHONPATH=src .venv/bin/python scripts/cane_preview.py
```

The window shows boxes with class, confidence and the cane's monocular distance,
the LEFT / CENTRE / RIGHT bands, and the decision the priority engine would
publish for that frame. Q closes it. Needs `python3-opencv`
(`sudo apt install -y python3-opencv`); `--no-display` prints decisions instead.

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
