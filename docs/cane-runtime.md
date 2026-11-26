# Running the whole cane: sensors, vision and haptics

The rest of this repository decides *what to say* (fusion policy, detector). The
`yolo_pi.cane` package is the part that *reads the real sensors and drives the real
motor*, and it runs everything as one program. It closes the "hardware-layer gap"
named in the capstone report.

```
 VL53L0X x3 (left/centre/right) ─┐
 HC-SR04 us-1 (tilted, forward) ─┤  CLOSE  (0 to 1.0 m)       ┌─► pattern player ─► ONE DRV2605L ─► 3 ERM motors
                                 ├─► PriorityFusion ─► smoother┤    (taps = direction,              (wired in parallel)
 HC-SR04 us-2 (straight, up) ────┤  OVERHEAD (two long pulses) │     speed = distance)
 IMX219 ─► YOLO ─────────────────┘  FAR (1.0 to 4.0 m)        │
                                                              │
 MPU-6050 ─► JoltDetector ── sudden tilt only ────────────────┘ (one 300 ms burst)
```

`fusion.py`, `detector.py`, `picamera2_source.py` and the rest of the existing
package are **unchanged**. The new code only feeds them real data and carries out
their decisions.

## Quick start on the Pi

```bash
cd ~/capstone-pi-benchmarking
git pull                                   # or copy this branch across
bash deploy/install-cane-on-pi.sh          # apt packages, I2C at 400 kHz, Python drivers
sudo reboot                                # first time only

PYTHONPATH=src .venv/bin/python -m yolo_pi.cane scan        # 1. what answers on each mux channel
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane selftest    # 2. read every sensor, play every pattern
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane run --no-vision   # 3. sensors + motors only
PYTHONPATH=src .venv/bin/python -m yolo_pi.cane run         # 4. everything
```

Bring it up in that order. Each step isolates one layer, so a failure points at one
thing. When step 4 behaves, install the boot service from
`deploy/yolo-pi-cane.service`.

On a laptop, with no hardware, the same software runs against a scripted walk:

```bash
PYTHONPATH=src python3 -m yolo_pi.cane run --simulate --duration 21
PYTHONPATH=src python3 -m yolo_pi.cane selftest --simulate
```

## Wiring this expects (`configs/cane-hardware.json`)

Taken from the report's tables. **Mux channel numbers are not fixed by the report**, so
the values below are planned defaults: confirm them with `scan` and edit the JSON
if the harness differs. The file rejects unknown keys, duplicate channels and
reserved pins at start-up.

| Part | Address / pins | Mux channel | Sector |
| --- | --- | --- | --- |
| TCA9548A | I2C `0x70`, GPIO2/GPIO3 | none (on the main bus) | n/a |
| VL53L0X left / centre / right | `0x29` | 0 / 1 / 2 | left / centre / right |
| DRV2605L (one, drives all three motors) | `0x5A` | main bus (or set `haptics.driver_mux_channel`) | n/a |
| MPU-6050 | `0x68` | main bus (or set `imu.mux_channel`) | n/a |
| HC-SR04 `us-1` (tilted, forward) | TRIG GPIO23, ECHO GPIO24 | n/a | centre, ground-level |
| HC-SR04 `us-2` (straight, overhead) | TRIG GPIO27, ECHO GPIO22 | n/a | centre, above the path |

Both ultrasonic sensors map to the centre sector. The HC-SR04 ECHO pins are 5 V signals and
must go through the level shifter before they reach the Pi.

## Who decides, by distance

| Zone | Decides | Direction | Feel |
| --- | --- | --- | --- |
| 0 to 1.0 m (`fusion.warning_distance_m`) | ToF x3 + forward ultrasonic (`us-1`) | the sector of the sensor that sees it | firm taps, faster and stronger the nearer it is |
| 1.0 m to 4.0 m (`priority.far_max_m`) | YOLO | the side of the detected object (nearest object only) | slow, gentle taps (`priority.far_intensity` 0.45) |
| Above the path (`us-2`) | overhead ultrasonic, 1.0 m / 0.5 m | n/a | two long pulses |
| Sudden tilt | IMU `JoltDetector` | n/a | one solid 300 ms burst at 0.8 |

## The vibration language (one driver, three motors)

One DRV2605L drives all three motors, so they always buzz together and
cannot be told apart by location. Direction and distance are therefore coded
in the *rhythm*, which the hand feels as a pattern:

| Meaning | Pattern |
| --- | --- |
| Object on the **left** | 1 tap, pause, repeat |
| Object **centre** | 2 taps, pause, repeat |
| Object on the **right** | 3 taps, pause, repeat |
| Left and right together | both groups in turn, with a longer pause between them |
| **Overhead** obstacle | 2 long pulses (250 ms each), longer pause |
| **Sensor fault** | 1 very long weak pulse (600 ms) |
| **Sudden tilt** | one solid 300 ms burst |
| **Distance** | the nearer, the faster the taps, the shorter the pause and the stronger the buzz. At 1.0 m the taps are slow (a left group every 0.7 s); at 0.15 m or less they are rapid (a group every 0.2 s). Tempo changes smoothly and never changes the tap count. |

Typical cycle lengths with the shipped timings: left at 0.95 m about 0.7 s, left at
0.1 m 0.21 s, centre 0.61 s to 0.33 s, right (very near) 0.45 s. The gap
between groups is always at least 1.8 times the gap between taps, so
"2 taps" cannot be mistaken for "1 tap + 1 tap".

Try them on the bench: `python -m yolo_pi.cane buzz --pattern right --urgency 1.0`.

Rules that follow from this:

* The handoff is the warning distance, so there is no gap and no overlap. In the
  near band the camera cannot raise or cancel an alert; it only adds a label.
* An active near alert (range or overhead) silences far camera alerts, so the
  hand is never given two competing patterns.
* If every forward range sensor is dead, the camera also covers the near band
  (with a normal near pattern) instead of leaving the user with only the
  fault pulse. With no camera detection either, the fault pulse plays.
* The camera must see the same sector in `priority.confirm_frames` (2)
  consecutive frames at confidence of at least 0.5 before it pulses.
* **The camera's distance is an estimate**: box height against the class's
  typical real height (`priority.class_heights_m`) and the lens's vertical field of view
  (`priority.vertical_fov_deg`, 48.8 degrees for the IMX219 full frame). Tune the
  heights and field of view from a measured walk-up test (stand a person at 2, 3
  and 4 m and compare the log's `nearest_distance_m`). A box clipped by the frame
  edge is treated as inside the alert band.

### The two ultrasonic sensors

`us-1` (`GPIO23/24`) is tilted to look forward along the walking path (centre sector).
`us-2` (`GPIO27/22`) is angled slightly toward the ground side (`"role": "ground"`, `"pitch_offset_deg": 8.0`) so it catches low obstacles, steps, and curbs along the centre corridor while receiving extra downward pitch compensation in the IMU ground-echo gate. (An optional `"role": "overhead"` remains supported if a sensor is ever re-aimed upward.)

### What the IMU does and does not do

The only thing the IMU vibrates for is a **sudden change in tilt**: the
direction of gravity moving by 25 degrees or more within 0.25 s (`imu.jolt_angle_deg`,
`imu.jolt_window_s`), then 1 s of quiet (`imu.jolt_cooldown_s`). It works from the
gyro and accelerometer together and does not depend on how the chip is oriented
on the cane. Slow tilting, turning about the vertical, and accelerometer knocks
without a real rotation do not trigger it. The IMU never reports obstacles. It
also still tells the detector when the cane is resting (so the detector can idle);
that produces no vibration, and `vision.idle_when_still: false` turns it off.

## What each module does

| Module | Role |
| --- | --- |
| `config.py` | Strict, validated wiring and runtime settings |
| `bus.py` | One I2C bus with a locked mux; register access used by the MPU and DRV drivers |
| `tof.py` | VL53L0X in continuous mode, non-blocking; raw millimetres to a fusion-ready reading |
| `ultrasonic.py` | HC-SR04 timed from `lgpio` edge timestamps, one sensor fired at a time |
| `imu.py` | MPU-6050 driver, start-up gyro-bias calibration, sudden-tilt detector, "cane is resting" flag |
| `priority.py` | The near / far / overhead rules and the camera distance estimate |
| `patterns.py` | The vibration language: direction to tap count, distance to tempo and strength |
| `haptics.py` | DRV2605L ERM driver and the thread that plays the pattern |
| `chaos.py` | The soak / fuzz / fault-injection harness behind `yolo-pi-cane soak` |
| `sensors.py` | Poller threads that restart after an unexpected error; a dead sensor just goes stale |
| `vision.py` | Camera + detector thread; every failure is caught and retried |
| `runtime.py` | Fuse, smooth, vibrate loop; JSON event log; status every 5 s |
| `app.py`, `cli.py` | Assembly, signal handling, the `yolo-pi-cane` command |
| `sim.py`, `selftest.py` | Laptop simulation and the bench checks |

## Decisions worth knowing about

**Out of range is "clear", not a fault.** A VL53L0X or HC-SR04 that sees nothing is
the good case for a cane. It is reported as a valid reading at 2.0 m (ToF) or 4.0 m
(ultrasonic). If it were marked invalid, the fusion engine would report SENSOR_FAULT
whenever the path is empty. A genuinely failed read (bus error, timeout) simply
produces no new reading, and the fusion engine ignores it after 250 ms.

**Priority by distance** is described in "Who decides, by distance" above.

**Direction is the rhythm.** One driver cannot pulse one motor at a time, so a hazard's
side is the number of taps (1 / 2 / 3) and its closeness is the tempo. A sensor fault
has its own long pulse. A direction change restarts the code only during a pause,
so a hazard flickering between sectors cannot hold the motor on.

**Escalate at once, release slowly.** An object sitting right on the 0.5 m or 1.0 m
line would flip the pattern on every reading. The smoother never delays a rise, but
holds WARNING/CRITICAL for 250 ms after the last reading that justified it
(`runtime.release_hold_ms`).

**The motors cannot get stuck on.** The haptic thread silences the driver if no
command arrives for 1 s (the loop sends one every 20 ms), forces a rest if any
stretch of vibration passes 0.95 s, retries a lost "off" write, every shutdown path
parks the driver, and `systemd` restarts the program if it dies.

**The camera decides only where the range sensors cannot.** Close in, detections
only add labels to a range alert. A camera, model or OpenVINO failure is logged and
retried; the cane keeps working on range sensors.

**One failure never takes the cane down.** A dead ToF chip, a quiet mux
channel, a failing ultrasonic ping, an IMU dropout, a detector error, or a lost
haptic write is counted in the runtime summary (`source_errors`) and in the
status health report (`poller_errors`) while the cane keeps warning on whatever
is left. Faulted VL53L0X chips are re-initialized every 2 s (`reinit_attempts`,
`reinit_recovered`), sensor pollers restart themselves after an unexpected
error instead of dying, and the fuse loop falls back to its last safe decision
for 500 ms before publishing `SENSOR_FAULT`. When a sector's range sensors are
all blind, the camera covers that sector: `WARNING` up to 2.0 m, `CRITICAL` at
0.5 m only with confidence and persistence. The program only stops when it has
neither sensor pollers nor a ready camera; a fully blind cane keeps
heartbeating `SENSOR_FAULT` rather than exiting silently.

**The detector idles when the cane rests.** After 3 s without movement the detector
drops to about one frame per second (the report's battery suggestion). Range sensors
keep running at full rate. If the IMU fails it reports "moving", so the detector
stays at full rate.

## Reading the logs

`journalctl -u yolo-pi-cane -f` shows one line per hazard change and a status line
every 5 s. The same records go to `logs/cane-events.jsonl`:

```json
{"kind": "status", "level": "clear", "fresh_range_sensors": 5,
 "loop": {"iterations": 250, "max_loop_ms": 1.3, "overruns": 0},
 "sensors": {"tof-left": {"ok": 140, "errors": 0, "age_ms": 10.3, "distance_m": 2.0}, ...},
 "vision": {"frames": 49, "mean_model_ms": 61.5},
 "cpu_temp_c": 58.2, "throttled": "0x0"}
```

`throttled` is `vcgencmd get_throttled`: anything but `0x0` means under-voltage or
thermal limiting has happened (the benchmark Pi was under-volted; check this on the
final battery and buck converter).

## Tuning on the real cane

| Setting | Why it may need changing |
| --- | --- |
| `haptics.amplitude_floor` (0.30) | The weak fault pattern must be felt through the grip. Raise it if the 0.35 pulse is not noticeable. |
| `haptics.motor_voltage_v` (3.0) | Sets the DRV2605L over-drive clamp for the BOM's "about 3 V" motor. Use the motor's real rating. |
| `haptics.tap_on_*_ms`, `tap_off_*_ms`, `group_gap_*_ms` | Tap and pause lengths, slow (far) to fast (very near). ERM motors need about 40 ms to spin up, so the fast taps (60 ms) are close to the limit: lengthen them if fast taps feel mushy. |
| `haptics.taps_left/center/right` (1/2/3), `urgent_distance_m` (0.15) | The tap code and the distance at which the pattern is fastest. |
| `tof.timing_budget_us` (33000) | 20000 gives faster updates at lower accuracy. |
| `fusion.*` | Warning 1.0 m, critical 0.5 m, 250 ms staleness are the report's values. `warning_distance_m` is also where the camera takes over. |
| `priority.class_heights_m`, `vertical_fov_deg` | Set the camera's distance estimate. Calibrate with a walk-up test. |
| `priority.far_intensity` | How gentle the far camera pattern is (0.45; it also plays at the slowest tempo). |
| `priority.vision_min_confidence`, `confirm_frames` | Raise either if far pulses fire on false detections. |
| `priority.blind_sector_max_distance_m` (2.0), `blind_sector_critical_confidence` (0.60), `blind_sector_critical_frames` (3) | How much the camera is trusted in a sector whose range sensors are all dead. Raise the confidence or frames if a blind sector buzzes at shadows; raise the distance if the camera should cover a dead ToF earlier. |
| `imu.jolt_angle_deg`, `jolt_window_s` | Raise the angle or shorten the window if ordinary cane sweeping triggers the pulse; lower it if a real knock does not. |
| `vision.threads` (3) | Leaves one core for the sensor and haptic threads. The benchmark used 2 to 4. |
| `runtime.release_hold_ms` (250) | Longer is calmer, shorter is more responsive. |

## What is and is not verified

Verified here, on a laptop: 150+ cane tests pass (configuration rules, every
conversion and timing rule, the DRV2605L and MPU-6050 register sequences against
in-memory fakes, the pattern language, the pollers, error handling and restart, the
smoother, vision retry, selftest, and the whole chain in simulation including thread
start and shutdown with the driver parked). `yolo-pi-cane soak` runs the whole system for
60 s with up to 60% of bus reads and writes failing, NaN and negative readings,
slow reads, sensors dying, a frozen runtime and signals; eight runs across four seeds
all passed, with the longest uninterrupted vibration 0.6 s. The soak found three
real bugs that are now fixed (a NaN IMU sample disabling the jolt detector for good,
a lost final "off" write leaving the motor running, and a flickering direction
restarting the first tap every reading). The existing tests are unaffected.

**Not verified: any of this against the real chips.** The code was written from the
datasheets and the Adafruit library sources, but no cane was attached. Treat first
power-up as bench testing, not deployment:

1. The mux channels, the second ultrasonic pin pair and the MPU-6050 location are
   planned values (`scan` shows the truth). So are the ultrasonic roles: confirm
   which sensor is the tilted one.
2. The DRV2605L open-loop setup (OD_CLAMP = volts / 21.18 mV, signed RTP full scale
   127) follows the datasheet; confirm that the motors spin and do not over-drive at
   3 V. **Three motors share one DRV2605L output:** check the chip's output-current
   limit against their combined start-up current (a sag or a reset on start-up is the
   symptom), and wire the motors with a flyback diode each if the board does not
   already. If the taps feel blurred, lengthen `tap_on_fast_ms`.
3. `lgpio` selects the Pi 5 GPIO chip by label; `selftest` shows whether the
   ultrasonic echoes arrive.
4. An HC-SR04 with a broken ECHO wire looks like an empty path. `selftest` asks for a
   hand in front of each sensor for this reason; the `no_echo_streak` is also useful
   when debugging. The ToF sensors and the camera cover that failure in use.
5. Camera-to-vibration latency and battery life are still to be measured end to end.
