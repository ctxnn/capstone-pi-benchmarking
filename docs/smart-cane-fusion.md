# Smart Assistive Cane: Live Sensor Fusion & Hardware Runtime

This document covers the architecture, expected runtime behavior, tuning knobs, and step-by-step Raspberry Pi 5 verification plan for the live Smart Assistive Cane pipeline.

## Hardware Map (`configs/smart-cane-pi5.json`)

| Subsystem | Device | Interface / Pins | Sector / Role |
| --- | --- | --- | --- |
| I2C Multiplexer | TCA9548A (`0x70`) | I2C Bus 1 (`GPIO2` SDA / `GPIO3` SCL) | Thread-safe channel router (`RLock`) |
| ToF Left | VL53L0X (`0x29`) | Mux `CH0` | `left` (30 mm – 2.0 m, 33 ms budget) |
| ToF Center | VL53L0X (`0x29`) | Mux `CH1` | `center` (30 mm – 2.0 m, 33 ms budget) |
| ToF Right | VL53L0X (`0x29`) | Mux `CH2` | `right` (30 mm – 2.0 m, 33 ms budget) |
| Ultrasonic #1 | HC-SR04 | `TRIG=GPIO23`, `ECHO=GPIO24` (3.3 V level-shifted) | `center` (alternating TDM) |
| Ultrasonic #2 | HC-SR04 | `TRIG=GPIO27`, `ECHO=GPIO22` (3.3 V level-shifted) | `center` (alternating TDM, ≥35 ms guard) |
| 6-Axis IMU | MPU-6050 (`0x68`) | Mux `CH7` (or `null` for main bus), optional `INT=GPIO17` | Attitude, sweep yaw, ground-echo & stationary detection |
| Haptic Left | DRV2605L (`0x5A`) + 3 V ERM | Mux `CH4` | `left` directional vibration |
| Haptic Center | DRV2605L (`0x5A`) + 3 V ERM | Mux `CH5` | `center` directional vibration |
| Haptic Right | DRV2605L (`0x5A`) + 3 V ERM | Mux `CH6` | `right` directional vibration |
| Camera | Arducam IMX219 | MIPI CSI-2 | 640×480 RGB888 @ 30 FPS (`ov-fp32-416` OpenVINO) |

## Five-Stage Fusion Algorithm (`src/yolo_pi/fusion_v2.py` + `src/yolo_pi/imu.py`)

1. **Per-sensor conditioning & liveness:**
   - Separates healthy `no_target` returns (open space beyond 2.0 m on VL53L0X or >25 ms timeout on HC-SR04) from true hardware faults (`valid=False` or stale >250 ms).
   - 3-sample median filter with an adaptive noise-immune 2-sample fast-attack bypass (`max(0.08 m, 4% × median)`) rejects single-ping acoustic spikes while keeping step-closer reaction time to 2 samples.
2. **IMU compensation (`IMUProcessor`):**
   - Startup gyro-bias calibration (~0.5 s) + adaptive 6-axis complementary filter (drops accelerometer trust during footstep impacts).
   - Projects slant ranges onto the walking plane (`cos(Δpitch)`), suppresses downward-tilt floor echoes (respecting a hard 0.35 m safety floor), gates TTC during rapid downward pitch, shifts stale YOLO boxes by `yaw_rate × vision_age` during cane sweeps, softens non-critical vibration when completely stationary, and latches cane drops.
3. **Co-aligned Center inverse-variance + consistency fusion:**
   - Combines `tof-center`, `us-center-1`, and `us-center-2` using freshness-inflated Gauss-Markov BLUE inverse-variance weighting when readings agree.
   - When sensors disagree (narrow pole only in ToF, or glass door only in HC-SR04), drops farther dissenters, rejects single-ping transient ghosts, and trusts closer obstacles once confirmed by 2 distinct sensor timestamps.
   - Tick-rate-independent 1D α-β tracker computes closing velocity and Time-to-Collision (TTC) only on fresh sensor timestamps.
4. **Distance-scheduled range + YOLO vision fusion:**
   - **Short range (`≤ 1.0 m`):** Physical ToF + HC-SR04 unconditionally own `CRITICAL` (`≤ 0.50 m`) and `WARNING` (`≤ 1.00 m`). Vision attaches semantic labels and can never suppress, delay, or downgrade a range hazard.
   - **Transition (`1.0–2.0 m`):** Smooth cosine handoff from range to YOLO with a cross-modal agreement boost when both detect the same sector.
   - **Longer range (`> 2.0 m` up to `3.8 m`) & range-blind hazards (`stairs`, `curb`, `vehicle`, `bicycle_motorcycle`, `overhead_obstacle`):** Monocular distance estimation + 15-class Cane V1 severity weights + 2-frame persistence filtering produce early directional `ADVISORY` (or `WARNING` for close high-severity hazards; never vision-only `CRITICAL`).
5. **Per-sector asymmetric hysteresis & directional 3-motor haptics:**
   - Instant escalation on range danger; deadband + dwell on de-escalation (`0.58 m / 150 ms` for CRITICAL, `1.12 m / 200 ms` for WARNING) eliminates boundary chatter.
   - Left (`CH4`), Center (`CH5`), and Right (`CH6`) DRV2605L motors actuate independently.

## What Happens at Runtime

1. **Startup (~1–2 s):** Initializes the `TCA9548A` mux, all three `VL53L0X` sensors, `MPU-6050`, both `HC-SR04` pins, all three `DRV2605L` motor drivers (all set to `0%`), and—unless `--no-camera` is passed—loads the direct zero-torch OpenVINO `ov-fp32-416` model and starts the IMX219 camera. Hold the cane steady for ~0.5 s while startup gyro-bias calibration finishes.
2. **4-thread steady state:**
   - **Control thread (50 Hz):** Polls 3× ToF + IMU, runs fusion (~0.2–0.5 ms), and drives the 3 DRV2605L motors independently.
   - **Ultrasonic TDM thread (~14 Hz):** Alternates `us-center-1` and `us-center-2` with a ≥35 ms quiet guard so echo waits never stall the 50 Hz loop or cause acoustic crosstalk.
   - **Camera thread (30 FPS) + Vision thread (~15 FPS):** Keeps only the newest frame (drops stale frames >200 ms old before inference) and pushes YOLO detections asynchronously to fusion.
3. **Directional haptic feel per motor (Left / Center / Right independently):**

| Level | Trigger | Motor Intensity | Pulse Period | Feel |
| --- | --- | --- | --- | --- |
| `CLEAR` | No hazard in sector | `0%` | Off | Silent |
| `ADVISORY` | Longer-range YOLO (`~1.2–3.8 m`) or `1.0–1.6 m` range | `28–48%` | `520 ms` (25% duty) | Soft slow directional tap |
| `WARNING` | `0.50–1.00 m` range, TTC `< 0.85 s`, or range-blind `stairs`/`curb`/`vehicle` | `60–88%` (scales with proximity) | `250 → 160 ms` (45% duty) | Firm rhythmic pulse |
| `CRITICAL` | `≤ 0.50 m` range or TTC `< 0.40 s` | `100%` | `100 ms` (70% duty) | Urgent rapid buzz |
| `SENSOR_FAULT` | All range sensors stale/dead (>250 ms) | `35%` (all 3 motors) | `1000 ms` (20% duty) | Slow fault heartbeat |

4. **Console output & safe shutdown:** Prints a 1 Hz status line (`L/C/R` distances and levels, per-motor `%`, top YOLO detection, vision age, and `sensor→haptic` p95 latency). Pressing `Ctrl-C` (or any unhandled error / 500 ms watchdog stall) immediately zeroes all 3 DRV2605L motors before closing I2C/GPIO and prints a full latency summary.

## Step-by-Step Raspberry Pi 5 Testing Protocol

### 0. Pull, install lean zero-torch stack, and verify I2C/camera

```bash
cd ~/capstone-pi-benchmarking
git fetch origin && git checkout capy/implement-live-multi-sensor
# (or `git checkout main && git pull` after PR #2 merges)
uv pip install -e ".[pi-hardware]"
sudo raspi-config nonint do_i2c 0
i2cdetect -y 1
```

Expected: `0x70` (TCA9548A) appears in the `i2cdetect -y 1` grid. (If your `MPU-6050` is wired to the main bus instead of `CH7`, `0x68` will also appear here—set `"mux_channel": null` under `"imu"` in `configs/smart-cane-pi5.json`.)

Before running on the weak power supply (4.80 V), keep the Pi safe from brownouts:

```bash
sudo systemctl stop avahi-daemon avahi-daemon.socket
echo 1500000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq > /dev/null
```

### 1. Software self-check (14-scenario physics simulator)

```bash
.venv/bin/python scripts/simulate_fusion.py
```

Expected: `14/14 scenarios passed` with `p95 < 1 ms` per tick.

### 2. Per-channel I2C & mux probe (verifies every wired address before running motors)

```bash
.venv/bin/python -c "
from yolo_pi.hardware import SMBus2Adapter, TCA9548AMux
bus = SMBus2Adapter(1)
mux = TCA9548AMux(bus, 0x70)
for ch in range(8):
    found = [hex(a) for a in mux.scan_channel(ch)]
    if found:
        print(f'CH{ch}: {found}')
mux.close(); bus.close()
"
```

Expected:
- `CH0`, `CH1`, `CH2`: `['0x29']` (3× VL53L0X)
- `CH4`, `CH5`, `CH6`: `['0x5a']` (3× DRV2605L)
- `CH7`: `['0x68']` (MPU-6050, unless wired to main bus)

### 3. Bench test: sensors + IMU + directional haptics (`--no-camera`)

Hold the cane still for 1 second at startup, then test each behavior in order:

```bash
.venv/bin/python scripts/run_smart_cane.py --mode hardware --no-camera --print-interval 0.5 --log-jsonl /tmp/bench-nocam.jsonl
```

Verify each case:
1. **Directional check:** Put your hand ~35 cm in front of `tof-left` → only the **Left** motor buzzes at `100%`; move hand to `tof-center` → only **Center** buzzes; move to `tof-right` → only **Right** buzzes.
2. **Simultaneous multi-sector:** Put a box at ~35 cm on Left and another at ~80 cm on Right → Left pulses `CRITICAL (100%)` while Right pulses `WARNING (~70–80%)` simultaneously and Center stays `0%`.
3. **Boundary stability:** Hold a flat board right at `50 cm` and `100 cm` in Center for 5 seconds → vibration level stays steady without rapid stuttering.
4. **Glass / ultrasonic check:** Hold a clear glass or acrylic sheet ~60 cm in front of Center → even if `tof-center` misses, the two HC-SR04 sensors keep Center in `WARNING`.
5. **Narrow pole check:** Hold a broomstick/pole ~65 cm directly in front of `tof-center` with open wall at 1.6 m behind → Center locks onto ~0.65 m `WARNING`.
6. **Downward tilt check:** Tilt the cane steeply downward (>30°) toward an open floor → floor echo is suppressed; put a shoe <30 cm from the tip → immediate `CRITICAL`.
7. **Stationary softening:** Leave the cane resting still facing a wall at ~80 cm for >2 s → Center `WARNING` softens noticeably; move hand to 40 cm → instant `100% CRITICAL`.

### 4. Full live walk test (with IMX219 + zero-torch OpenVINO 416)

```bash
.venv/bin/python scripts/run_smart_cane.py --mode hardware --threads 2 --haptic-mode rtp --print-interval 0.5 --log-jsonl /tmp/walk-live.jsonl
```

Verify:
1. **Long-range YOLO handoff:** Walk toward a person, chair, or wall from 3.0 m → 0.3 m. Around 2.0–3.0 m you should feel a gentle `ADVISORY` tap in the matching direction with the YOLO label printed; below 1.0 m range sensors seamlessly take over (`WARNING` → `CRITICAL`).
2. **Range-blind hazard:** Point the cane toward stairs or a curb 1.5–2.5 m ahead → YOLO triggers Center `ADVISORY`/`WARNING` even when ToF/US read clear.
3. **Latency summary on `Ctrl-C`:** Confirm `sensor->haptic` stays in single-digit milliseconds (p95 well under 15 ms), `vision` runs asynchronously without stalling the 50 Hz loop, and all 3 motors stop immediately on exit.
4. **Optional waveform-library feel:** Compare `--haptic-mode rtp` (smooth continuous intensity) against `--haptic-mode library` (crisp built-in DRV2605L ERM click/buzz waveforms) and keep whichever feels clearer on your handle.

## Tuning Weights & Thresholds

- **Hardware pins, I2C channels, 3 V ERM voltage registers, and timing budgets:** edit `configs/smart-cane-pi5.json`.
- **Fusion thresholds, hysteresis, distance weights, TTC, and 15-class YOLO severity priors:** adjust `SmartCaneFusionConfig` and `DEFAULT_CLASS_PRIORS` in `src/yolo_pi/fusion_v2.py`, then re-run `.venv/bin/python scripts/simulate_fusion.py` to confirm all 14 safety scenarios stay green before deploying to the Pi.
