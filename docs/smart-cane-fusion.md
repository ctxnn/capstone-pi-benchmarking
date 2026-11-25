# Smart Assistive Cane: Live Sensor Fusion & Hardware Runtime

This document describes the live multi-sensor perception, fusion, IMU compensation, and directional haptic runtime targeting the Raspberry Pi 5 (2 GB) Smart Assistive Cane.

## Hardware Map (`configs/smart-cane-pi5.json`)

| Subsystem | Device | Interface / Pins | Sector / Role |
| --- | --- | --- | --- |
| I2C Multiplexer | TCA9548A (`0x70`) | I2C Bus 1 (`GPIO2` SDA / `GPIO3` SCL) | Thread-safe channel router (`RLock`) |
| ToF Left | VL53L0X (`0x29`) | Mux `CH0` | `left` (30 mm – 2.0 m, 33 ms budget) |
| ToF Center | VL53L0X (`0x29`) | Mux `CH1` | `center` (30 mm – 2.0 m, 33 ms budget) |
| ToF Right | VL53L0X (`0x29`) | Mux `CH2` | `right` (30 mm – 2.0 m, 33 ms budget) |
| Ultrasonic #1 | HC-SR04 | `TRIG=GPIO23`, `ECHO=GPIO24` (level-shifted) | `center` (alternating TDM) |
| Ultrasonic #2 | HC-SR04 | `TRIG=GPIO27`, `ECHO=GPIO22` (level-shifted) | `center` (alternating TDM, ≥35 ms guard) |
| 6-Axis IMU | MPU-6050 (`0x68`) | Mux `CH7` (or `null` for main bus), optional `INT=GPIO17` | Attitude, sweep yaw, ground-echo & stationary detection |
| Haptic Left | DRV2605L (`0x5A`) + 3V ERM | Mux `CH4` | `left` directional vibration |
| Haptic Center | DRV2605L (`0x5A`) + 3V ERM | Mux `CH5` | `center` directional vibration |
| Haptic Right | DRV2605L (`0x5A`) + 3V ERM | Mux `CH6` | `right` directional vibration |
| Camera | Arducam IMX219 | MIPI CSI-2 | 640×480 RGB888 @ 30 FPS (`ov-fp32-416` OpenVINO) |

## Five-Stage Fusion Algorithm (`src/yolo_pi/fusion_v2.py` + `src/yolo_pi/imu.py`)

1. **Per-sensor conditioning & liveness:**
   - Distinguishes healthy `no_target` returns (open space beyond 2.0 m on VL53L0X or >25 ms timeout on HC-SR04) from true hardware faults (`valid=False` / stale > 250 ms).
   - 3-sample median filter with an adaptive noise-immune 2-sample fast-attack bypass (`max(0.08 m, 4% * median)`) rejects single-ping acoustic spikes while keeping step-closer reaction time to 2 samples.
2. **IMU compensation (`IMUProcessor`):**
   - Startup gyro-bias calibration + adaptive 6-axis complementary filter (drops accelerometer trust during footstep impacts).
   - Horizontal range projection (`cos(Δpitch)`), ground-echo suppression when pitched downward (respecting a hard 0.35 m safety floor), TTC gating during rapid downward tilt, vision latency yaw-lag compensation (`Δx = ω_z · Δt_age`), stationary anti-fatigue softening (never softens `CRITICAL`), and drop detection.
3. **Co-aligned Center inverse-variance + consistency fusion:**
   - Combines `tof-center`, `us-center-1`, and `us-center-2` using freshness-inflated Gauss-Markov BLUE inverse-variance weighting when readings agree.
   - When sensors disagree (e.g. narrow pole only in ToF or glass door only in HC-SR04), drops farther dissenters, rejects single-ping transient ghosts, and trusts closer obstacles once confirmed by 2 distinct sensor timestamps.
   - Tick-rate-independent 1D α-β tracker computes closing velocity and Time-to-Collision (TTC) only on fresh sensor timestamps.
4. **Distance-scheduled range + YOLO vision fusion:**
   - **Short range (`≤ 1.0 m`):** Physical ToF + HC-SR04 unconditionally own `CRITICAL` (`≤ 0.50 m`) and `WARNING` (`≤ 1.00 m`). Vision enriches with semantic labels and can never suppress, delay, or downgrade a range hazard.
   - **Transition (`1.0–2.0 m`):** Smooth cosine/sigmoid weight handoff from range to YOLO with cross-modal agreement boost.
   - **Longer range (`> 2.0 m` up to `3.8 m`) & range-blind hazards (`stairs`, `curb`, `vehicle`):** Monocular distance estimation + per-class severity weights + 2-frame temporal persistence filter drive directional `ADVISORY` (or `WARNING` for high-severity hazards like `stairs`/`curb`/`vehicle`, never vision-only `CRITICAL`).
5. **Per-sector asymmetric hysteresis & directional 3-motor haptics:**
   - Zero-delay escalation on range danger; deadband + minimum dwell on de-escalation (`0.58 m / 150 ms` for CRITICAL, `1.12 m / 200 ms` for WARNING) eliminates boundary chatter.
   - Left, Center, and Right DRV2605L motors actuate independently (`RTP` or `library` mode) with a 500 ms hardware watchdog and guaranteed zero-on-exit teardown.

## Running on Raspberry Pi 5

Install the lean zero-torch hardware stack on the Pi (keeps peak RAM well below the 2 GB ceiling):

```bash
cd ~/capstone-pi-benchmarking
git pull origin main
uv pip install -e ".[pi-hardware]"
```

Ensure I2C is enabled (`sudo raspi-config nonint do_i2c 0`) and verify the mux and devices are visible:

```bash
i2cdetect -y 1
```

Run the 14-scenario physics simulator first to confirm the software stack:

```bash
.venv/bin/python scripts/simulate_fusion.py
```

Run the live smart cane runtime (safe 2-thread thermal budget by default; use `--no-camera` to test sensors + haptics alone first):

```bash
# 1. Sensors + IMU + directional haptics only (fast 50 Hz loop check)
.venv/bin/python scripts/run_smart_cane.py --mode hardware --no-camera

# 2. Full live pipeline (ToF + 2x HC-SR04 + MPU-6050 + IMX219 OpenVINO 416 + 3x DRV2605L)
.venv/bin/python scripts/run_smart_cane.py --mode hardware --threads 2 --haptic-mode rtp
```

If your MPU-6050 is wired directly on the main I2C bus rather than `CH7` of the TCA9548A, set `"mux_channel": null` under `"imu"` in `configs/smart-cane-pi5.json`.
