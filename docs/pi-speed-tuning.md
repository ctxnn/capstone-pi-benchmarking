# Cane V1 speed tuning on the Raspberry Pi 5

This records the measured effect of reducing the detector input size, so the
choice of deployment model is evidence-backed rather than assumed. Every Pi
number below was measured on the physical Pi; every quality number comes from
the common 1,515-image held-out test split.

## Why input size dominates

The accepted Cane V1 model was trained at `imgsz=416`, but every row of the
2026-09-06 formal matrix ran it at 640. A 640 input does 2.4x the arithmetic of
416 and shows the model objects at a larger scale than it learned. Restoring the
training resolution is therefore both faster and more accurate.

The exports were produced from the accepted `best.pt` (SHA-256
`85168a2b...`) with Ultralytics 8.4.132 and OpenVINO 2026.3.1:

```bash
yolo export model=best.pt format=openvino imgsz=416 batch=1 device=cpu
yolo export model=best.pt format=openvino imgsz=320 batch=1 device=cpu
```

Only the two OpenVINO FP32 exports are kept in
`artifacts/models/cane-v1-fast/`, together with `export-manifest.json` (source
hash, tree hashes, quality at each export's own size) and `quality.json` (the
complete sweep below). They are tracked in Git on purpose: the Pi pulls them
from the repository, and the tree hashes catch a truncated copy. The rejected
INT8, NCNN FP16 and LiteRT INT8 exports are not kept; they can be regenerated
with `int8=True data=...` or `format=ncnn half=True`.

## Held-out quality

Evaluated with the documented test protocol; `OpenVINO` and `NCNN` rows were
scored after export, so they include any conversion loss. `LiteRT INT8` is not
competitive and is omitted from the deployment shortlist.

| Artifact | Input | mAP50 | mAP50-95 | Recall |
| --- | --- | ---: | ---: | ---: |
| PyTorch FP32 (current deployment input) | 640 | 52.72 | 36.32 | 51.00 |
| PyTorch FP32 | 512 | 60.34 | 43.22 | 55.60 |
| PyTorch FP32 | 416 | 60.80 | 43.95 | 55.89 |
| PyTorch FP32 | 352 | 59.49 | 42.85 | 53.59 |
| PyTorch FP32 | 320 | 58.54 | 42.12 | 53.35 |
| OpenVINO FP32 | 416 | **62.13** | **43.49** | **58.29** |
| OpenVINO INT8 | 416 | 61.39 | 42.37 | 59.04 |
| NCNN FP16 | 416 | 62.12 | 43.51 | 58.13 |
| OpenVINO FP32 | 320 | 59.10 | 41.59 | 54.04 |
| LiteRT INT8 | 416 | 55.57 | 33.78 | 55.54 |

Every 416 and 320 artifact beats the 640 deployment on quality *and* on work
per frame. `door`, `slope` and `overhead_obstacle` still have no annotations, so
these numbers say nothing about those classes.

## Pi throughput

30 fixed manifest images, cycled, five excluded warmups followed by 40 measured
frames per artifact. Each artifact ran in its own process with a 30 s idle gap
to keep the power draw inside what this Pi's supply can deliver.

| Artifact | Input | Inference mean (ms) | Total (ms) | P95 (ms) | FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| OpenVINO FP32 (current) | 640 | 242.9 | 252.2 | 281.1 | 3.96 |
| OpenVINO FP32 | 416 | 99.2 | 105.3 | 128.4 | 9.49 |
| OpenVINO FP32 | 320 | **62.6** | **67.4** | 81.4 | **14.85** |
| OpenVINO INT8 | 416 | 530.4 | 536.3 | 570.8 | 1.86 |
| NCNN FP16 | 416 | 446.5 | 452.4 | 479.0 | 2.21 |

Under identical conditions, 416 is **2.4x** faster than the current deployment
input and 320 is **3.9x** faster, while both remain more accurate.

These fixed-image numbers are slower than the camera numbers below (99.2 ms
versus 61.5 ms of inference for 416). The two sessions differed: the fixed-image
runs followed several reboots caused by the supply, and decode of saved JPEGs
competes for the same two cores. Quote the camera table for deployment and the
fixed-image table only for the relative ranking until the formal R3/R7/R8 rows
measure both sources under one protocol. A separate
single-artifact run at a 1.8 GHz cap rather than 1.5 GHz measured 416 at 73.1 ms
(12.6 FPS), so the clock limitation below costs roughly a quarter of the speed.

### Why INT8 and NCNN FP16 are dropped

OpenVINO's INT8 path is 5.3x slower than its FP32 path on this ARM CPU, and the
NCNN build available for the Pi emulates FP16 rather than using the CPU's native
support, making it 4.5x slower. Neither is a deployment candidate here.

## Live IMX219 camera latency

Both artifacts measured back to back on the same camera, same session, with
`scripts/live_camera_demo.py --no-display --threads 2`, 15 s each. `cam->result`
runs from the frame's sensor timestamp to a finished detection, so it includes
capture availability, letterboxing, inference and postprocessing.

| Artifact | Input | Inference mean (ms) | Frame age (ms) | cam->result mean (ms) | cam->result p95 (ms) | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OpenVINO FP32 (current) | 640 | 149.2 | 41.0 | 196.5 | 228.2 | 6.43 |
| OpenVINO FP32 | 416 | **61.5** | **23.8** | **89.6** | 135.3 | **15.20** |

That is a 2.4x reduction in inference time and a 2.2x reduction in
camera-to-result latency, with 2.4x the throughput. The published 2026-09-06
figure for the same path was 215.6 ms camera-to-model-finish; the 416 artifact
measures 89.6 ms on the same hardware.

Dropped frames are expected and intentional: the camera produces 30 fps while
the loop consumes 15.2, and the one-slot buffer discards the surplus instead of
queueing it. Frame age stays at 24 ms, which is the proof that no backlog forms.

Camera-to-haptic latency remains unmeasured. That requires the vibration
hardware and the range sensors, but the vision portion of that budget is now
about 126 ms smaller.

## Measurement conditions that must be stated with these numbers

- Both Pi comparison runs used `--threads 2` and a 1.5 GHz maximum clock. The
  board reports `throttled=0x50000` (under-voltage, currently throttled) with
  `EXT5V_V` at 4.80 V on a supply without a fan. Sustained four-core load at
  full clock reboots the Pi under this supply, so all comparisons were made
  under the same reduced setting instead of at the hardware maximum.
- `avahi-daemon` was found spinning at 97% of one core (repeatedly
  re-registering `capstone16-2.local`) and was stopped before these runs. Any
  earlier measurement taken while it was spinning is contaminated.
- Peak temperatures stayed between 57 and 63 C, so thermal throttling was not a
  factor in these runs; the binding limit is supply voltage.
- These are single-artifact quick comparisons, not replacements for the formal
  resumable matrix, which also records CPU, RAM, temperature, power and P99.
  The camera comparison is one 15-second run per artifact, pointed at a person.

## Formal matrix rows

The matrix now carries the exports as R7 (OpenVINO FP32, 416) and R8 (OpenVINO
FP32, 320). Register them with
`--input-size-manifest artifacts/models/cane-v1-fast/export-manifest.json` on
`scripts/build_pi_model_registry.py`, then run R3, R7 and R8 together for both
sources as described in `docs/pi-terminal-setup-and-run.md` sections 8, 10 and
12. Running R3 in the same series keeps 640 under identical conditions. Once the
supply is replaced, repeat the series at four threads and full clock; those rows
supersede every number on this page.

## Pipeline prerequisites found while measuring

- The IMX219 is a third-party module, so `camera_auto_detect=1` never finds it;
  it auto-detects genuine Raspberry Pi cameras and OV5647 only. An explicit
  `dtoverlay=imx219` line is required. The bare overlay addresses `cam1`; the
  `,cam0` suffix addresses `cam0`; a `,cam1` suffix does not exist and is
  silently ignored.
- The ribbon must be the 22-pin type on the Pi 5 end. A partially seated or
  laterally offset ribbon can leave I2C working (the sensor enumerates and
  `rpicam-hello --list-cameras` succeeds) while the MIPI data lanes fail, which
  surfaces as repeated `Camera frontend has timed out!` and no frames. Reseating
  fixes it; no software change does.

## Deployment choice

`artifacts/models/cane-v1-fast/ov-fp32-416/` is the recommended deployment
artifact: more accurate than the 640 export and 2.4x faster. Use the 320 export
only if the live camera test shows the 416 pipeline cannot keep up on the final
hardware.

Run the live preview headless with:

```bash
cd ~/capstone-pi-benchmarking
PYTHONPATH=src .venv/bin/python -u scripts/live_camera_demo.py \
  --model artifacts/models/cane-v1-fast/ov-fp32-416/best_openvino_model \
  --no-display --duration 30
```

`--no-display` skips drawing and the window, which is the deployment mode on the
cane and the only honest way to measure camera-to-result latency.
