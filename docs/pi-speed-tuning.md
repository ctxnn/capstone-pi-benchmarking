# Cane V1 speed tuning on the Raspberry Pi 5

This records the measured effect of reducing the detector input size, so the
choice of deployment model is evidence-backed rather than assumed. Every Pi
number below was measured on the physical Pi; every quality number comes from
the common 1,515-image held-out test split.

## Why input size dominates

The accepted Cane V1 model was trained at `imgsz=416`, but every row of the
2026-09-06 formal matrix ran it at 640. A 640 input does 2.4x the arithmetic of
416, and because the IMX219 frames are 640x480, a square 640 input also pads
away roughly a quarter of the tensor. Restoring the training resolution is
therefore both faster and more accurate.

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
input and 320 is **3.9x** faster, while both remain more accurate. A separate
single-artifact run at a 1.8 GHz cap rather than 1.5 GHz measured 416 at 73.1 ms
(12.6 FPS), so the clock limitation below costs roughly a quarter of the speed.

### Why INT8 and NCNN FP16 are dropped

OpenVINO's INT8 path is 5.3x slower than its FP32 path on this ARM CPU, and the
NCNN build available for the Pi emulates FP16 rather than using the CPU's native
support, making it 4.5x slower. Neither is a deployment candidate here.

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
