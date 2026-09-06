# YOLO Pi Cane Toolkit

This repository prepares, deploys, and benchmarks the software side of a
low-latency haptic cane on a Raspberry Pi 5.

The laptop is used for functional inference checks, dataset preparation,
fine-tuning, export parity, deterministic sensor replay, fusion-policy tests, and
benchmark-harness validation. Raspberry Pi measurements remain the authority for
latency, FPS, CPU, RAM, temperature, throttling, power, camera, physical sensors,
and camera-to-haptic response time.

## Raspberry Pi 5 first-pass results

Both formal matrices completed on a Raspberry Pi 5 Model B Rev 1.1 with 2 GB
RAM, 64-bit Raspberry Pi OS, FP32 models at 640 pixels, and four threads. Every
row used 20 excluded warmups followed by 100 measured observations.

The returned evidence is versioned under
[`artifacts/pi-benchmarks/returned/2026-09-06/`](artifacts/pi-benchmarks/returned/2026-09-06/).
It contains every raw row, environment record, console log, generated CSV and
Markdown table, the Pi-side OpenVINO smoke, configuration snapshots, and the
original checksum-verified transfer archive.

### Model quality and PyTorch comparison

Quality metrics come from the common 1,515-image held-out test set. Pi timings
come from the completed architecture rows.

| Source | Model | Mean inference (ms) | FPS | Peak RAM (MB) | Temperature max (C) | mAP50 (%) | mAP50-95 (%) | Recall (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed images | YOLO26n pretrained | 362.69 | 2.70 | 363.84 | 72.15 | 24.13 | 17.49 | 23.32 |
| fixed images | YOLO26n-Cane V1 | 394.40 | 2.49 | 367.38 | 77.10 | 52.72 | 36.32 | 51.00 |
| live IMX219 | YOLO26n pretrained | 290.98 | 3.39 | 361.62 | 69.95 | 24.13 | 17.49 | 23.32 |
| live IMX219 | YOLO26n-Cane V1 | 279.88 | 3.53 | 387.98 | 78.20 | 52.72 | 36.32 | 51.00 |

Fine-tuning improved mAP50 by 28.59 percentage points, mAP50-95 by 18.83
points, and recall by 27.68 points under the shared held-out protocol.

### Fine-tuned runtime comparison

| Runtime | Fixed-image inference (ms) | Fixed-image FPS | Camera inference (ms) | Camera-to-model-finish (ms) | Camera FPS | Camera peak RAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch | 379.41 | 2.58 | 270.07 | 316.07 | 3.64 | 394.16 |
| ONNX Runtime | 311.09 | 3.03 | 294.86 | 347.40 | 3.26 | 541.25 |
| OpenVINO | 173.16 | 5.43 | **166.21** | **215.57** | **5.76** | 594.81 |
| MNN | 175.31 | 5.42 | 189.96 | 346.68 | 5.07 | 574.16 |
| NCNN | 272.18 | 3.55 | 181.53 | 330.12 | 5.32 | 582.98 |
| LiteRT | **147.33** | **6.43** | 171.59 | 217.89 | 5.62 | 631.83 |

LiteRT was fastest on fixed images. OpenVINO produced the best live-camera
combination: lowest mean inference time, lowest camera-to-model-finish time,
and highest measured camera FPS. The live annotated preview therefore selects
the fine-tuned OpenVINO export by default.

These are preliminary measurements. Saved row evidence records nonzero
`get_throttled` flags after portions of both matrices, maximum temperatures
near 85 C, and `NA` power because no external meter was used. Repeat the
matrices under stable power and controlled cooling before treating small
runtime differences as final. The camera benchmark measures through model
completion; physical camera-to-haptic latency remains unmeasured.

Detailed generated tables:

- [fixed-image model comparison](artifacts/pi-benchmarks/returned/2026-09-06/artifacts/pi-benchmarks/images-20260905T103604Z/tables/model-architecture-benchmark.md)
- [fixed-image runtime comparison](artifacts/pi-benchmarks/returned/2026-09-06/artifacts/pi-benchmarks/images-20260905T103604Z/tables/runtime-backend-benchmark.md)
- [live-camera model comparison](artifacts/pi-benchmarks/returned/2026-09-06/artifacts/pi-benchmarks/camera-preliminary-20260905T180557Z/tables/model-architecture-benchmark.md)
- [live-camera runtime comparison](artifacts/pi-benchmarks/returned/2026-09-06/artifacts/pi-benchmarks/camera-preliminary-20260905T180557Z/tables/runtime-backend-benchmark.md)

## Planned model progression

```text
V0  yolo26n.pt pretrained baseline
V1  V0 + harmonized public assistive-navigation datasets
V2  V1 + 500–1500 images from the finalized cane camera
```

V2 intentionally waits until the camera mount and viewpoint are final. The
current deployment contract is a forward-facing camera within 80 mm of the top
handle grip, defined in `configs/camera-mount-top-handle.json`. Its nominal
dimensions remain provisional until measured on the physical cane.

## Development setup

The core package has no mandatory third-party dependencies:

```bash
python3 -m unittest discover -s tests -v
```

Install the optional vision runtime when running real inference:

```bash
uv sync --extra vision --extra edge --extra export --extra audit --extra pi --extra dev
uv run yolo-pi inspect-model --model yolo26n.pt --input path/to/image.jpg
uv run yolo-pi validate-mount --profile configs/camera-mount-top-handle.json
```

The model command may download pretrained weights through Ultralytics. Keep the
fixed evaluation inputs and generated model artifacts under their documented
directories rather than committing large binaries.

The `edge` extra is required when loading an exported NCNN model through the
Ultralytics adapter. It is separate from `vision` so training-only environments
do not install a deployment runtime they do not use.

Recreate and checksum the fixed evaluation input, then export the V0 baseline:

```bash
uv run python scripts/fetch_evaluation_inputs.py
uv run python scripts/export_baseline.py \
  --model artifacts/models/yolo26n.pt \
  --sample artifacts/models/bus.jpg \
  --imgsz 416
```

## Live inference on the Raspberry Pi

After the Raspberry Pi environment, IMX219 camera, and Cane V1 runtime artifacts
are configured, this is the command to start the annotated live inference demo:

```bash
cd ~/capstone-pi-benchmarking
PYTHONPATH=src .venv/bin/python scripts/live_camera_demo.py
```

Run this from the Pi desktop / Raspberry Pi Connect Screen Sharing when you want
to view live detections. The demo is qualitative and should not run at the same
time as a formal benchmark or another process that owns the camera. The full
camera setup and preview procedure is also documented in
`docs/pi-terminal-setup-and-run.md`.

## Project layout

```text
configs/       versioned experiment definitions
notebooks/     Colab training/export companion
scripts/       GPU and deployment helpers
src/yolo_pi/   detector, benchmark, fusion, and dataset code
tests/         deterministic unit and integration-style tests
```

See `changes.md` for the evidence log, `docs/pi-handoff.md` for the physical
handoff, and `docs/pi-benchmarking.md` for the resumable runner and the exact
saved architecture/runtime table outputs.

For the actual Raspberry Pi session, use
`docs/pi-terminal-setup-and-run.md`. It is the standalone terminal-only runbook
for Raspberry Pi Connect Remote Shell, Tailscale rsync when Wi-Fi blocks local
SSH, Raspberry Pi OS
packages, uv/Picamera2 setup, checksum and
OpenVINO gates, production registry creation, `tmux`, fixed-image runs, IMX219
camera runs, power input, resume/retry, result inspection, evidence return, and
common mistakes. The Linux ARM64 dependency plan uses PyTorch's official CPU
wheel index and must never download NVIDIA CUDA packages on the Pi.

`docs/completion-readiness.md` is the requirement-by-requirement audit. It
separates completed model/software evidence from measurements that
must come from the physical Pi/IMX219 rather than a laptop or cloud GPU.

Team members should start with `docs/data-preprocessing.md` for the singular,
current explanation of raw sources, class mappings, split/decontamination rules,
resize policy, final counts, limitations, and reproduction commands.

The completed GPU handoff and recovery design remain documented in
`docs/lightning-ai-training.md`. The returned epoch-30 package has been accepted;
`artifacts/training/lightning-final/best.pt` is the deployment selection and
`latest-resumable.pt` is the optimizer-bearing recovery model.
`scripts/finalize_lightning_return.py` completed import, contiguous history,
five exports, and the complete held-out comparison. The Linux/ARM OpenVINO
functional smoke and production registry gate subsequently passed on the Pi,
and both formal benchmark matrices completed.
