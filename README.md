# YOLO Pi Cane Toolkit

This repository prepares and verifies the software side of a low-latency haptic
cane before Raspberry Pi 5 benchmarking begins.

The laptop is used for functional inference checks, dataset preparation,
fine-tuning, export parity, deterministic sensor replay, fusion-policy tests, and
benchmark-harness validation. Raspberry Pi measurements remain the authority for
latency, FPS, CPU, RAM, temperature, throttling, power, camera, physical sensors,
and camera-to-haptic response time.

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
for Raspberry Pi Connect Remote Shell, LAN rsync transfer, Raspberry Pi OS
packages, uv/Picamera2 setup, checksum and
OpenVINO gates, production registry creation, `tmux`, fixed-image runs, IMX219
camera runs, power input, resume/retry, result inspection, evidence return, and
common mistakes.

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
`scripts/finalize_lightning_return.py` has completed import, contiguous history,
five exports, and the complete held-out comparison. Its only pending state is a
Linux/ARM OpenVINO functional smoke, which the terminal Pi runbook performs
before it generates `configs/pi-model-registry.json` and starts formal timing.
