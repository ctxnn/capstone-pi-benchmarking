# Goal completion and evidence audit

This file maps every explicit project requirement to current authoritative
evidence. A software implementation is not treated as physical Pi evidence, and
a prepared workflow is not treated as a finished model.

Last audited: 2026-08-30.

| Requirement | Current state | Authoritative evidence | Remaining gate |
|---|---|---|---|
| Keep a major-change/rationale log | Complete and active | `changes.md` | Continue updating after final model and Pi run |
| Fine-tuned Cane V1 model | Complete | Accepted epoch-30 `best.pt` SHA-256 `85168a2bc2f8a87c6b4361f16e3e7ec7394312341ee99a766f368be6e28e0dca`; local import receipt | Physical Pi performance remains separate |
| Save latest model if agent/provider time ends | Complete and verified | Epoch-30 `latest-resumable.pt` retains optimizer state and exact taxonomy; SHA-256 `e9f3548158be346d7e037416ef7b8f3160efea6d3bba13f170c1498d7c6b0b2c`; Downloads copy matches | None for training recovery |
| Save finished model | Complete | `best.pt`, `last.pt`, latest resumable model, val/test metrics, hashes, accepted ZIP, and local importer receipt | None for model acceptance |
| M1 pretrained / M2 fine-tuned architecture table generator | Complete | Full 1,515-image comparison at 640 plus exact two-row table schema | Physical Pi latency/resource measurements |
| Architecture metrics: inference mean/P50/P95/P99, FPS, RAM, CPU, temperature, mAP50, mAP50-95, recall | Complete schema | Generated `model-architecture-benchmark.csv` contains every named column | Values stay `NA` until quality/Pi evidence exists |
| R1–R6 PyTorch/ONNX/OpenVINO/MNN/NCNN/LiteRT at 640 FP32 | Converted and matrix-ready | Final export manifest; ONNX/MNN/NCNN/LiteRT smoke complete; generated runtime CSV has exactly six rows | Run checksum-bound OpenVINO smoke on Pi, build registry, then measure |
| Runtime metrics: preprocess/inference/postprocess/total, P50/P95/P99, FPS, CPU, RAM, temperature, power | Complete schema and collector | `src/yolo_pi/pi_benchmark.py`; generated runtime CSV | Physical Pi run; power additionally needs timestamped external-meter JSONL |
| Save raw benchmark observations and regenerated tables | Complete | Resumable row JSON files plus CSV, Markdown, and `table-manifest.json` described in `docs/pi-benchmarking.md` | Run on Pi |
| Preserve saved-image benchmark | Complete | Default `--source images`; 30-file checksum-verified test manifest under `artifacts/benchmarks/pi-runtime-inputs/` | Transfer unchanged pack to Pi |
| Flexible `images` versus `camera` input | Complete | `src/yolo_pi/input_sources.py`; CLI `--source {images,camera}` | Physical camera run |
| Isolate Picamera2/IMX219 code | Complete | `src/yolo_pi/picamera2_source.py`; Picamera2 imported only for camera source | Install Raspberry Pi OS Picamera2 package |
| Avoid stale queues and process latest frame | Complete in software | One-slot `LatestFrameBuffer`, overwrite count, `capture_request(flush=True)`; unit tests | Observe with actual IMX219 |
| Add capture and frame-age timing | Complete in software | Capture, availability, sensor age, model-start age, camera-to-model-finish, and dropped-frame fields flow into raw/table outputs | Observe with actual IMX219 |
| Keep model/runtime/resolution/threads/source configurable | Complete | Matrix/registry plus CLI flags shown by `scripts/run_pi_benchmarks.py --help` | Use fixed choices for formal run |
| Pretrained and fine-tuned compatibility | Complete and quality-bound | Baseline/fine-tuned hashes and full held-out metrics in `model-comparison-640.json`; common detector adapter | Pi-side OpenVINO smoke unlocks production registry generation |
| Camera near/on top handle | Complete provisional contract | `configs/camera-mount-top-handle.json`: `near_top_handle`, 40 mm below grip, forward, 12-degree downward pitch | Measure mount; set `hardware_verified=true` only after physical checks |
| Detailed singular data-preprocessing document | Complete | `docs/data-preprocessing.md` contains sources, hashes, mappings, counts, split protection, decontamination, resize, transforms, final environment, artifact hashes, and both quality protocols | Update only if dataset/model protocol changes |
| Final backend export/quality/registry pipeline | Awaiting one Pi functional gate | Import/history/exports/full comparison complete; state is `awaiting_linux_export_smoke` | Run OpenVINO smoke and registry commands in terminal runbook on Pi |
| Terminal-only Pi setup and run instructions | Complete | `docs/pi-terminal-setup-and-run.md` covers an already-open Raspberry Pi Connect shell, Tailscale rsync for client-isolated Wi-Fi, OS/uv/Picamera2 setup, artifact checks, OpenVINO/registry gate, tmux, images, camera, power, resume, outputs, return, and common mistakes | Execute on physical Pi |
| Automated verification | Complete at current state | 75/75 tests; executed notebook validates with 7 code cells and 0 error outputs | Repeat after Pi evidence returns |

## Reproducible audit commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_notebook.py notebooks/cane_v1_training_export_output.ipynb
.venv/bin/python scripts/prepare_lightning_handoff.py --check

PYTHONPATH=src .venv/bin/python -c \
  "from pathlib import Path; from yolo_pi.pi_benchmark import load_fixed_inputs; print(len(load_fixed_inputs(Path('artifacts/benchmarks/pi-runtime-inputs/manifest.json'), Path.cwd())))"

.venv/bin/yolo-pi validate-mount \
  --profile configs/camera-mount-top-handle.json

PYTHONPATH=src .venv/bin/python scripts/compile_pi_tables.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.example.json \
  --runs EMPTY_RUN_DIRECTORY \
  --output TABLE_OUTPUT_DIRECTORY
```

Expected current results are 75 passing tests, an accepted epoch-30 model,
contiguous 30-epoch history, five converted backends, complete 1,515-image model
comparison, a valid seven-code-cell notebook, 30 fixed images, a valid provisional top-handle profile with
`hardware_verified=false`, two architecture rows, and six runtime rows.

## Conditions that still prevent goal completion

The model and benchmark software are ready. The overall experimental goal is
not complete because Raspberry Pi, IMX219, thermal, RAM/CPU, and measured-power values require the
   physical hardware run. The software deliberately renders them as `NA` rather
   than copying laptop or cloud measurements.

Follow `docs/pi-terminal-setup-and-run.md` through Raspberry Pi Connect Remote
Shell. Its Pi-side OpenVINO smoke
is the last functional gate and produces the production registry immediately
before the formal saved-image and camera matrices.
