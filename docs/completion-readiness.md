# Goal completion and evidence audit

This file maps every explicit project requirement to current authoritative
evidence. A software implementation is not treated as physical Pi evidence, and
a prepared workflow is not treated as a finished model.

Last audited: 2026-08-30.

| Requirement | Current state | Authoritative evidence | Remaining gate |
|---|---|---|---|
| Keep a major-change/rationale log | Complete and active | `changes.md` | Continue updating after final model and Pi run |
| Fine-tuned Cane V1 model | In progress | Protected epoch-22 optimizer checkpoint SHA-256 `0da75726f7a058e63d7b030601447ddeb4bd98b420d1387482761aa5799a0eea`; `lightning-ai-upload/` | Lightning must finish or early-stop, validate, and return its ZIP |
| Save latest model if agent/provider time ends | Complete | At startup and each completed epoch, Lightning writes immutable `last-epoch-NNN.pt`, atomic `LATEST_RESUMABLE.json`, and CRC-checked `cane-v1-latest-recovery.zip`; returned final ZIP must include `latest-resumable.pt` with optimizer state | Verify actual remote/returned recovery checkpoint |
| Save finished model | Implemented, awaiting run | Lightning completion receipt binds `best.pt`, `last.pt`, val/test metrics, hashes, and final ZIP | Verify `LIGHTNING_COMPLETE.json` and local importer receipt |
| M1 pretrained / M2 fine-tuned architecture table generator | Complete | `configs/pi-benchmark-matrix.json`; `src/yolo_pi/pi_benchmark.py`; generated audit has exactly two rows | Final model quality and physical Pi measurements |
| Architecture metrics: inference mean/P50/P95/P99, FPS, RAM, CPU, temperature, mAP50, mAP50-95, recall | Complete schema | Generated `model-architecture-benchmark.csv` contains every named column | Values stay `NA` until quality/Pi evidence exists |
| R1–R6 PyTorch/ONNX/OpenVINO/MNN/NCNN/LiteRT at 640 FP32 | Complete matrix | `configs/pi-benchmark-matrix.json`; generated runtime CSV has exactly six rows in requested order | Export final model and run on Pi |
| Runtime metrics: preprocess/inference/postprocess/total, P50/P95/P99, FPS, CPU, RAM, temperature, power | Complete schema and collector | `src/yolo_pi/pi_benchmark.py`; generated runtime CSV | Physical Pi run; power additionally needs timestamped external-meter JSONL |
| Save raw benchmark observations and regenerated tables | Complete | Resumable row JSON files plus CSV, Markdown, and `table-manifest.json` described in `docs/pi-benchmarking.md` | Run on Pi |
| Preserve saved-image benchmark | Complete | Default `--source images`; 30-file checksum-verified test manifest under `artifacts/benchmarks/pi-runtime-inputs/` | Transfer unchanged pack to Pi |
| Flexible `images` versus `camera` input | Complete | `src/yolo_pi/input_sources.py`; CLI `--source {images,camera}` | Physical camera run |
| Isolate Picamera2/IMX219 code | Complete | `src/yolo_pi/picamera2_source.py`; Picamera2 imported only for camera source | Install Raspberry Pi OS Picamera2 package |
| Avoid stale queues and process latest frame | Complete in software | One-slot `LatestFrameBuffer`, overwrite count, `capture_request(flush=True)`; unit tests | Observe with actual IMX219 |
| Add capture and frame-age timing | Complete in software | Capture, availability, sensor age, model-start age, camera-to-model-finish, and dropped-frame fields flow into raw/table outputs | Observe with actual IMX219 |
| Keep model/runtime/resolution/threads/source configurable | Complete | Matrix/registry plus CLI flags shown by `scripts/run_pi_benchmarks.py --help` | Use fixed choices for formal run |
| Pretrained and fine-tuned compatibility | Complete architecture | Registry keys and common Ultralytics detector adapter support both M1/M2 | Populate production registry from returned model |
| Camera near/on top handle | Complete provisional contract | `configs/camera-mount-top-handle.json`: `near_top_handle`, 40 mm below grip, forward, 12-degree downward pitch | Measure mount; set `hardware_verified=true` only after physical checks |
| Detailed singular data-preprocessing document | Complete | `docs/data-preprocessing.md` contains sources, hashes, mappings, counts, split protection, decontamination, resize, model-time transforms, reproduction, and current host/checkpoint | Update final environment/metrics after Lightning return |
| Final backend export/quality/registry pipeline | Implemented, awaiting model | `scripts/finalize_lightning_return.py` and `artifacts/training/finalization-state.json` contract | Run after ZIP arrives; merge Linux OpenVINO smoke if macOS cannot execute it |
| Automated verification | Complete at current state | 72/72 tests; executed notebook validates with 7 code cells and 0 error outputs | Repeat after final artifacts and Pi results |

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

Expected current results are 72 passing tests, a checksum-valid Lightning upload
folder, a valid seven-code-cell notebook, 30 fixed images, a valid provisional top-handle profile with
`hardware_verified=false`, two architecture rows, and six runtime rows.

## Conditions that still prevent goal completion

The goal is not complete yet because two kinds of evidence cannot be fabricated:

1. Lightning has not yet returned the finished/early-stopped fine-tuned model
   package. The epoch-22 checkpoint is recoverable progress, not the final
   deployment selection.
2. Raspberry Pi, IMX219, thermal, RAM/CPU, and measured-power values require the
   physical hardware run. The software deliberately renders them as `NA` rather
   than copying laptop or cloud measurements.

After the Lightning ZIP is placed in `artifacts/training/lightning-inbox/`, run:

```bash
.venv/bin/python scripts/finalize_lightning_return.py --device cpu
```

After the production registry is complete, follow `docs/pi-benchmarking.md` on
the Raspberry Pi and retain the entire timestamped run directory.
