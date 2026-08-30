# Reproducible Raspberry Pi benchmark runner

The runner saves raw observations first and derives both requested tables from
those saved files. It refuses to label a run as Raspberry Pi evidence unless
`/proc/device-tree/model` identifies Raspberry Pi hardware. The escape hatch
`--allow-non-pi` exists only for harness diagnostics and marks that scope in
every row.

Saved images remain the default and preserve the fixed/checksummed comparison.
The optional camera source is isolated in `src/yolo_pi/picamera2_source.py`; it
does not make Picamera2 a dependency for laptop tests or image benchmarks.

The saved-image comparison uses 30 deterministic images selected only from the
final Cane V1 **test** split. The selector first covers every populated class,
then adds up to three empty-label scenes and evenly spaced test images. This pack
is a repeatable runtime/postprocessing input, not a substitute for test-set mAP
evaluation.

Create it once on the laptop after building `cane-v1-training-640`:

```bash
uv run python scripts/prepare_pi_runtime_inputs.py \
  --dataset data/processed/cane-v1-training-640 \
  --output artifacts/benchmarks/pi-runtime-inputs \
  --count 30
```

Transfer the complete `artifacts/benchmarks/pi-runtime-inputs/` directory to the
same repository-relative location on the Pi. The benchmark loader verifies every
image against `manifest.json` before warmup begins. The checked matrix points to
that manifest, so a missing, renamed, or changed image fails before collecting a
row.

## 1. Install and register artifacts

```bash
uv sync \
  --extra pi \
  --extra edge \
  --extra export \
  --extra runtime-openvino \
  --extra runtime-mnn \
  --extra runtime-litert
cp configs/pi-model-registry.example.json configs/pi-model-registry.json
```

Replace every placeholder path/hash in the copied registry with the transferred
V0 and Cane V1 artifacts. For the production registry, use
`scripts/build_pi_model_registry.py` rather than editing hashes by hand. It
requires the final training summary, exact export manifest, and the shared
640-pixel quality comparison and refuses checkpoint/hash mismatches. A directory
export has a file-level manifest plus a deterministic tree hash.

The M1/M2 accuracy cells must come from
`scripts/evaluate_model_comparison.py` on the complete 1,515-image test split at
640. The fine-tuned model uses the native 15-class taxonomy. The stock model's
COCO predictions are mapped by explicit class name only (`person`; car/bus/truck
to `vehicle`; bicycle/motorcycle; dog; stop sign), while unsupported target
hazards stay in the ground truth and therefore count as false negatives. This
prevents the numerically incompatible COCO and Cane class IDs from producing a
misleading baseline.

Run an ordinary one-image functional check on all six backends before collecting
timings. A missing or unloadable backend produces a saved failed row; it is never
silently omitted or replaced with a different runtime.

R4 has an additional precision guard. Ultralytics 8.4.x defaults its native MNN
runtime to `precision=low` and derives its own thread count, which would not
match the requested FP32/four-thread row. This repository replaces only that
loader with an explicit native-MNN configuration of `precision=high`, CPU
backend, and the row's configured thread count. The runner records the resolved
backend and rejects R4 unless `mnn-high-precision` is actually active. If the Pi
console says `MNN use low precision`, discard that row rather than relabeling it
FP32.

Also confirm the fixed saved-image pack resolves and all 30 hashes pass:

```bash
PYTHONPATH=src python -c \
  "from pathlib import Path; from yolo_pi.pi_benchmark import load_fixed_inputs; print(len(load_fixed_inputs(Path('artifacts/benchmarks/pi-runtime-inputs/manifest.json'), Path.cwd())))"
```

The expected output is `30`.

## 2. Stabilize the device

- Use the same 64-bit OS image, active cooler, enclosure, power supply, camera
  mode, CPU governor, swap policy, and background services for every row.
- Keep the top-handle camera profile fixed and record `hardware_verified=true`
  only after measuring the actual mount.
- Stop desktop/browser workloads and disconnect any avoidable load.
- Attach the external power meter before the run if power is required.

The matrix cools the CPU to 55 C (or times out after ten minutes) before every
row, excludes 20 warmups, and records 100 iterations. Change those values only
before the experiment series and retain the changed matrix with the results.

## 3. Optional measured-power stream

Power is not inferred from CPU load. If an inline sensor/logger is available,
write newline-delimited JSON using Unix epoch nanoseconds:

```json
{"timestamp_ns": 1787976000000000000, "power_w": 6.42}
```

The logger may append while the benchmark runs. Only samples whose timestamps
fall within each row's measured window are summarized. Without this stream the
Power column is `NA`, which is preferable to a fabricated estimate.

## 4. Run or resume

```bash
RUN_DIR="artifacts/pi-benchmarks/$(date -u +%Y%m%dT%H%M%SZ)"

uv run python scripts/run_pi_benchmarks.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.json \
  --output "$RUN_DIR"
```

Model paths/runtimes live in the registry, inference resolution lives in the
matrix, and CPU threads can be overridden without editing either:

```bash
uv run python scripts/run_pi_benchmarks.py \
  --registry configs/pi-model-registry.json \
  --output "$RUN_DIR" \
  --threads 4 \
  --source images
```

### Arducam 8MP IMX219 live-camera run

Install Picamera2 through the Raspberry Pi OS packages and first verify the
camera with the normal `rpicam-*` tools. Then use a separate run directory so
saved-image and camera evidence cannot be mixed:

```bash
CAMERA_RUN_DIR="artifacts/pi-benchmarks/camera-$(date -u +%Y%m%dT%H%M%SZ)"

uv run python scripts/run_pi_benchmarks.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.json \
  --output "$CAMERA_RUN_DIR" \
  --source camera \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --camera-num 0 \
  --camera-pixel-format RGB888 \
  --threads 4
```

The Picamera2 worker captures continuously into the existing one-slot
`LatestFrameBuffer`. If inference is slower than capture, unconsumed frames are
overwritten and the next model call receives the newest frame; stale frames are
never accumulated in an application queue. Picamera2 requests also use
`flush=True` to discard an already queued request before capture.

Camera rows add these raw and summarized measurements:

- capture request start to frame-array availability;
- sensor frame age when the array becomes available;
- sensor frame age at model start;
- sensor timestamp to model finish;
- capture-request start to model finish;
- overwritten-frame count between model calls.

Saved-image rows retain `NA` for camera-only metrics rather than recording a
fictional zero. The IMX219 path is unit-tested with a fake Picamera2 API here but
must still pass the physical Pi/camera test before its numbers are used.

With a power stream:

```bash
uv run python scripts/run_pi_benchmarks.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.json \
  --output "$RUN_DIR" \
  --power-jsonl artifacts/power/power-samples.jsonl
```

The command is resumable: completed `rows/<ID>.json` files are skipped. Retry a
specific failed or intentionally repeated row with `--row R4 --force`. Raw
per-iteration stage/resource samples, artifact identity, environment, cooldown,
throttling state, and errors remain in the run directory.

## 5. Saved outputs

Every run produces:

```text
environment.json
rows/M1.json ... rows/R6.json
tables/model-architecture-benchmark.csv
tables/model-architecture-benchmark.md
tables/runtime-backend-benchmark.csv
tables/runtime-backend-benchmark.md
tables/table-manifest.json
```

Recompile tables without rerunning inference:

```bash
uv run python scripts/compile_pi_tables.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.json \
  --runs "$RUN_DIR" \
  --output "$RUN_DIR/tables"
```

The architecture table uses inference latency percentiles and held-out model
quality. The runtime table reports per-stage means, wall-clock total pipeline
latency percentiles, throughput, system CPU utilization, process peak RSS,
maximum CPU temperature, measured mean power, and—when camera input is selected—
capture/frame-age timing. Pending, failed, or unmeasured values remain `NA` with
an explicit status.
