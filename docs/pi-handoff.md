# Raspberry Pi 5 benchmark handoff

This is the boundary between verified laptop preparation and physical-device
evidence. Do not populate paper performance tables until these steps run on the
actual Raspberry Pi 5 2 GB.

## Hardware and environment record

Record before every experiment series:

- Pi model, RAM, board revision, OS image and kernel;
- storage medium and free space;
- official/other active cooler, enclosure and room temperature;
- power supply and any inline power meter;
- camera model, mode, exposure and mount geometry;
- every ToF, ultrasonic, IMU and haptic component plus bus/address;
- Python, Ultralytics, NCNN, ONNX Runtime and OpenVINO versions;
- enabled background services, swap configuration and CPU governor.

Use the same cooler, enclosure, supply, camera configuration, sensor wiring, and
room conditions for compared rows. Report changes rather than hiding them.

## Top-handle camera contract

The camera belongs near the top handle, not near the cane tip. Start from
`configs/camera-mount-top-handle.json`: the provisional optical center is 40 mm
below the lower edge of the normal grip and must remain within the declared
80 mm top-handle zone. The lens faces forward with a provisional 12-degree
downward pitch. These are setup targets, not validated physical measurements.

Before collecting V2 images or running A9:

1. mark the user's repeatable grip reference on the handle;
2. measure and record the optical-center offset from that reference;
3. measure pitch and roll with the cane in its documented neutral pose;
4. calibrate camera intrinsics at the exact capture mode and mount extrinsics;
5. record at least 60 seconds of normal grip and cane sweep, confirming that the
   hand, sleeve, cable and enclosure do not occlude the walking corridor;
6. verify that near-ground obstacles and head/torso-height obstacles are both
   represented during the sweep;
7. set `hardware_verified` true only after the measurements and occlusion review
   are retained with the experiment evidence.

Public-data V1 is useful before the physical mount exists, but V2 must include
images captured from this top-handle viewpoint because height, downward pitch,
cane sweep and hand occlusion create a domain shift that generic sidewalk data
cannot reproduce exactly.

## Installation and functional gate

```bash
uv sync --extra vision --extra edge --extra export
uv run python scripts/fetch_evaluation_inputs.py
uv run pytest
uv run yolo-pi validate-mount \
  --profile configs/camera-mount-top-handle.json
uv run yolo-pi replay-sensors \
  --input data/sample_sensor_replay.jsonl \
  --output artifacts/benchmarks/pi-replay-decisions.jsonl
```

Copy/download the versioned model artifacts whose hashes match the export
manifest. Run `inspect-model` on every backend and confirm expected classes,
coordinates, confidence ranges, and sector mapping before measuring speed.

## Staged experiment ladder

Use `configs/pi-benchmark-matrix.json` and the executable procedure in
`docs/pi-benchmarking.md`. M1–M2 compare pretrained and fine-tuned architectures;
R1–R6 hold the Cane V1 weights and 640/FP32 configuration constant while changing
only PyTorch, ONNX Runtime, OpenVINO, MNN, NCNN, or LiteRT. A4–A7 are the later
NCNN precision/resolution optimization ladder after the fair runtime comparison.
Run the fixed saved-image matrix first. Run the IMX219 camera matrix into a
different directory only after the top-handle mount and camera mode are recorded;
its capture/frame-age metrics answer a different question and must not replace
the fixed-input backend comparison.

For each row:

1. Start from the same idle thermal window and record initial temperature.
2. Use the fixed evaluation manifest and reject checksum drift.
3. Perform 20 unrecorded warm-up calls.
4. Run the matrix's 100 measured repetitions over the fixed manifest.
5. Retain every raw observation; compute mean, standard deviation, p50, p95 and
   p99 only after collection.
6. Record preprocessing, inference, postprocessing and model-call latency
   separately.
7. Record peak RSS, available RAM, swap, CPU utilization, temperature,
   throttling flags, dropped frames and measured power.
8. Repeat a 30-minute soak for the selected configuration and retain the whole
   time series rather than only the final summary.

Laptop reports may debug the harness but must not be inserted into Pi rows.

## End-to-end timing

Use a monotonic clock for every event:

```text
camera frame timestamp
  → frame available to application
  → preprocessing start/end
  → inference start/end
  → postprocessing end
  → fusion decision
  → haptic command issued
  → physical actuator response (when measurement equipment is available)
```

Report at least:

```text
model latency = inference end - inference start
software e2e = haptic command - camera frame timestamp
physical e2e = measured actuator response - camera exposure/event
```

Software e2e is not physical e2e. State clearly when the actuator response was
not instrumented.

## Sensor and safety tests

- close obstacle with vision disabled: range warning must still occur;
- empty/blocked camera with valid ranges: range warning remains available;
- no detections with a close range: issue an unknown-obstacle warning;
- one stale or disconnected sensor: health state and remaining coverage are
  visible;
- all range sensors stale: emit the sensor-fault pattern, never `clear`;
- contradictory sensors: nearest fresh credible range controls severity;
- delayed vision result: stale semantics are ignored;
- fast camera producer: newest frame replaces unconsumed stale frames;
- I²C/serial exception and haptic-driver exception: process remains diagnosable
  and safety degradation is explicit;
- bright sun, low light, motion blur, cane swing, close object, stairs/curb,
  overhead hazard, indoor and crowded scenes are represented in field trials.

Warning-distance and response-latency acceptance thresholds must be chosen from
the intended walking-speed and user-study safety requirements before inspecting
final results. Do not select a threshold after seeing the benchmark table.

## Paper evidence package

Retain, checksum and cite:

- source and processed dataset manifests, licences and build report;
- model checkpoint and every exported runtime artifact;
- training configuration, package versions, seed and full metrics;
- raw per-frame benchmark logs and summarized tables;
- temperature, throttling, memory, CPU and power time series;
- camera/sensor/haptic configuration and calibration records;
- failed runs and exclusions with reasons;
- photographs/diagrams of the tested physical configuration.

The paper should compare V0, V1 and eventually V2 on the same held-out hazard set
and the same Pi protocol. COCO8 smoke metrics prove only that the workflow ran.
