# Change Log and Engineering Rationale

This file is updated after every major change. It records what changed, why the
change was made, and what evidence currently supports it. It intentionally
separates completed validation from planned Raspberry Pi work.

## 2026-08-28 — Repository audit and scope lock

### What changed

- Confirmed that the workspace was empty and was not a Git checkout.
- Indexed the workspace with the project knowledge-graph tool; the one-node
  result independently confirmed that there was no source code to preserve.
- Locked the laptop-side scope to six deliverables:
  1. a backend-neutral detector interface and functional inference checks;
  2. reproducible benchmark inputs, metrics, and structured logs;
  3. asynchronous sensor replay/fusion with fail-safe range warnings;
  4. dataset taxonomy harmonization and annotation validation;
  5. a Colab GPU training/export workflow for YOLO26n;
  6. a documented handoff that reserves latency, thermal, power, and physical
     sensor claims for the Raspberry Pi 5.

### Engineering rationale

The Pi should receive software whose basic behavior is already deterministic and
tested. Laptop results can establish functional correctness and model accuracy,
but they cannot establish Raspberry Pi latency, throughput, memory, thermal,
power, camera, GPIO, or end-to-end haptic performance.

The implementation will therefore keep hardware integration behind small
interfaces. Fake model, sensor, clock, and haptic implementations will exercise
the complete decision path locally; Pi-specific adapters can replace them later
without changing the safety policy.

### Current evidence

- `ls -la` showed only `.` and `..` before this file was created.
- `git status` reported that the directory is not a Git repository.
- Full knowledge-graph indexing returned one node and zero edges.
- Official Ultralytics documentation currently lists `yolo26n.pt` as trainable,
  validatable, inferable, and exportable, including NCNN export.
- The authorized Colab CLI supports executing local Python scripts on an
  ephemeral T4 runtime and releasing the runtime when the job completes.

### Assumptions in force

- Primary model: YOLO26n detection nano checkpoint.
- Initial laptop development target: macOS with Python 3.9+ (the inspected
  system interpreter is Python 3.9.0).
- V0 is the stock pretrained model; V1 is fine-tuned on harmonized public
  assistive-navigation data; V2 waits for the final cane-mounted camera.
- No Raspberry Pi performance claim will be inferred from laptop or Colab runs.

### Next gate

Create the installable Python package and prove its detector/result contracts
with deterministic tests before adding sensor fusion or GPU training.

## 2026-08-28 — Core package and detector contract

### What changed

- Added an installable `src/`-layout Python package with no mandatory runtime
  dependencies and optional `vision`, `dev`, and `notebook` dependency groups.
- Added immutable backend-neutral contracts for bounding boxes, detections,
  frame timestamps, stage timings, model identity, and original-frame geometry.
- Added a lazy Ultralytics adapter that loads either the PyTorch checkpoint or
  an Ultralytics-compatible exported model without importing ML dependencies in
  core-only workflows.
- Added deterministic left/center/right sector assignment and a CLI command for
  single-image inspection.
- Added repository layout, ignore rules, a laptop smoke configuration, and the
  initial README.

### Engineering rationale

Perception output is normalized before it enters fusion. This prevents fusion
and haptic policy from depending on PyTorch, ONNX, or NCNN-specific result
objects. Captured time and model-call boundaries are retained separately so
stale-frame age is not confused with raw inference duration.

Ultralytics is a lazy optional dependency because dataset and fusion validation
must remain runnable on the 2 GB Pi and on development machines without a full
training environment.

### Verification evidence

- `PYTHONPATH=src python3 -m unittest discover -s tests -v`: 8/8 tests passed.
- Tests cover box geometry and invalid coordinates, confidence validation,
  timing/serialization, equal-third sector mapping, exact boundary behavior,
  immutable assignment, and invalid sector configuration.
- `PYTHONPATH=src python3 -m yolo_pi.cli --help`: CLI parser loaded and advertised
  the `inspect-model` command successfully.
- An initial sector-boundary failure exposed floating-point rounding at exactly
  one third of the image; a small comparison tolerance fixed it and the
  regression is now covered.

### Next gate

Implement a benchmark runner with warm-up exclusion and percentile summaries,
then prove asynchronous sensor fusion never waits for vision to issue a close
range warning.

## 2026-08-28 — Benchmark, replay, fusion, and bounded frame transport

### What changed

- Added a fixed-input benchmark runner with explicit warm-up exclusion,
  repetition tracking, per-input observations, model/stale-frame timing, stage
  timings, p50/p95/p99 summaries, standard deviation, and JSON output.
- Added a range-first fusion engine for timestamped ToF and ultrasonic values,
  optional vision semantics, stale/invalid sensor rejection, and an explicit
  sensor-fault decision.
- Added deterministic haptic-command mapping and a recording sink for laptop
  verification without physical motors.
- Added monotonic JSONL sensor replay and a sample clear → warning → critical
  trace.
- Added a thread-safe one-slot latest-frame buffer. An unconsumed frame is
  overwritten by a newer one and counted, preventing unbounded stale queues.
- Added CLI commands for fixed-input benchmarking and sensor replay plus a
  versioned Pi experiment matrix.

### Engineering rationale

Warm-up measurements must not bias steady-state latency. Percentiles and raw
observations are retained because a mean alone hides stalls. The local report is
explicitly labelled as laptop model-call evidence, not Pi performance.

The fusion policy treats range sensing as the immediate safety signal. Vision
may add a class label for the affected sector but cannot downgrade, delay, or be
required for the warning. No fresh valid range value produces a distinguishable
sensor-fault pattern rather than silently reporting a clear path.

### Verification evidence

- Full standard-library suite: 21/21 tests passed after this change.
- A four-reading CLI replay produced `clear`, `clear`, `warning`, and `critical`
  decisions; the critical decision was triggered at 0.400 m in the center.
- Tests prove warm-ups are excluded, output reports are written, close range
  warns without vision, the nearest contradictory sensor controls severity,
  stale/invalid readings cause sensor fault, vision only enriches a warning,
  haptic commands are recorded, stale frames are overwritten, and replay rejects
  non-monotonic timestamps.

### Next gate

Make public-data harmonization fail closed on taxonomy and label problems, then
create a reproducible Colab training/export artifact.

## 2026-08-28 — Dataset and Colab training/export workflow

### What changed

- Added a strict public-dataset builder that requires an explicit mapping for
  every source class, validates normalized YOLO boxes, converts segmentation
  polygons into boxes, rejects malformed labels, preserves source splits,
  removes exact duplicate images by SHA-256, and emits a traceable build report.
- Added a reviewed example taxonomy for the Mendeley VI Navigation, SOD, and
  Indian Unstructured Road datasets, pinned to named/numbered host versions.
- Added provenance, licence, manual-review, and temporal-leakage acceptance
  gates. Classes intended for later cane-camera collection remain visible even
  if public-data counts are zero.
- Added a self-contained Colab GPU script that trains, validates the best
  checkpoint, exports ONNX and FP16 NCNN, validates both exports, records runtime
  versions/hardware/metrics, and packages artifacts for download.
- Added a 12-cell reader-facing notebook with Goal, Setup, Steps, Checks, and
  Next Steps sections plus the named-session download/cleanup procedure.

### Engineering rationale

A mapping omission is treated as an error rather than guessed. Images whose
existing labels are all intentionally dropped are not converted into false
negative examples. Exact hash de-duplication protects split integrity for exact
copies; the report still requires a human video-level review because adjacent
frames can leak without sharing hashes.

The Colab smoke run defaults to COCO8 for a fast end-to-end GPU proof. This does
not substitute for V1 training: full V1 begins only after raw exports, class
orders, harmonizer output, and visual samples have been reviewed.

### Verification evidence

- Full standard-library suite: 24/24 tests passed.
- Dataset tests cover detection labels, segmentation conversion, dropped
  mappings, unmapped classes, out-of-bounds boxes, packaging, reports, and
  cross-split exact duplicate removal.
- All Python files compile and both dataset configuration files parse as JSON.
- The notebook was generated through `nbformat`, contains 12 cells, and passes
  `nbformat.validate` as a version-4 notebook.
- GPU execution itself is the next gate; static validation is not recorded as a
  successful training run.

### Next gate

Use the authorized Colab CLI to obtain live T4 evidence, download the produced
archive, inspect its summary and exported contents, and verify the session is
stopped.

## 2026-08-29 — Live T4 training/export validation

### What changed

- Executed a one-epoch YOLO26n COCO8 smoke run on an authorized named Colab T4
  session using Python 3.13.15, Torch 2.11.0+cu128, and Ultralytics 8.4.132.
- The first live run trained and validated successfully and exported ONNX, but
  exposed an NCNN conversion failure: `KeyError: 'feats'` after the exporter
  switched the YOLO26 end-to-end head automatically.
- Tested the deployment path against the retained checkpoint and confirmed that
  explicitly exporting NCNN with `end2end=False` selects the required
  one-to-many head. Both FP32 and FP16 NCNN conversion then succeeded.
- Updated both the automated script and notebook to load a fresh model for each
  export and pass the explicit NCNN fallback. Added automatic checkpoint resume
  so a retained session does not wastefully retrain after an export-only failure.
- Executed the corrected script and the 12-cell notebook top-to-bottom, validated
  PyTorch/ONNX/NCNN metrics, downloaded both artifact archives, and terminated
  the Colab session.

### Engineering rationale

NCNN does not support YOLO26's native end-to-end branch and needs the
one-to-many output plus runtime NMS. Relying on the automatic fallback was not
sufficient in the inspected Ultralytics version; an explicit argument is now a
tested invariant in both GPU artifacts.

COCO8 has only four validation images, so the observed metrics are smoke-test
evidence, not a model-quality conclusion. The exported mAP differences are kept
visible rather than presented as equivalent predictions.

### Verification evidence

- Live hardware: Tesla T4 with 14,913 MiB reported by the runtime.
- Corrected script result: PyTorch mAP50-95 0.4783, ONNX 0.4314, FP16 NCNN
  0.4111 on the four-image COCO8 validation split.
- `artifacts/training/yolo-pi-colab-artifacts.zip`: SHA-256
  `8a8f289eb70a6b748820f9e044cb4a8dbd33d7558b8473b5cf18ac89069af2a6`.
- `artifacts/training/yolo-pi-notebook-artifacts.zip`: SHA-256
  `017752f13930a5dd3e0cd8b7fbe1337793d668f1fca84fc327b77c7eb42c7eb1`;
  ZIP integrity check passed.
- The executed notebook contains seven code cells, outputs in every
  evidence-producing cell, and zero error outputs. Colab CLI leaves execution
  counts null, which the validator recognizes explicitly.
- Final `colab sessions` response: no active sessions.

### Next gate

Run real laptop inference, build the pinned V0 export set, and prove that the
same fixed input loads through PyTorch, ONNX Runtime, and NCNN.

## 2026-08-29 — Real laptop inference and V0 export set

### What changed

- Installed locked local `vision`, `edge`, `export`, `notebook`, and `dev`
  dependency groups through `uv.lock`.
- Added the Python NCNN runtime as an explicit optional group after the first
  local exported-model load correctly failed instead of auto-installing while
  offline.
- Downloaded and checksummed the official V0 `yolo26n.pt` checkpoint and fixed
  bus image under ignored artifact paths.
- Added a reproducible evaluation-input manifest and checksum-enforcing fetcher.
- Added a V0 exporter that produces end-to-end ONNX and explicit one-to-many
  FP16 NCNN artifacts, smoke-tests all three formats, and records every artifact
  checksum in `artifacts/models/v0-export-manifest.json`.
- Replaced a misleading Cartesian benchmark configuration with ten explicit
  staged Pi experiments and added the complete physical Pi handoff protocol.
- Made benchmark reports identify the actual PyTorch, ONNX Runtime, NCNN, or
  OpenVINO adapter rather than the generic `ultralytics` wrapper.

### Engineering rationale

The first URL-based prediction took about 32 seconds because model and input
downloads occurred inside the measured call. It is retained only as functional
evidence; warmed local-file runs exclude initialization and downloads.

Exact output equality is not expected across the end-to-end PyTorch/ONNX head
and NCNN's one-to-many+NMS path. The fixed image retained the expected visible
class set in every backend, while the observed detection counts (4/6/5) make the
need for dataset-level export-parity validation explicit.

### Verification evidence

- Stock PyTorch inference detected three people and one bus with valid original
  810×1080 coordinates and left/center/right sectors.
- V0 artifact checksums were written for the 5.3 MB PyTorch checkpoint, 9.4 MB
  ONNX file, and 4.8 MB FP16 NCNN directory.
- Twenty warmed laptop model calls at 416 on the inspected Apple M5 averaged:
  PyTorch 13.064 ms, ONNX Runtime 10.641 ms, and FP16 NCNN 7.644 ms. These are
  laptop-only harness results and are not Pi performance evidence.
- The suite now passes 27/27 tests through the locked virtual environment.

### Next gate

Acquire the pinned public V1 exports if their hosts permit non-interactive
download, build the combined dataset, inspect its report, and run the full V1
Colab fine-tune. If a host requires a user-owned API credential, preserve all
completed work and document the exact acquisition boundary rather than silently
substituting another dataset.

## 2026-08-29 — Top-handle camera viewpoint contract

### What changed

- Added `configs/camera-mount-top-handle.json` as the single camera-viewpoint
  reference for data collection and Pi experiments.
- Added a validated `CameraMountProfile`, a `validate-mount` CLI command, and
  four tests covering the accepted profile, distance boundary, mandatory hand
  occlusion gate, and JSON loading.
- Linked the mount profile from the Pi benchmark matrix and expanded the Pi
  handoff with measurement, intrinsic/extrinsic calibration, grip, occlusion,
  cane-sweep, and retained-evidence checks.
- Updated the model progression to require V2 images from the finalized
  near-top-handle viewpoint.

### Engineering rationale

The user's mounting requirement changes the visual domain: a camera near the
grip sees a higher viewpoint, repeated cane swing, and possible hand/sleeve
occlusion. A model trained only on generic sidewalk imagery cannot prove
performance from that geometry. The mount is therefore a versioned input to the
experiment rather than an informal assembly detail.

The current 40 mm offset and 12-degree downward pitch are provisional starting
targets, not fabricated hardware measurements. The invariant is a forward-facing
camera no more than 80 mm below the grip reference. `hardware_verified` remains
false until the physical cane is measured and the retained occlusion review
passes.

### Verification evidence

- `yolo-pi validate-mount --profile configs/camera-mount-top-handle.json`
  succeeded and printed the normalized profile.
- The benchmark matrix remains valid JSON and references that exact profile.
- The complete suite passes 31/31 tests.

### Next gate

Finish acquiring the signed-in Roboflow exports, build and visually audit the
combined V1 dataset, then fine-tune and export the actual cane V1 checkpoint.

## 2026-08-29 — First two public V1 sources acquired

### What changed

- Downloaded the Mendeley VI Navigation v1 archive from its versioned public API
  and the signed-in SOD v1 YOLO26 archive from Roboflow.
- Retained both raw ZIPs, computed SHA-256 digests, ran full compressed-data
  tests, checked all member paths, and extracted them under ignored `data/raw`
  directories.
- Corrected the Mendeley root path and replaced the assumed SOD class order with
  the exact order in the downloaded `data.yaml`.
- Recorded the SOD ZIP's three identical `data.yaml` entries as a source
  packaging anomaly instead of treating them as three different taxonomies.

### Engineering rationale

The download host's page is useful discovery evidence, but the training pipeline
must follow the immutable archive actually used. Hashes, archive integrity,
member safety, and the embedded class order are therefore acceptance gates
before any label remapping occurs.

The SOD export preserves a clean 70/20/10 split with no export-time augmentation.
That does not itself rule out semantic or temporal leakage, so near-duplicate and
source-sequence checks remain required after harmonization.

### Verification evidence

- Mendeley v1: 579 MB archive, 16,222 members, SHA-256
  `0ec4d17af22257d8c36c1d4d36eb4bf33cf759167a7f950cfcbc78c05ff672ca`;
  full ZIP test passed and unsafe-path count is zero.
- SOD v1: 279,541,334-byte archive, 20,005 members, SHA-256
  `d4ed5a394b650dc21226c69a8aa2e8114d123f0479c1ae65e3efc8c8aaa0c8d4`;
  full ZIP test passed and unsafe-path count is zero.
- The embedded SOD metadata records CC BY 4.0, version 1, and 22 classes.

### Next gate

Acquire and validate the Indian-road v10 export, then run the three-source
harmonizer and leakage audit before committing GPU time to V1 fine-tuning.

## 2026-08-29 — Source-frame lineage leakage gate

### What changed

- Upgraded the harmonizer from exact-byte deduplication to a two-pass candidate
  catalog that also recognizes Roboflow `.rf.<hash>` variants as descendants of
  the same original source frame.
- When one lineage appears across splits, all of its variants are retained only
  in the highest-integrity split using the deterministic priority
  `test > validation > train`.
- Exact duplicates are also processed evaluation-first so a train copy cannot
  displace an identical test or validation copy merely because of iteration
  order.
- Added report fields for detected cross-split lineages and skipped variants,
  plus a regression test with three byte-distinct variants of one source frame.

### Engineering rationale

SHA-256 catches only identical bytes. Roboflow augmentation changes pixels and
therefore hashes while retaining the same scene and labels. Allowing those
variants into both training and evaluation would inflate reported metrics. The
lineage key strips the export-specific `.rf.<hash>` suffix and makes the split
decision before any image is copied.

The policy deliberately favors held-out evidence. Losing a few train variants is
less harmful than contaminating validation or test data. This gate still cannot
identify adjacent video frames with different timestamps, so temporal-neighbor
auditing remains a separate required step.

### Verification evidence

- A synthetic lineage with distinct train, validation, and test bytes is kept
  only in test and reports two skipped cross-split variants.
- The exact-duplicate regression now proves validation wins over train.
- The complete suite passes 32/32 tests.

### Next gate

Complete the Indian-road download, build the combined dataset, and quantify both
lineage conflicts and adjacent-frame leakage in the real source exports.

## 2026-08-29 — Third public source acquired and pinned

### What changed

- Downloaded, retained, checksummed, integrity-tested, path-audited, and
  extracted the Indian Unstructured Road v10 YOLO26 export.
- Replaced the earlier page-derived class assumptions with the archive's exact
  seven-class order and labels.
- Recorded that the export has 7,620 train images, 296 validation images, no
  test split, and three augmented outputs per original training example.
- Disclosed the two identical embedded `data.yaml` entries instead of silently
  hiding the source packaging anomaly.

### Engineering rationale

The 2,208-image project count describes source images, while version 10 expands
to 7,916 exported images. Treating those as independent photographs would
misstate dataset size and leakage risk. The immutable export metadata and
lineage-aware harmonizer are now the authority for training.

### Verification evidence

- Archive size: 4,254,194,773 bytes; 15,836 members.
- SHA-256:
  `9d042cc5ad57a33bd1a965243cff4ba49817e8c8ea4d5f85d186d661fcb25b82`.
- Full compressed-data test passed and unsafe-path count is zero.
- Both 359-byte `data.yaml` entries have identical CRC and content.

### Next gate

Build the three-source V1 dataset, audit its real class/split/leakage report,
then package and launch the full Colab fine-tune.

## 2026-08-29 — Corrected three-source V1 build

### What changed

- Ran the first full three-source harmonization and observed 5,452 images being
  removed by the provisional filename-lineage rule.
- Visually inspected train/test examples sharing `frame_0s` and proved they were
  different streets from different source videos, not transformed copies.
- Corrected the harmonizer so reused pre-export filenames are audit signals only;
  they never cause deletion. Exact SHA-256 duplicates still prefer
  `test > validation > train`.
- Preserved the rejected build under
  `data/processed/cane-v1-rejected-filename-lineage-build` and rebuilt the
  canonical `data/processed/cane-v1` from scratch.

### Engineering rationale

A conservative-looking filter was actually corrupting the dataset by assuming
generic filenames encode video identity. The visual counterexample invalidated
that assumption. Keeping the failed build and documenting the correction makes
the decision auditable; training proceeds only from the corrected build.

The remaining 380 cross-split filename keys cover 6,158 images and are disclosed
as ambiguous. They require visual/perceptual review, not automatic removal.

### Verification evidence

- Corrected canonical build: 26,030 images seen; 25,816 retained; 21,482 train,
  2,819 validation, and 1,515 test.
- 69,848 annotations retained; 2,505 deliberately dropped by explicit class
  mappings; 151 images skipped after all of their labels were dropped; nine
  image files lacked label files.
- Per-source retained images: Mendeley 8,034; SOD 9,920; Indian-road 7,862.
- Twelve target classes have annotations. `door`, `slope`, and
  `overhead_obstacle` remain zero-shot placeholders for later top-handle V2
  collection and must not be claimed as trained capabilities.
- The complete suite passes 32/32 tests after the correction.

### Next gate

Run the processed-dataset integrity and visual near-duplicate audit, package the
validated build, and launch the real V1 Colab fine-tune.

## 2026-08-29 — Full integrity audit and evaluation decontamination

### What changed

- Added a read-only dataset auditor covering image/label parity, YOLO box
  validity, corrupt images, exact SHA-256 duplicates, and cross-split 64-bit
  dHash candidates.
- Audited all 25,816 canonical images and rendered contact sheets for visual
  review. The structural gate had zero errors, but 397 cross-split perceptual
  pairs exposed source-export leakage and adjacent frames.
- Added a deterministic decontaminator that keeps the higher-priority split
  (`test > val > train`) for same-source candidate pairs and never mutates the
  source build.
- Created `data/processed/cane-v1-clean` using hardlinks: 21,270 train, 2,768
  validation, and all 1,515 test images; 263 lower-priority images were removed.
- Re-audited the clean dataset. The only remaining dHash candidate is
  cross-source and visual inspection proves it is a false positive (nighttime
  bollard versus university entrance); the decision is saved in
  `reports/dataset-visual-review.json`.

### Engineering rationale

Exact hashes alone missed small augmentation and temporal changes. A 64-bit
dHash with Hamming distance at most four produced a deliberately conservative
candidate set; visual samples across the distance range confirmed genuine
same-scene leakage. Decontamination removes only lower-priority, same-source
members so held-out evidence is preserved. Cross-source similarity is reviewed
rather than deleted because low-resolution perceptual hashes can collide.

This policy may discard a few difficult-but-distinct training examples, which
is preferable to optimistic validation metrics. The original canonical build
and a complete removal manifest remain available for audit and reversal.

### Verification evidence

- 36/36 automated tests pass, including exact-duplicate detection, atomic audit
  output, and higher-split-priority decontamination.
- Initial full audit: zero missing/orphan/invalid labels, zero unreadable images,
  zero exact duplicates, and 397 perceptual candidates.
- Clean full audit: zero hard errors, zero exact duplicates, and one visually
  rejected cross-source false positive.
- All 51,106 retained image/label files were hardlinked, so source bytes are
  unchanged while disk usage is controlled.

### Next gate

Package `cane-v1-clean`, launch the full Colab fine-tune, export the trained
weights into every requested Pi backend, and record held-out quality metrics.

## 2026-08-29 — Training-scale dataset and resumable Colab transfer

### What changed

- Added an aspect-preserving dataset resize stage capped at 640 pixels with
  normalized YOLO labels left unchanged.
- Produced `data/processed/cane-v1-training-640`: 21,269 train, 2,768 validation,
  and 1,515 test images. One new same-video cross-split candidate exposed by
  resize/re-encoding was removed from train; the single remaining dHash match is
  the already-reviewed cross-source false positive.
- Reduced image bytes from 5,063,472,209 to 1,931,900,118 (38.15%) without
  reducing below the 416-pixel fine-tuning input requirement.
- Packaged the exact training dataset as
  `artifacts/training/cane-v1-training-640.tar` with SHA-256
  `f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91`.
- Added a resumable, bounded-parallel Colab chunk uploader with per-chunk
  retries/timeouts, atomic local state, and active kernel keepalives.

### Engineering rationale

The first 4.8 GB monolithic upload was rejected by Colab's base64 contents API.
A 50 MB chunk transfer then reached 56 parts, but the VM was reclaimed because
file uploads alone did not count as kernel activity. The replacement transfer
both reduces irrelevant source resolution and actively executes a kernel
keepalive. This is a transport correction, not a smaller training subset: every
retained example and annotation is still present.

At 416-pixel training resolution, pre-resizing the long side to 640 preserves
more input detail than the model consumes while avoiding repeated 5 GB remote
transfer and decode overhead. The archival `cane-v1-clean` dataset remains
unchanged and authoritative.

### Verification evidence

- Full post-resize audit: zero structural errors, corrupt images, exact
  cross-split duplicates, or unresolved same-source dHash candidates.
- The resize unit test proves dimensions change while normalized labels remain
  byte-identical.
- The complete suite passes 40/40 tests after the dataset and benchmark work.

### Next gate

Verify the reassembled remote SHA-256, execute the real T4 fine-tune, download
the best checkpoint/metrics, and export all six Pi runtime formats.

## 2026-08-29 — Exact Pi tables and IMX219-ready input abstraction

### What changed

- Replaced the provisional matrix with exact M1–M2 architecture rows and the
  requested R1–R6 PyTorch, ONNX Runtime, OpenVINO, MNN, NCNN, and LiteRT rows at
  640/FP32.
- Added a Raspberry-Pi-only runner that saves environment identity, cooldown,
  raw per-iteration stage/resource samples, errors, artifact hashes, throttling,
  and optional externally measured power before compiling CSV and Markdown.
- Added resumable row execution and deterministic table compilation. Missing,
  failed, or unmeasured values are `NA`; no performance or power estimate can
  silently enter a paper table.
- Added configurable model registry, inference resolution, runtime, CPU threads,
  and input source while preserving the fixed saved-image default.
- Isolated an Arducam 8MP IMX219 Picamera2 adapter. It continuously overwrites a
  one-slot `LatestFrameBuffer`, requests flushed camera frames, and records
  capture, sensor-frame age, camera-to-model-finish, and dropped-frame timing.
- Added the physical Pi commands and saved-output contract in
  `docs/pi-benchmarking.md`; generated pending templates prove both requested
  table shapes before hardware is present.

### Engineering rationale

Backend comparisons require the same trained weights, resolution, precision,
input order, warmup, thermal window, and thread count. The matrix fixes those
variables and changes only runtime. Model quality comes from the immutable held-
out test run and is joined into the architecture table rather than redundantly
or inconsistently measured by each deployment backend.

Camera capture is a different source of latency than saved-image inference.
Keeping it behind an input protocol prevents Picamera2 from contaminating laptop
or image-only installations. A background producer plus a one-slot overwrite
buffer makes latency bounded: a slow model drops old frames and consumes the
latest available frame rather than draining an increasingly stale FIFO.

### Verification evidence

- 40/40 tests pass, including table metrics/status behavior, saved-image cycling,
  and a fake-Picamera2 test proving fresh increasing frame IDs and overwritten
  frame accounting.
- `reports/pi-benchmark-template/` contains both exact tables with M1–M2 and
  R1–R6 rows and explicit pending/NA cells.
- Non-Pi execution is rejected by default, so laptop numbers cannot be
  mislabeled as Pi evidence.

### Next gate

Run the saved-image matrix on the Pi, then run the IMX219 matrix into a separate
directory after tomorrow's physical camera/top-handle verification.

## 2026-08-29 — Singular team data-preprocessing contract

### What changed

- Added `docs/data-preprocessing.md` as the current, non-chronological team
  reference for every raw source, archive hash, anomaly, class mapping, dropped
  label, geometry conversion, split rule, exact/perceptual duplicate policy,
  resize setting, final count, limitation, and reproduction command.
- Linked it from the README and clearly separated the archival
  `cane-v1-clean` dataset from the 640-pixel `cane-v1-training-640` transfer form.
- Documented model-time letterboxing/normalization separately from offline JPEG
  preprocessing and reserved the on-the-fly augmentation section for the exact
  downloaded training `args.yaml`.

### Engineering rationale

`changes.md` explains decisions over time, but teammates need one authoritative
answer to “what data enters the model now?” The new document avoids making them
reconstruct current policy from old experiments, failed assumptions, and log
entries. It also makes negative facts explicit: footpaths and cane annotations
are intentionally dropped, zero-count classes are not trained capabilities, and
dHash is a review candidate rather than proof of semantic duplication.

### Verification evidence

- Every numeric total was checked against the current build, resize,
  decontamination, and final audit JSON reports.
- Every source mapping was checked against `configs/dataset-sources.json`.
- The documented reproduction flags match the current script CLIs.

### Next gate

Update the runtime-augmentation section with the immutable final training
`args.yaml` and held-out metrics after the T4 artifacts are downloaded.

## 2026-08-29 — Held-out Pi runtime fixture and protected live checkpoint

### What changed

- Added `scripts/prepare_pi_runtime_inputs.py`, which deterministically selects
  30 images from the final Cane V1 test split: one image for every populated
  class, up to three empty-label scenes, then evenly spaced remaining examples.
- Generated `artifacts/benchmarks/pi-runtime-inputs/manifest.json` plus its image
  pack and changed the Pi matrix to use it instead of the single demonstration
  bus image. Every repository-relative input path and SHA-256 is recorded.
- Documented fixture creation, transfer, and checksum verification in
  `docs/pi-benchmarking.md`.
- Confirmed the real T4 fine-tune reached epoch 4/30 and downloaded both current
  `last.pt` and `best.pt` after epoch 3. The checkpoint loads as YOLO26n with
  all 15 intended class names and 2,509,650 parameters.

### Engineering rationale

Runtime comparisons can be distorted by one unusually easy image or by changing
input order between backends. A class-covering multi-image fixture makes decode
and NMS work more representative while keeping every R1–R6 row byte-for-byte
comparable. The full 1,515-image test split remains the authority for accuracy;
the 30-image pack exists only for timing and resource measurements.

Checkpoint downloads are deliberately periodic because an ephemeral Colab VM
can be reclaimed. They protect completed training without interrupting the live
process or misrepresenting a partial epoch as the final model.

### Verification evidence

- The production loader resolved all 30 manifest entries and verified their
  hashes successfully.
- The fixture covers all 12 classes with public-data annotations; `door`,
  `slope`, and `overhead_obstacle` correctly remain absent rather than being
  fabricated.
- Epoch-3 checkpoint SHA-256:
  `95189cb0c3f4747eac79d8b99a5796db9b4eb891df1798c0751ba8f895d1dca7`.
  The current best and last files are identical because the latest completed
  epoch is also the best checkpoint at this point in the run.

### Next gate

Allow early stopping or all 30 epochs to finish, download and integrity-check
the final archive, then export the selected `best.pt` to all five non-PyTorch
formats at the exact 640/FP32 Pi comparison setting.

## 2026-08-29 — Semantically valid pretrained-versus-fine-tuned quality gate

### What changed

- Added `scripts/evaluate_model_comparison.py` to evaluate M1 and M2 over the
  same complete Cane V1 test split at the exact 640-pixel architecture-matrix
  resolution.
- Added an explicit COCO-name-to-Cane mapping for the stock checkpoint and a
  secondary target-class NMS after mapping. Unmapped stock predictions are
  dropped, while unsupported target ground truths remain and count as false
  negatives.
- Changed registry generation to require that comparison artifact, require the
  complete test split at 640, populate quality for both M1 and M2, and verify
  checkpoint hashes across training, evaluation, and export manifests.

### Engineering rationale

COCO class ID 2 and Cane class ID 2 do not mean the same thing, so passing the
stock 80-class model directly to a 15-class validator would yield numerically
plausible but semantically invalid results. Name mapping fixes the shared
classes. Retaining unsupported hazards in the target labels makes missing stock
capabilities visible instead of giving the baseline credit on an easier subset.

Quality is evaluated once per architecture, not once per backend. R1–R6 use the
same final weights and inherit M2's held-out result; their purpose is execution
latency and resource comparison.

### Verification evidence

- A full laptop functional run completed on all 1,515 test images for both the
  mapped stock checkpoint and the protected epoch-3 fine-tuned checkpoint.
- The diagnostic result was directionally sensible: the partial fine-tune
  exceeded the mapped stock baseline at 640. It is not the final quality result
  because training was still running when this validator check was executed.
- Registry generation now fails closed if the evaluated baseline, evaluated
  fine-tuned model, training best checkpoint, exported source checkpoint, or
  640/full-test protocol do not match.

### Next gate

Run the same evaluator on the final selected checkpoint on the T4, download its
JSON, and use only that immutable comparison in the production Pi registry.

## 2026-08-29 — Colab reclamation recovery without lost epochs

### What changed

- Detected that the first full-training Colab VM had been reclaimed during
  epoch 7; the server reported no active session, so the closed local process
  handle was not mistaken for a completed run.
- Verified the protected epoch-6 checkpoint contains `epoch=5` (zero-based),
  optimizer state, original 30-epoch arguments, and best fitness 0.35486.
- Created a replacement T4 session, uploaded all 39 dataset chunks with zero
  failures, and reassembled the exact 2,013,204,992-byte archive with SHA-256
  `f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91`.
- Added a checksum-gated resume launcher that installs the protected checkpoint
  as the expected `last.pt`, invokes Ultralytics resume mode, and persists all
  output to a new durable log.

### Engineering rationale

An ephemeral runtime disappearing is not evidence of training completion and
does not justify silently calling a partial checkpoint final. Resuming with the
serialized optimizer, epoch, scheduler, and fitness state preserves the real
training trajectory and avoids repeating completed epochs. The replacement VM
is allowed to proceed only after both dataset and checkpoint hashes match.

### Verification evidence

- Colab server state: the original named session was absent, confirming actual
  reclamation rather than a local observation timeout.
- Resume checkpoint SHA-256:
  `68f8181dfde245a70c7c935cae68cc64f8ae71c73f7e9ed07753a316a10d3c70`.
- Upload result: `complete=39 failed=0`.
- Remote archive verification reproduced the pinned local byte count and hash.

### Next gate

Resume at epoch 7 on the replacement T4, continue periodic checkpoint
protection, and do not accept the model until early stopping or epoch 30 plus
held-out validation and artifact download completes.

## 2026-08-29 — Six-backend exporter preflight and exact MNN FP32 enforcement

### What changed

- Synchronized the locked OpenVINO, MNN, NCNN, ONNX, and LiteRT export/runtime
  dependencies. Added `litert-torch` for Python 3.10+ because Ultralytics'
  exporter requires the converter while core Python 3.9 compatibility remains.
- Re-ran 640-pixel FP32 export/load checks with the protected checkpoint. MNN,
  NCNN, and LiteRT each exported, loaded, and returned four detections on the
  fixed sample. OpenVINO conversion succeeded; after its Apple M5 runtime failed
  internally, the exact same tree was uploaded and validated on Linux/Colab.
- Detected that Ultralytics hard-codes MNN `precision=low` and an unrelated
  thread count. Added a repository-owned adapter that uses native MNN
  `precision=high`, CPU, and the exact matrix thread count.
- Made the Pi runner fail if a requested runtime resolves to a different backend
  and record the resolved backend in every completed row.
- Prevented MNN's converter helper from attempting a package auto-install or
  initializing its telemetry logger during export.

### Engineering rationale

An FP32 label must describe runtime execution, not merely an FP32-looking model
file. The stock MNN loader would have made R4 incomparable to the other requested
rows even though the table still said FP32. An explicit high-precision adapter
and runtime-name assertion turn that silent methodological error into a hard
failure.

Export success and runtime smoke success are distinct gates. The OpenVINO XML
and BIN were produced correctly, but the inspected Apple runtime cannot execute
them. A checksum-bound Linux smoke closes the laptop-side functional gate; the
Pi itself must still run R3 tomorrow before any Pi performance claim is made.

### Verification evidence

- The exact MNN adapter logged `precision=high threads=4`, native MNN printed
  `MNN use high precision`, and the sample produced four detections.
- MNN, NCNN, and LiteRT preflight manifests contain complete file/tree hashes and
  functional-smoke timing metadata.
- Linux OpenVINO evidence at `artifacts/models/openvino-preflight-smoke-linux.json`
  matched tree SHA-256
  `70853d80e183308cc06f14fb2416aac7c91c33aaf68a0d5a9235bef62be7a450`,
  resolved `openvino-ultralytics`, and produced four detections. Its temporary
  CPU Colab session was terminated after download.
- 43/43 tests pass, including the FP32 MNN configuration invariant and explicit
  MNN/LiteRT backend identification.

### Next gate

Let the pinned resumed T4 training finish, then regenerate and revalidate all
production exports from the immutable final `best.pt`.

## 2026-08-29 — Exact trainer-version resume and deterministic recovery check

### What changed

- Detected that the replacement VM selected Ultralytics 8.4.133 from a broad
  `>=8.4,<9` install while epochs 1–6 used 8.4.132.
- Stopped that mixed-version process after one completed resumed epoch, pinned
  the training script and both notebook sources to `ultralytics==8.4.132`, and
  restarted from the original protected epoch-6 checkpoint on the same verified
  VM/dataset.
- Separated the accepted pinned log from the rejected mixed-version log.

### Engineering rationale

A patch-version change during a resumed experiment is an avoidable confounder.
Even when metrics look identical, a broad live install makes exact reproduction
depend on PyPI state. Pinning and replaying the single affected epoch costs only
minutes and keeps the accepted training trajectory version-consistent.

### Verification evidence

- The clean log identifies Ultralytics 8.4.132, Torch 2.11.0+cu128, Python
  3.13.15, and Tesla T4 before entering `7/30`.
- Both the rejected 8.4.133 and accepted 8.4.132 epoch-7 runs produced fitness
  0.35915 and the identical model-state tensor SHA-256
  `b0238689a581062d789b22e6db8e6ed0cde793c998f15badb7f23afebc0d2d78`.
  Their whole checkpoint-file hashes differ because serialized metadata differs.
- Accepted epoch 8 improved to precision 0.656, recall 0.476, mAP50 0.526, and
  mAP50-95 0.369; its best checkpoint is protected locally.

### Next gate

Continue the pinned run to early stopping or epoch 30, with periodic accepted
checkpoint downloads and no further environment changes.

## 2026-08-29 — Usage-window-independent checkpoint guard

### What changed

- Immediately downloaded the accepted pinned `best.pt`, `last.pt`, training log,
  `results.csv`, and `args.yaml` before the current Codex usage window expires.
- Added `scripts/colab_checkpoint_guard.py`, a local background guard that polls
  Colab without model/API tokens, validates the 15-class taxonomy, atomically
  replaces `latest-best.pt`/`latest-last.pt`, and creates hard-linked snapshots
  named by completed epoch and checkpoint hash.
- The guard also refreshes logs/sidecars and watches for the final Colab ZIP. If
  it appears, the guard downloads it, runs a full ZIP integrity test, requires
  the validation summary plus best/last weights, downloads the standalone
  summary, and exits successfully.
- Repeated remote failures stop only after ten intervals and leave the latest
  verified local recovery checkpoint intact.

### Engineering rationale

The Colab VM and this Codex conversation have independent lifetimes. Saving only
inside `/content` is unsafe if either the VM is reclaimed or agent usage ends.
A detached local process is preferable to relying on another model turn: it
uses the already-authorized Colab CLI, consumes no Codex tokens, and persists
completed epochs directly to the workspace.

### Verification evidence

- Pre-guard recovery files are locally present at completed epoch 9. Both best
  and last include optimizer state and SHA-256
  `3aa69ff1a72b1d9fc7266820a270f108fe7ec41a116c5fff2f31ce4b0e3bd554`.
- The local live log, results CSV, and exact args YAML were downloaded in the
  same durability pass.
- The interactive guard check then captured completed epoch 10. Both validated
  best and resumable last checkpoints include optimizer state, preserve the
  exact 15-class taxonomy, and have SHA-256
  `6ca0776181d5d5bfb81cabcf8c04a12fc133bacc1abdd1bb0523a6a4617e8d2c`.
- Detached guard PID `20940` was confirmed alive with parent PID 1 after the
  launching shell exited. Its state is `guarding`, its first cycle has no
  errors, and all latest files plus immutable epoch/hash snapshots are present.

### Next gate

Let training continue. The detached guard now polls every 120 seconds and will
stop successfully only after the final ZIP and validation summary have been
downloaded and verified, or conservatively after ten consecutive remote
failures while retaining its latest local recovery snapshot.

## 2026-08-30 — Guarded VM reclamation and epoch-15 recovery

### What changed

- Corrected the checkpoint guard's liveness counter so a previously saved local
  checkpoint cannot mask a failed current Colab poll. `--once` now succeeds
  only when the current cycle refreshes `last.pt`, and the state file records
  the actual consecutive-failure count.
- Replaced the original guard daemon with the corrected implementation without
  touching the remote training process.
- The Colab VM was later reclaimed. The corrected guard observed ten consecutive
  missing-remote-file polls, stopped conservatively, and retained all locally
  verified checkpoints and sidecars.
- Promoted the protected epoch-15 `last.pt` as the next resume source and created
  a fresh T4 session for an exact optimizer-state resume.

### Engineering rationale

Remote-session failure and model-training completion are different terminal
states. A stale local checkpoint proves recoverability, but it must not be
treated as proof that a current poll succeeded. Counting only a freshly
downloaded `last.pt` makes the guard's terminal status meaningful while keeping
the recovery artifact safe.

The validation fitness was still improving at epoch 15, so the reclaimed VM is
not used as an implicit early-stopping decision. Training resumes from the exact
optimizer-bearing checkpoint instead.

### Verification evidence

- The final successful guard cycle captured completed epoch 15 with all 15 class
  names, optimizer state, best fitness `0.41313`, and SHA-256
  `cfeb7c9a252b2344c713a9721c06ab6d5b5439ce4f8b98080413dd754d578274`.
- Versioned best/last snapshots exist for the completed epochs retained by the
  guard, and the latest log, `results.csv`, and `args.yaml` remain local.
- The terminal guard state is
  `stopped_after_repeated_failures_with_latest_local_snapshot` with
  `consecutive_failures=10`; an authoritative Colab session listing confirmed
  that no original session remained.
- A replacement T4 session named `cane-v1-resume-2` reached READY before the
  checksummed dataset transfer began.
- Added regression tests for a successful current-cycle `last.pt` refresh and
  for stopping after repeated failed current polls even when an older snapshot
  exists; the complete suite passes 48/48.
- The replacement transfer completed all 39 chunks with zero failures. Remote
  reassembly produced exactly 2,013,204,992 bytes with SHA-256
  `f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91`.
- The resume launcher independently verified the epoch-15 checkpoint SHA-256,
  installed it as `last.pt`, and started detached remote PID `11905`. Corrected
  local guard PID `12945` now monitors `cane-v1-resume-2`.
- Removed the guard's last post-archive network dependency: after the final ZIP
  passes `ZipFile.testzip()` and contains best/last weights plus the validation
  summary, the summary is atomically extracted from that already-verified local
  archive. The guard can therefore finish even if the VM disappears immediately
  after the ZIP download.
- The guard also atomically promotes final `best.pt` and `last.pt` from the
  verified archive into `checkpoint-guard/final/`, validates their 15-class
  taxonomy, and requires final `best.pt` to match the summary SHA-256. This makes
  the finished model directly usable even if no further Codex turn occurs.
- Final checkpoint inspection now recovers the completed epoch from embedded
  `train_results` when Ultralytics has stripped a finished checkpoint and set
  its top-level epoch to `-1`.
- Added end-to-end final-archive promotion and stripped-checkpoint coverage; the
  complete suite now passes 54/54.
- The replacement log identifies the exact pinned Ultralytics 8.4.132/Tesla T4
  environment and entered epoch `16/30`, proving optimizer-state continuation
  rather than a fresh epoch-1 run.
- Added `scripts/consolidate_training_history.py` to reconstruct a contiguous
  accepted metrics CSV from immutable checkpoints. It ignores elapsed-time
  differences across VMs, accepts only metric-identical replayed epochs, rejects
  gaps or divergent duplicates, records every checkpoint hash, and validates a
  shared class taxonomy. The provisional epoch 1–15 consolidation passes with
  one identical epoch-7 replay removed.

### Next gate

Confirm the remote log enters epoch 16, then continue through early stopping or
epoch 30. The restarted independent local guard will retain checkpoints and
download the final validated artifact archive when it appears.

## 2026-08-29 — Kaggle notebook for Cane V1 fine-tune resume only

### What changed

- Confirmed the Colab T4 assign path is unavailable (HTTP 503,
  `sub:0 / subTier:0 / outcome:2`). No new Colab training session was started.
- Added `notebooks/cane_v1_kaggle_resume.ipynb` as a Kaggle-hosted **resume**
  of the existing Cane V1 fine-tune from the protected epoch-15 checkpoint.
- Added `docs/kaggle-resume.md` stating that only the training host changes.
  Dataset, fusion, Pi benchmarking, camera mount, and V0 remain the same.
- The notebook pins `ultralytics==8.4.132`, checksums the 640-pixel archive and
  epoch-15 `last.pt`, rewrites stored `/content` paths, calls `train(resume=True)`,
  validates at imgsz 416, and exports ONNX plus `end2end=False` NCNN.

### Engineering rationale

The experiment is still the pinned 30-epoch YOLO26n run that reached epoch 15
on Colab. Moving the remaining epochs to Kaggle is a host failover, not a new
model version. Hash gates prevent a fresh `yolo26n.pt` train or a wrong dataset
from being mistaken for that resume.

### Verification evidence

- `uv run python scripts/build_kaggle_resume_notebook.py` wrote a version-4
  notebook that passes `nbformat.validate`.
- All nine code cells parse as valid Python (the first is a `%pip` magic).
- Resume checkpoint SHA-256 remains
  `cfeb7c9a252b2344c713a9721c06ab6d5b5439ce4f8b98080413dd754d578274`.
- Dataset archive SHA-256 remains
  `f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91`.

### Next gate

Upload the notebook plus the two hashed artifacts to Kaggle with GPU and
internet enabled, confirm the log starts at 16/30, and download
`yolo-pi-kaggle-artifacts.zip` when early stopping or epoch 30 completes.

## 2026-08-30 — Kaggle fine-tune resume failed; return to Colab

### What changed

- Marked the Kaggle resume path as failed in `docs/kaggle-resume.md`, the
  README, `docs/colab-workflow.md`, and `docs/data-preprocessing.md`.
- No Kaggle training epoch ran. The live Cane V1 checkpoint remains the
  protected Colab epoch-15 `last.pt` /
  `cfeb7c9a252b2344c713a9721c06ab6d5b5439ce4f8b98080413dd754d578274`.
- Recorded the attach/extract failure: report JSON said 21,269 / 2,768 / 1,515
  while Kaggle JPEGs counted 20,580 / 4,432 / 2,504. Ignoring macOS
  `._sod-v1__*.jpg` sidecars still left real images at 13,707 / 2,468 / 1,515.

### Engineering rationale

A renamed folder on Kaggle is not the pinned training set. Auto-extracted
AppleDouble junk and a truncated train/val tree would have been a different
dataset. Failing closed and returning to Colab preserves the epoch-15 resume.

### Verification evidence

- Kaggle diagnostic: `report {'test': 1515, 'train': 21269, 'val': 2768}` versus
  `disk train 20580 / val 4432 / test 2504`.
- After dropping `._*` files: real `13707 / 2468 / 1515` plus junk
  `6876 / 1968 / 989` (the SOD AppleDouble counts).
- Local tar inspection: 9,833 unlabeled `._sod-v1__*.jpg` members; labels still
  21,269 / 2,768 / 1,515.

### Next gate

Resume on Colab from epoch-15 `latest-last.pt` with the checksum-gated launcher,
the 39-chunk dataset, and Ultralytics 8.4.132. Do not use the Kaggle extract.

## 2026-08-30 — Epoch-22 recovery and Lightning AI training handoff

### What changed

- Recovered the second Colab continuation through completed epoch 22. The
  optimizer-bearing `last.pt` has best fitness `0.43616`, the exact 15-class
  taxonomy, and SHA-256
  `0da75726f7a058e63d7b030601447ddeb4bd98b420d1387482761aa5799a0eea`.
- Replaced the hard-coded Colab resume source with
  `artifacts/training/cane-v1-resume-manifest.json`, so a replacement host can
  verify an explicit checkpoint path, completed epoch, and checksum.
- Added `scripts/colab_chunk_probe.py` and hardened
  `scripts/colab_chunk_upload.py` to verify remote byte count and SHA-256 before
  retrying a client timeout. A third Colab assignment proved that six remote
  chunks were complete and checksum-correct even though two calls had timed
  out locally.
- Stopped the third transfer when the user selected Lightning AI as the training
  host. No training process was launched there. The Colab CLI no longer had the
  local session name; it still displayed an unresolved remote assignment, which
  could not be explicitly released by either known name and should expire on
  the provider side.
- Added `scripts/lightning_train_resume.py`, a persistent-Studio runner that:
  - verifies the 2,013,204,992-byte dataset archive and epoch-22 checkpoint;
  - requires CUDA, pinned Ultralytics 8.4.132, optimizer state, and the exact
    15-class taxonomy;
  - resumes the existing 30-epoch experiment instead of starting a new model;
  - chooses the highest valid local snapshot on every rerun;
  - copies and independently validates an immutable resumable checkpoint after
    every completed epoch;
  - performs final validation on both `val` and held-out `test` at image size
    416; and
  - packages best/last weights, the latest resumable checkpoint, metrics,
    arguments, hashes, and a completion receipt into one download ZIP.
- Added `docs/lightning-ai-training.md` with the browser and CLI upload paths,
  CPU-first setup, GPU switch, integrity gates, exact start/status commands,
  required `23/30` resume proof, interruption recovery, OOM fallback, completion
  checks, and local return path.
- Assembled `lightning-ai-upload/` as the only folder the user needs to upload.
  Its large archive and checkpoint are local hard links, so the folder behaves
  as normal uploadable files without duplicating 1.9 GB on disk.
- Added `scripts/prepare_lightning_handoff.py` so a fresh clone can reconstruct
  the two Git-ignored large inputs from canonical local artifacts. The builder
  checks fixed hashes, archive size, completed epoch, optimizer state, taxonomy,
  and scaffold parity; it is idempotent, writes an ignored package receipt, and
  refuses to replace a mismatched destination. The handoff documentation now
  tells teammates to treat same-filesystem hard links as immutable.
- Hardened `lightning_train_resume.py` so it validates a checkpoint before
  replacing any same-epoch snapshot, writes an epoch-22 recovery snapshot at
  startup, and atomically refreshes a CRC-checked
  `cane-v1-latest-recovery.zip` after every completed epoch. This directly
  preserves a download-ready optimizer checkpoint if a usage limit ends before
  final validation and packaging.
- Added `artifacts/training/lightning-inbox/README.md` to define the exact local
  return filename for the completed Lightning deliverable.
- Added `scripts/import_lightning_deliverable.py` as the local acceptance gate.
  It validates ZIP CRCs and paths, required contents, summary/model hashes,
  target epochs, image size, val/test metric completeness, and the exact
  15-class taxonomy before atomically promoting the returned files.
- Made `latest-resumable.pt` and its atomic receipt mandatory in the returned
  ZIP. The Lightning summary now binds its hash, completed epoch, optimizer
  state, and taxonomy; the importer rejects a final package that contains only
  stripped deployment weights without a recoverable training checkpoint.
- Added `scripts/finalize_lightning_return.py` as a resumable post-training
  state machine. One command now performs the accepted ZIP import, normalized
  training-summary generation, contiguous multi-VM history consolidation,
  640/FP32 backend exports, full 1,515-image M1/M2 held-out comparison, Linux
  OpenVINO smoke-bundle preparation, and production registry generation. It
  records every stage and refuses to build the registry while a runtime smoke
  remains incomplete.
- Added `docs/completion-readiness.md` as a requirement-by-requirement evidence
  audit. It records completed code/schema work, exact commands and expected
  counts, and the two honest remaining evidence classes: the returned Lightning
  model and physical Pi/IMX219 measurements.

### Engineering rationale

Lightning Studio persists the home directory and installed environment across
machine switches, so it is a better fit for the remaining eight epochs than
repeatedly rebuilding short-lived Colab VMs. Persistence alone is not enough:
the compute process can still stop. Immutable, hash-validated per-epoch
snapshots make the exact same start command safe after a Studio restart or
preemption.

The upload package intentionally excludes the full repository. Pi benchmark
code, old provider attempts, historical checkpoints, exported formats, and
caches are irrelevant to GPU training and would increase transfer time and the
chance of selecting the wrong artifact.

### Verification evidence

- Manifest, source snapshot, and prepared upload checkpoint all agree on
  completed epoch 22 and SHA-256
  `0da75726f7a058e63d7b030601447ddeb4bd98b420d1387482761aa5799a0eea`.
- Prepared upload archive SHA-256 is
  `f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91`.
- Filesystem inode checks prove the prepared large inputs are hard links to the
  immutable sources, not divergent copies.
- Both Lightning shell scripts pass `bash -n`; the packaged runner compiles and
  exposes the expected CLI.
- Eighteen dedicated Lightning handoff/recovery/import/finalization tests pass,
  including idempotent materialization and mismatch refusal, newest
  snapshot selection, atomic recovery-ZIP contents, pre-copy rejection of an
  epoch mismatch without clobbering a valid snapshot, rejection of missing
  optimizer state or wrong taxonomy,
  safe archive extraction, completion-receipt hash validation, safe ZIP import,
  and rejection of path traversal or mismatched best-weight hashes.
- The complete automated suite passes 72/72, including the four new
  finalization-state tests and explicit rejection of a return package without
  its latest optimizer-bearing recovery checkpoint. It also covers the earlier
  Lightning-summary layout by requiring its independent atomic recovery receipt
  to match the model hash and epoch. The executed training/export
  notebook also validates with seven code cells, no error outputs, and the
  expected Colab-CLI-null execution-count mode.
- A fresh table compilation against an empty run directory produced exactly two
  architecture rows and six runtime rows. The CSV headers contain all requested
  latency, resource, quality, capture, and power columns; unmeasured values are
  blank/`NA`, never estimates.
- The fixed saved-image loader validates all 30 manifest entries, the benchmark
  CLI exposes source/thread/camera controls, and mount validation confirms the
  provisional camera at 40 mm below the top handle with
  `hardware_verified=false` pending tomorrow's physical test.

### Next gate

Upload only `lightning-ai-upload/` to one persistent Lightning Studio, verify
both hashes on CPU, switch to one CUDA GPU, run `bash start-training.sh`, and
confirm the first new epoch is `23/30`. After
`lightning-output/LIGHTNING_COMPLETE.json` appears, download
`cane-v1-lightning-deliverable.zip` into
`artifacts/training/lightning-inbox/`. Local export, held-out model comparison,
production registry generation, and final Pi handoff validation then continue.

## 2026-08-30 — Git repository initialization and GitHub origin

### What changed

- Extended `.gitignore` before repository initialization so the hard-linked
  1.9 GB Lightning dataset archive, 20 MB recovery checkpoint, future Lightning
  output, and timestamped physical Pi run directories cannot enter ordinary Git
  history accidentally.
- Initialized the workspace as a Git repository on branch `main`.
- Added the requested remote:
  `https://github.com/ctxnn/capstone-pi-benchmarking.git` as `origin` for fetch
  and push.
- Did not stage, commit, or push. The first commit remains gated on the returned
  model/final artifact audit unless the user explicitly asks to publish sooner.

### Engineering rationale

Initializing before checking ignore coverage could have made a later broad
`git add` capture tens of gigabytes of reproducible datasets, training outputs,
or raw Pi observations. Keeping those artifacts outside Git history while
tracking their manifests, scripts, hashes, and documentation preserves a usable
repository and a reproducible experiment record.

### Verification evidence

- `git remote -v` reports the exact requested URL for both fetch and push.
- `git status` reports an empty repository on `main` with no commits.
- `git check-ignore -v` confirms the dataset tar, epoch-22 checkpoint, raw and
  processed data, Lightning hard links, and virtual environment are ignored.
- The largest non-ignored candidate is `uv.lock` at about 1.2 MB; no candidate
  file approaches GitHub's 100 MB object limit.

### Next gate

Complete the Lightning return, final exports, production registry, and physical
handoff audit. Then review the exact first-commit file set and push only with
explicit user authorization.
