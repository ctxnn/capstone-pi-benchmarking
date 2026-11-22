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

## 2026-11-22 — Input-size tuning for the Pi

### What changed

- Measured held-out quality for Cane V1 at 640, 512, 416, 352 and 320, and for
  OpenVINO FP32/INT8, NCNN FP16 and LiteRT INT8 exports at 416 and 320.
- Measured those exports on the Pi with the new `scripts/quick_speed_test.py`
  and on the live IMX219 with the reworked `scripts/live_camera_demo.py`.
- Kept the OpenVINO FP32 416 and 320 exports in
  `artifacts/models/cane-v1-fast/` with an export manifest that binds them to
  the accepted `best.pt`, their tree hashes and their own held-out quality.
- Added matrix rows R7 (416) and R8 (320), and taught
  `build_pi_model_registry.py` to register them through
  `--input-size-manifest` with source, tree and resolution checks.
- `live_camera_demo.py` now defaults to the 416 export, reads its input size
  from model metadata, and has a headless `--no-display` mode that reports
  inference, frame age and camera-to-result latency.
- Documented the IMX219 overlay and ribbon failure modes, the spinning
  `avahi-daemon`, and the clock-cap procedure for a weak supply in the runbook.

### Engineering rationale

The model was trained at 416 but deployed at 640, which did 2.4x the work and
scored lower: mAP50 52.72 at 640 against 62.13 for the OpenVINO 416 export on
the same 1,515-image test split. Restoring the training resolution is the
largest available speed-up and also the accuracy fix. INT8 and NCNN FP16 are the
usual next step, but on this Pi they measured 5.3x and 4.5x slower than FP32,
so they are rejected on evidence rather than kept as options.

### Verification evidence

- Live IMX219, same session, OpenVINO FP32, 2 threads, 1.5 GHz cap: 416
  measured 61.5 ms inference, 89.6 ms camera-to-result and 15.20 FPS; 640
  measured 149.2 ms, 196.5 ms and 6.43 FPS.
- Fixed images under the same cap: 416 at 99.2 ms and 320 at 62.6 ms against
  242.9 ms at 640; OpenVINO INT8 at 530.4 ms and NCNN FP16 at 446.5 ms.
- R7 and R8 completed through the formal runner in a harness-only
  `--allow-non-pi` dry run, and the registry builder accepted the manifest
  against the real training summary and export manifest.
- The full test suite passes.

### Next gate

The supply reads 4.80 V and the Pi rebooted under sustained four-core load, so
these are quick comparisons at reduced clock. Run the formal R3/R7/R8 rows for
both sources, then repeat them at four threads and full clock once the supply is
replaced.
