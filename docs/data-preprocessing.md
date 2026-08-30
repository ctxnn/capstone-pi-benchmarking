# Cane V1 data preprocessing and dataset contract

This is the single team-facing explanation of how the Cane V1 training data is
created. It is not a chronological log. Update this file whenever the accepted
dataset, taxonomy, preprocessing policy, audit rule, or training input changes.

The short version is:

```text
three immutable public exports
  -> explicit class harmonization
  -> normalized YOLO detection labels
  -> exact-duplicate removal
  -> visual/perceptual cross-split decontamination
  -> aspect-preserving 640-pixel transfer resize
  -> second decontamination and full audit
  -> YOLO26n training at imgsz=416
```

The current training authority is:

```text
data/processed/cane-v1-training-640/data.yaml
```

The first 22 epochs ran across guarded Colab T4 sessions. The current
**fine-tune host** is one persistent Lightning AI Studio, resuming the same run
from the protected epoch-22 checkpoint. This is a host change only: model,
dataset, split membership, target 30 epochs, optimizer state, and training
arguments remain one experiment. A Kaggle failover was attempted on
2026-08-29/30 and rejected before any epoch ran because its extracted dataset
did not match this contract. Follow `docs/lightning-ai-training.md` for the
active handoff and `docs/kaggle-resume.md` only for the failed-attempt record.

The higher-resolution archival authority is:

```text
data/processed/cane-v1-clean/data.yaml
```

## 1. Purpose and boundaries

Cane V1 is a public-data fine-tune for generic pedestrian/navigation hazards.
It is meant to improve over stock COCO weights before the physical cane capture
campaign exists. It is not the final camera-domain dataset.

The finalized camera will be an Arducam 8MP IMX219 mounted near the top handle,
with a forward view and downward pitch. Images from that viewpoint will become
Cane V2 because camera height, cane swing, hand/sleeve occlusion, lens geometry,
and exposure differ from public handheld/roadside data.

Do not claim that V1 has learned a class merely because the class name exists in
`data.yaml`. `door`, `slope`, and `overhead_obstacle` currently have zero public
annotations and are placeholders for V2.

## 2. Immutable raw sources

All three acquired exports record CC BY 4.0. Preserve their original archives,
embedded README/licence files, host metadata, and the hashes below with paper
artifacts.

| Source | Pinned export | Archive SHA-256 | Published/source size | Export details |
|---|---|---|---:|---|
| Mendeley Navigation Assistance for Visually Impaired People | v1, 24 Mar 2025 | `0ec4d17af22257d8c36c1d4d36eb4bf33cf759167a7f950cfcbc78c05ff672ca` | 8,114 images | 22 classes; pathway/roadside video frames |
| SOD Sidewalk Obstacle Detection | Roboflow v1 YOLO26 | `d4ed5a394b650dc21226c69a8aa2e8114d123f0479c1ae65e3efc8c8aaa0c8d4` | 10,000 images | 7,000 train, 2,000 val, 1,000 test; no export augmentation |
| Indian Unstructured Road Obstacle Detection | Roboflow v10 YOLO26 | `9d042cc5ad57a33bd1a965243cff4ba49817e8c8ea4d5f85d186d661fcb25b82` | 2,208 originals | 7,916 exported images; 7,620 train, 296 val; three augmented outputs per training original |

Archive integrity checks were not limited to file existence:

- every ZIP passed a full compressed-data test;
- every archive member was checked for unsafe extraction paths;
- class order came from the downloaded `data.yaml`, not a web-page guess;
- the SOD archive's three repeated `data.yaml` members were verified as
  byte-identical;
- the Indian export's two repeated `data.yaml` members were also byte-identical.

The raw files are never edited in place.

## 3. Target taxonomy

The ordered 15-class target taxonomy is:

| ID | Target class | Intended meaning | Current public-data status |
|---:|---|---|---|
| 0 | `person` | pedestrian/person | trained |
| 1 | `vehicle` | car, bus, truck, train, van, rickshaw and related vehicles | trained |
| 2 | `bicycle_motorcycle` | bicycles, motorcycles and two-wheelers | trained |
| 3 | `pole` | utility/light poles and structural pillars | trained |
| 4 | `tree` | tree/trunk obstacles | trained |
| 5 | `stairs` | stair hazards | trained |
| 6 | `curb` | curb boundary/hazard | trained, SOD only |
| 7 | `barrier` | warning columns and roadblocks | trained, SOD only |
| 8 | `cone` | parking/traffic cones | trained, Mendeley only |
| 9 | `dog` | dog/animal obstacle | trained |
| 10 | `signboard` | signs and signal heads | trained |
| 11 | `door` | door/doorway | zero public examples; V2 placeholder |
| 12 | `slope` | ramp/slope transition | zero public examples; V2 placeholder |
| 13 | `overhead_obstacle` | head/torso-height collision hazard | zero public examples; V2 placeholder |
| 14 | `generic_obstacle` | bins, benches, carts, hydrants and unspecified obstacles | trained |

The taxonomy is deliberately compact. Range sensors answer whether something is
close; the detector supplies a useful semantic category. Combining bicycles and
motorcycles avoids thin classes that demand the same immediate haptic action.

## 4. Source-specific class preprocessing

Every source class must have an explicit mapping. An absent mapping is a hard
build error. `null` means intentionally discarded, not forgotten.

### 4.1 Mendeley VI Navigation v1

| Target | Source classes mapped into it |
|---|---|
| `person` | `Person` |
| `vehicle` | `Bus`, `CNG`, `Car`, `Food Van`, `Leguna`, `Pickup`, `Rickshaw`, `Truck`, `Van`, `Van gari` |
| `bicycle_motorcycle` | `Cycle`, `Motorcycle` |
| `pole` | `Building Pillar`, `Electric Pole` |
| `tree` | `Tree` |
| `stairs` | `Stairs` |
| `cone` | `Parking Cone` |
| `generic_obstacle` | `Bin`, `Food cart`, `Obstacle` |
| discarded | `Footpath` |

`Footpath` is dropped because it is normally the traversable background region,
not a discrete obstacle box. Treating it as an obstacle would teach the warning
system to fire on the desired walking surface.

### 4.2 SOD v1

| Target | Source classes mapped into it |
|---|---|
| `person` | `person` |
| `vehicle` | `bus`, `car`, `train`, `truck` |
| `bicycle_motorcycle` | `bicycle`, `motorcycle` |
| `pole` | `pole`, `street_light` |
| `tree` | `tree` |
| `stairs` | `stairs` |
| `curb` | `curb` |
| `barrier` | `spherical_roadblock`, `warning_column` |
| `dog` | `dog` |
| `signboard` | `stop_sign`, `traffic_light` |
| `generic_obstacle` | `bench`, `bus_stop`, `fire_hydrant`, `waste_container` |
| discarded | `cane` |

The SOD `cane` class is dropped because the product's own cane can enter the
top-handle camera view. Detecting the mobility aid itself as a hazard would be a
systematic false-positive failure.

### 4.3 Indian Unstructured Road v10

| Target | Source class |
|---|---|
| `vehicle` | `vehicle` |
| `bicycle_motorcycle` | `two-wheeler` |
| `pole` | `pole` |
| `tree` | `tree` |
| `stairs` | `stairs` |
| `dog` | `dog` |
| `signboard` | `signboard` |

No Indian-road class is discarded. Its value is not only class count: it adds
Indian roadway/pathway appearance that is closer to the intended deployment
than many generic Western sidewalk images.

## 5. Label normalization and geometry

The harmonizer accepts two YOLO annotation forms:

1. detection: `class x_center y_center width height`;
2. segmentation: `class x1 y1 x2 y2 ...` with at least three points.

Segmentation polygons are converted to their minimum axis-aligned bounding
rectangle. Every output is a five-field YOLO detection line. The build rejects:

- non-integer or out-of-range source class IDs;
- source classes without an explicit mapping;
- non-numeric coordinates;
- coordinates outside normalized `[0, 1]` bounds;
- zero/negative width or height;
- boxes extending beyond the normalized image on either axis;
- malformed detection or polygon field counts.

Coordinates are not rounded to coarse precision. They are serialized as
floating-point normalized values and remain invariant during later
aspect-preserving image resize.

## 6. Split policy and exact duplicate handling

Source-provided train/validation/test membership is preserved whenever the image
is retained. The build does not randomly reshuffle public exports because that
would lose their declared evaluation structure and make source comparisons hard
to reproduce.

Every image receives a SHA-256. If identical bytes appear in multiple splits,
only the highest-priority copy remains:

```text
test > validation > train
```

This priority protects held-out evidence at the cost of a small amount of
training data. Fifty-four exact duplicate images were removed in the initial
three-source build.

The first attempted build also stripped Roboflow `.rf.<hash>` suffixes and used
the remaining filename as lineage. That removed 5,452 images. Visual inspection
proved the assumption false: generic names such as `frame_0s` came from different
source videos and different streets. That rejected build is retained under
`data/processed/cane-v1-rejected-filename-lineage-build`, and filename collisions
are now audit signals only. They never cause deletion.

## 7. Initial harmonized build

The corrected three-source build observed 26,030 images and produced:

| Measurement | Count |
|---|---:|
| retained train images | 21,482 |
| retained validation images | 2,819 |
| retained test images | 1,515 |
| retained annotations | 69,848 |
| annotations deliberately dropped by class mapping | 2,505 |
| images skipped after every annotation was dropped | 151 |
| missing raw label files | 9 |
| exact duplicate images skipped | 54 |
| ambiguous cross-split filename keys | 380 |
| images covered by ambiguous filename keys | 6,158 |

Missing raw labels are counted and skipped during harmonization; they do not
become unlabeled negatives. The processed output has zero missing label files.

Initial retained source balance was 8,034 Mendeley, 9,920 SOD, and 7,862
Indian-road images. The sources are combined by concatenation after class
normalization; no source-specific resampling or class weighting is applied.

## 8. Perceptual leakage audit and decontamination

SHA-256 detects only identical bytes. Roboflow augmentation, JPEG re-encoding,
and adjacent video frames can preserve the scene while changing the hash.

The read-only audit therefore computes a 64-bit difference hash (dHash) for
every image and reports cross-split pairs at Hamming distance at most four. dHash
is only a candidate generator; it is not treated as proof of identity.

The first full audit found:

- zero corrupt/unreadable processed images;
- zero missing or orphan processed labels;
- zero invalid processed annotation files;
- zero remaining exact duplicates;
- 397 cross-split perceptual candidates.

Contact sheets across the distance range showed repeated scenes, export
transforms, and adjacent frames. The decontaminator removes the lower-priority
member of same-source candidate pairs using the same `test > val > train` rule.
It does not delete cross-source candidates automatically because unrelated low-
detail scenes can have similar 64-bit hashes.

This removed 263 lower-priority images while retaining all 1,515 test images.
The remaining cross-source candidate was visually reviewed: one image is a
yellow-black bollard at night and the other is a person outside a university
department entrance. The decision to retain both is saved in
`reports/dataset-visual-review.json`.

## 9. Transfer/training resize

The cleaned archival images occupy 5,063,472,209 bytes. YOLO26n training is run
at `imgsz=416`, so repeatedly uploading and decoding pixels far above the model's
consumed resolution adds transport and I/O cost without adding model input
detail.

The training-transfer dataset applies:

```text
long side > 640 px: resize aspect-preservingly to long side 640
long side <= 640 px: copy without recompression
resized JPEG: quality 95, chroma subsampling 0, optimized encoding
labels: byte-identical normalized YOLO coordinates
```

Results:

| Measurement | Value |
|---|---:|
| images resized | 7,856 |
| images copied without resize | 17,697 |
| original image bytes | 5,063,472,209 |
| resized image bytes | 1,931,900,118 |
| retained byte ratio | 38.15% |

Because resize/re-encoding can alter perceptual hashes, the audit was rerun. It
exposed one additional adjacent Indian-video pair (`...mp4-2` in validation and
`...mp4-3` in train); the train member was removed. The final training split is:

| Split | Images |
|---|---:|
| train | 21,269 |
| validation | 2,768 |
| test | 1,515 |
| **total** | **25,552** |

## 10. Final training-dataset composition

| Source | Final images |
|---|---:|
| Mendeley VI Navigation | 7,864 |
| SOD | 9,833 |
| Indian Unstructured Road | 7,855 |

| Target class | Final annotations |
|---|---:|
| `person` | 16,597 |
| `vehicle` | 19,914 |
| `bicycle_motorcycle` | 4,842 |
| `pole` | 6,518 |
| `tree` | 3,927 |
| `stairs` | 1,907 |
| `curb` | 986 |
| `barrier` | 3,984 |
| `cone` | 220 |
| `dog` | 1,366 |
| `signboard` | 3,183 |
| `door` | 0 |
| `slope` | 0 |
| `overhead_obstacle` | 0 |
| `generic_obstacle` | 5,460 |

Class imbalance is retained and disclosed; it has not been hidden by copying
rare images into held-out splits. Cone and curb results should be interpreted
with their smaller support in mind. Zero-count classes must not appear in a
claimed per-class performance average as if they were evaluated capabilities.

## 11. Final acceptance gate

`reports/dataset-audit-training-640.json` is the machine-readable final audit.
It records:

- zero missing labels;
- zero orphan labels;
- zero invalid label files;
- zero unreadable images;
- zero exact duplicate groups;
- zero unresolved same-source dHash candidates;
- one reviewed cross-source dHash false positive retained.

The remaining limitation is important: dHash cannot prove that every possible
temporally adjacent frame has been found. Training/test metrics must therefore
be described as evaluated on preserved-and-decontaminated source splits, not as
a perfect video-identity split.

## 12. Model-time preprocessing versus offline preprocessing

Offline dataset preprocessing ends at the 640-pixel JPEG dataset described
above. During YOLO training/inference, Ultralytics still performs its model input
pipeline: aspect-preserving letterbox to the requested `imgsz`, channel/tensor
conversion, and numeric normalization. Those runtime operations are not baked
into saved JPEGs.

Current fine-tuning inputs are:

```text
base weights: yolo26n.pt
data: cane-v1-training-640/data.yaml
training imgsz: 416
epochs requested: 30
early-stopping patience: 8
batch: 32
workers: 4
seed: 42
deterministic: true
mixed precision: enabled on CUDA
source splits: preserved train/val/test
```

The accepted first-22-epoch trainer reported Ultralytics 8.4.132, Python
3.13.15, PyTorch 2.11.0+cu128, and a Tesla T4. The Lightning continuation pins
Ultralytics 8.4.132 and records its actual Python, PyTorch, CUDA, and GPU values
in the final validation summary. The checkpoint's on-the-fly training
transforms are:

| Operation | Setting |
|---|---:|
| HSV hue | 0.015 |
| HSV saturation | 0.7 |
| HSV value | 0.4 |
| translation | 0.1 |
| scale | 0.5 |
| horizontal flip probability | 0.5 |
| vertical flip probability | 0.0 |
| mosaic | 1.0 |
| close mosaic | final 10 epochs |
| generic `erasing` argument | 0.4 |
| generic `auto_augment` argument | RandAugment |
| degrees / shear / perspective | 0.0 / 0.0 / 0.0 |
| mixup / cutmix / copy-paste | 0.0 / 0.0 / 0.0 |

The launch configuration requested optimizer `auto`. Ultralytics explicitly
ignored the generic `lr0=0.01` and `momentum=0.937` defaults and selected
`AdamW(lr=0.000526, momentum=0.9)`, with weight decay 0.0005 applied to the
appropriate parameter group. The final learning-rate factor remains 0.01, with
three warmup epochs and nominal batch size 64 for loss scaling. These transforms
operate only on training batches; they never modify validation or test files on
disk. The downloaded final `args.yaml` remains the immutable authority and will
be checked against this section after the run completes.

Training durability does not change any dataset or augmentation setting. The
Lightning runner validates and snapshots the newest optimizer-bearing
checkpoint at startup and after every completed epoch, and atomically refreshes
`cane-v1-latest-recovery.zip`. That bundle is a recovery artifact, not a new
dataset version or final-model selection; the final `best.pt` still requires the
held-out validation/test completion gate documented in
`docs/lightning-ai-training.md`.

`erasing` and `auto_augment` are present in Ultralytics' shared argument schema,
but may be classification-only rather than executed detection transforms. They
are recorded for completeness and must not be claimed as applied to Cane V1
without confirming the 8.4.132 detection augmentation call path.

## 13. Reproduction commands

Build the harmonized dataset from immutable raw sources:

```bash
uv run python scripts/harmonize_datasets.py \
  --config configs/dataset-sources.json \
  --output data/processed/cane-v1
```

Audit and visually review candidates:

```bash
uv run python scripts/audit_dataset.py \
  --dataset data/processed/cane-v1 \
  --output reports/dataset-audit.json \
  --contact-sheet reports/dataset-near-duplicates.jpg
```

Create the training-transfer form and rerun the same acceptance gate:

```bash
uv run python scripts/resize_training_dataset.py \
  --dataset data/processed/cane-v1-clean \
  --output data/processed/cane-v1-clean-640 \
  --max-dimension 640

uv run python scripts/decontaminate_dataset.py \
  --dataset data/processed/cane-v1-clean-640 \
  --audit reports/dataset-audit-clean-640.json \
  --output data/processed/cane-v1-training-640

uv run python scripts/audit_dataset.py \
  --dataset data/processed/cane-v1-training-640 \
  --output reports/dataset-audit-training-640.json
```

The exact transferred archive is:

```text
artifacts/training/cane-v1-training-640.tar
SHA-256: f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91
```

## 14. Files teammates should consult

| Need | File |
|---|---|
| source roots, class orders, mappings | `configs/dataset-sources.json` |
| raw provenance and immutable hashes | `docs/dataset-provenance.md` |
| canonical harmonization implementation | `src/yolo_pi/dataset.py` |
| audit/decontamination implementation | `src/yolo_pi/dataset_audit.py` |
| resize implementation | `src/yolo_pi/training_dataset.py` |
| final data definition | `data/processed/cane-v1-training-640/data.yaml` |
| final structural/leakage evidence | `reports/dataset-audit-training-640.json` |
| visual false-positive disposition | `reports/dataset-visual-review.json` |
| active Lightning resume and final-model workflow | `docs/lightning-ai-training.md` |
| guarded Colab epochs and recovery record | `docs/colab-workflow.md` |
| checkpoint-backed accepted metrics | `artifacts/training/cane-v1-training-history.json` |
| chronological engineering log | `changes.md` |

If any of the mapping, split, count, resize, or leakage policies change, update
this document and regenerate all downstream reports before training another
model version.
