# Dataset provenance and acceptance gates

The V1 model combines public assistive-navigation datasets only after each
download is pinned to a version, checked against its published class order, and
passed through `scripts/harmonize_datasets.py`.

## Candidate sources

| Source | Pinned version | Published scope | Licence recorded by host |
|---|---|---|---|
| [Mendeley VI Navigation](https://doi.org/10.17632/m68g3h7p87.1) | Version 1, 24 March 2025 | 8,114 annotated images derived from pathway/roadside video; 22 classes | CC BY 4.0 |
| [SOD](https://universe.roboflow.com/yunuss-workspace-kkn8c/sod-enect-gyoww/dataset/1) | Dataset version 1 | 10,000 images, 30,834 instances, 22 sidewalk-obstacle classes | CC BY 4.0 |
| [Indian Unstructured Road Obstacles](https://universe.roboflow.com/mtechcv/unstructured-road-obstacle-detection/dataset/10) | Dataset version 10 | 2,208 images captured from Indian-road pathways; 7 classes | CC BY 4.0 |

The Mendeley page is the authoritative record for its dataset. The two Roboflow
pages are host-provided records; retain the downloaded export's README and
licence alongside the raw data for the paper archive.

## Acquired immutable inputs

| Source export | Local archive | SHA-256 | Integrity |
|---|---|---|---|
| Mendeley VI Navigation v1 | `data/raw/mendeley-vi-navigation-v1.zip` | `0ec4d17af22257d8c36c1d4d36eb4bf33cf759167a7f950cfcbc78c05ff672ca` | 16,222 entries; ZIP test passed; no unsafe paths |
| SOD v1 YOLO26 | `data/raw/sod-v1-yolo26.zip` | `d4ed5a394b650dc21226c69a8aa2e8114d123f0479c1ae65e3efc8c8aaa0c8d4` | 20,005 entries; ZIP test passed; no unsafe paths |
| Indian Unstructured Road v10 YOLO26 | `data/raw/indian-unstructured-road-v10-yolo26.zip` | `9d042cc5ad57a33bd1a965243cff4ba49817e8c8ea4d5f85d186d661fcb25b82` | 15,836 entries; ZIP test passed; no unsafe paths |

The SOD archive repeats the same 533-byte `data.yaml` member three times. All
three members have identical size, CRC, class order, license, and project URL;
extraction uses the final identical member. This packaging anomaly is retained
in the raw ZIP and disclosed rather than silently repaired.

The inspected SOD export contains 7,000 train, 2,000 validation, and 1,000 test
images with no export-time augmentation. Its actual 22-class order is recorded
in `configs/dataset-sources.example.json`.

The Indian-road v10 archive is 4,254,194,773 bytes and contains 7,620 train and
296 validation images with no test split. The version was generated with three
outputs per original training example (horizontal flip, rotation, blur, and
noise options recorded by the host), so lineage and temporal-leakage checks are
especially important. Its two `data.yaml` members are byte-identical and record
the actual class order `dog, pole, signboard, stairs, tree, two-wheeler,
vehicle`.

## Required checks before V1 training

1. Export every source in YOLOv8/YOLO11 detection format, retaining its README.
2. Confirm the `names` order in the export matches the corresponding entry in
   `configs/dataset-sources.json`; edit the config if the downloaded version
   differs from the example.
3. Review every explicit class mapping. `null` means the source annotation is
   intentionally excluded; an unmapped source class is a hard error.
4. Run the harmonizer. It validates normalized boxes, converts segmentation
   polygons to bounding rectangles, removes exact duplicate images by SHA-256,
   preserves source-provided train/validation/test splits, and emits a report.
5. Review `build-report.json`, especially missing labels, dropped classes,
   skipped images, per-class counts, and source balance.
6. Manually inspect a stratified sample from every target class and split.
7. Check sequential frames from source videos. Exact hashes cannot detect
   adjacent near-duplicates, so video-level grouping may be needed to prevent
   train/test leakage.

## Taxonomy policy

The target taxonomy is intentionally semantic and compact. Range sensors answer
whether something is close; vision identifies a useful category. Classes with
insufficient public examples (`door`, `slope`, and `overhead_obstacle`) remain in
the target list for later cane-camera collection, but their zero/low counts must
be disclosed rather than hidden.
