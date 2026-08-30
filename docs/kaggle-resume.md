# Cane V1 fine-tune resume on Kaggle

**Status: failed. Do not use this path for the live Cane V1 run.**

A Kaggle resume was attempted on 2026-08-29/30 after Colab T4 assign returned
503. No training epochs ran. The attached image tree never matched the pinned
`cane-v1-training-640` split (21,269 / 2,768 / 1,515). The live resume remains
the Colab epoch-15 `last.pt` described in `docs/colab-workflow.md`.

This page is kept only as a record of that failed host failover. It does not
change the dataset, the fusion policy, the Pi tables, the camera mount, or V0.

## Why it failed

- The Kaggle dataset named like `cane-v1-training-640` still had the wrong JPEG
  counts (`20580 / 4432 / 2504`). `decontamination-report.json` on the same
  folder still said `21269 / 2768 / 1515`.
- `artifacts/training/cane-v1-training-640.tar` contains 9,833 extra macOS
  AppleDouble files named `._sod-v1__….jpg`. They look like images. After Kaggle
  auto-extracted the archive they were counted as JPEGs.
- Even after ignoring those files, real images on the Kaggle volume were
  incomplete: train **13,707** (need 21,269), val **2,468** (need 2,768), test
  **1,515** (complete). Train/val were truncated; do not train on that extract.

## Scope lock (historical)

Kaggle was intended only as a training-host substitute. The experiment is the
same run that started on Colab:

| Item | Unchanged value |
|---|---|
| Model | YOLO26n, 15 cane classes, 2,509,650 parameters |
| Dataset | `cane-v1-training-640` (21,269 / 2,768 / 1,515) |
| Archive SHA-256 | `f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91` |
| Resume checkpoint | protected epoch 15 `last.pt` |
| Checkpoint SHA-256 | `cfeb7c9a252b2344c713a9721c06ab6d5b5439ce4f8b98080413dd754d578274` |
| Trainer | `ultralytics==8.4.132` |
| Schedule | 30 epochs, patience 8, imgsz 416, batch 32, seed 42, AMP |
| Next epoch | 16 / 30 |

Do not start from `yolo26n.pt`. Do not rebuild or re-harmonize the dataset for
this run. Do not treat Kaggle numbers as Raspberry Pi latency. Do not resume
from any checkpoint produced on Kaggle; none exists.

The notebook is `notebooks/cane_v1_kaggle_resume.ipynb`. Regenerate it with:

```bash
uv run python scripts/build_kaggle_resume_notebook.py
```

## What you upload to Kaggle

Create a notebook from that `.ipynb` (Kaggle UI: **File → Import Notebook**, or
the kernel metadata file next to it).

Attach one or two private datasets that contain:

1. `artifacts/training/cane-v1-training-640.tar`
2. `artifacts/training/checkpoint-guard/checkpoints/latest-last.pt`

Renaming the files is allowed. The notebook searches `/kaggle/input` by name and
then **refuses to run** if the SHA-256 values above do not match.

Optional: upload the extracted `cane-v1-training-640/` folder instead of the tar.
The notebook then checks the 21,269 / 2,768 / 1,515 split counts instead of the
tar hash. The tar remains the preferred, pinned transfer form.

## Notebook settings

- Accelerator: **GPU**. Prefer T4 when Kaggle offers it (same class as Colab).
  P100 is acceptable for this nano model; it is a host difference, not a new
  training recipe.
- Internet: **On**, so `ultralytics==8.4.132` can be pinned. Do not accept a
  newer Ultralytics patch.
- Time: remaining work is about 15 epochs. On a T4 that was ~4 minutes per
  epoch plus extract/val/export. A 12-hour Kaggle GPU session is enough.

## What the notebook does

1. Pins `ultralytics==8.4.132` and refuses to run on CPU.
2. Checksums the archive and the epoch-15 checkpoint.
3. Extracts the dataset (or reuses a previous extract in `/kaggle/working`).
4. Rewrites `data.yaml` `path:` to an absolute directory (same Ultralytics cwd
   fix as Colab).
5. Copies the checkpoint to
   `/kaggle/working/yolo-pi-runs/cane-v1-yolo26n-e30-img416/weights/last.pt`,
   verifies the 15-class taxonomy and optimizer, and rewrites only the stored
   `/content/...` host paths so resume does not look for Colab directories.
6. Calls `YOLO(last.pt).train(resume=True)`. Epoch, optimizer, scaler, and
   fitness stay serialized.
7. Validates `best.pt` on val and test at **imgsz 416**.
8. Exports ONNX and FP16 NCNN with `end2end=False`.
9. Writes `validation-summary.json` and
   `/kaggle/working/yolo-pi-kaggle-artifacts.zip`.

If Kaggle interrupts the session, the latest `last.pt` in the notebook output
can be attached on a later run. The notebook will resume from a working-dir
checkpoint whose completed epoch is ≥ 15 and that still has optimizer state.

## What you download afterward

From the completed version output:

- `yolo-pi-kaggle-artifacts.zip`
- `protected/best.pt`
- `protected/last.pt`
- `protected/validation-summary.json`
- `protected/results.csv`
- `protected/args.yaml`

Store them under `artifacts/training/` in this repository. They are a resumed
Cane V1 checkpoint, not a new model family.

## What this notebook does *not* do

These stay in the original repo workflows:

- Dataset acquisition, harmonization, audit, decontamination, and 640-pixel
  resize (`docs/data-preprocessing.md`)
- Range-first fusion, haptic policy, and detector contracts
- Architecture-table quality at **imgsz 640** on the full 1,515-image test split
  (`scripts/evaluate_model_comparison.py`)
- Six Pi backend exports at 640/FP32 (`scripts/export_pi_backends.py`)
- Raspberry Pi measurements (`docs/pi-benchmarking.md`)

After `best.pt` is on the laptop, run the 640-pixel comparison and the Pi
exports from that immutable file, exactly as planned when Colab was the host.

## Known host differences versus Colab

- The GPU may be a Kaggle T4 or P100 instead of the original Tesla T4.
- The preinstalled PyTorch wheel is Kaggle's, not Colab's `2.11.0+cu128`.
  Ultralytics is still pinned to **8.4.132**.
- Stored Colab paths inside the checkpoint are rewritten to `/kaggle/working`.
  Training hyperparameters are not rewritten.

Those differences affect wall-clock time, not the dataset or the class map.
Record the printed `gpu` / `torch` / `ultralytics` dict in `changes.md` when
the run finishes.
