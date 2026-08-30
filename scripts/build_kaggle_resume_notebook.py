#!/usr/bin/env python3
"""Generate the Kaggle-only Cane V1 fine-tune resume notebook."""

from pathlib import Path

import nbformat as nbf

OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "cane_v1_kaggle_resume.ipynb"


def main() -> int:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {
            "accelerator": "gpu",
            "internet": True,
            "isInternetEnabled": True,
        },
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Cane V1 YOLO26n — Kaggle fine-tune resume (epoch 15 → 30)

**Status: failed. Do not use this notebook for the live run.**

The 2026-08-29/30 Kaggle attempt never trained. The attached image tree did not
match the pinned 21,269 / 2,768 / 1,515 split. Resume on Colab from the
protected epoch-15 `last.pt` instead. See `docs/kaggle-resume.md`.

## Scope lock

This notebook was meant to do **only** one thing: resume the already-started Cane V1
YOLO26n fine-tune from the protected **epoch 15** checkpoint and continue to
early stopping or epoch 30. It did not reach that point.

Everything else in the repository stays the same:

- dataset taxonomy, splits, decontamination, and 640-pixel transfer archive
- fusion policy, haptic mapping, and detector contracts
- Pi benchmark matrix, camera-mount contract, and handoff docs
- V0 baseline, export-parity rules, and held-out 640-pixel paper evaluation

Kaggle is a **training host substitute** because the Colab T4 assign path is
currently unavailable. It is not a new experiment, not a new dataset, and not
Raspberry Pi evidence.

Do **not** train from `yolo26n.pt`. Do **not** change `imgsz`, `epochs`,
`patience`, `batch`, `seed`, or class names."""
        ),
        nbf.v4.new_markdown_cell(
            """## Kaggle setup (do this in the UI before Run All)

1. **Notebook settings**
   - Accelerator: **GPU** (T4 if the menu offers it; P100 is acceptable)
   - Internet: **On** (needed to pin `ultralytics==8.4.132`)
   - Persistence: default is fine; outputs land in `/kaggle/working`
2. **Create and attach two datasets** (or one dataset that contains both files)
   - `cane-v1-training-640.tar` from `artifacts/training/cane-v1-training-640.tar`
   - `latest-last.pt` from `artifacts/training/checkpoint-guard/checkpoints/latest-last.pt`
3. Confirm both files appear under `/kaggle/input/**` then **Run All**.

Exact hashes are enforced in code. A renamed file is fine; a different file is not.

| Artifact | SHA-256 |
|---|---|
| `cane-v1-training-640.tar` | `f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91` |
| epoch-15 `last.pt` / `best.pt` | `cfeb7c9a252b2344c713a9721c06ab6d5b5439ce4f8b98080413dd754d578274` |

Training recipe (unchanged from Colab): YOLO26n, 30 epochs, patience 8, imgsz 416,
batch 32, seed 42, AMP, `ultralytics==8.4.132`. Resume continues at **epoch 16/30**.

After the run, download `/kaggle/working/yolo-pi-kaggle-artifacts.zip` and the
top-level `protected/` weights. Laptop-side 640-pixel M1/M2 evaluation and the
six Pi backend exports still happen in this repo, not in this notebook.

Full host instructions: `docs/kaggle-resume.md`."""
        ),
        nbf.v4.new_markdown_cell("## 1. Pin the trainer and fail closed without a GPU"),
        nbf.v4.new_code_cell(
            r"""%pip install -q 'ultralytics==8.4.132'"""
        ),
        nbf.v4.new_code_cell(
            r"""from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTHONUNBUFFERED"] = "1"

for name in list(sys.modules):
    if name == "ultralytics" or name.startswith("ultralytics."):
        del sys.modules[name]

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS["wandb"] = False
if "analytics" in SETTINGS:
    SETTINGS["analytics"] = False

assert ultralytics.__version__ == "8.4.132", ultralytics.__version__
assert torch.cuda.is_available(), (
    "Enable a GPU accelerator in Kaggle Notebook settings before running."
)

INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
PROJECT = WORKING / "yolo-pi-runs"
RUN_NAME = "cane-v1-yolo26n-e30-img416"
RUN_DIR = PROJECT / RUN_NAME
WEIGHTS_DIR = RUN_DIR / "weights"
PROTECTED = WORKING / "protected"
DATASETS = WORKING / "datasets"

EXPECTED_ARCHIVE_SHA256 = (
    "f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "cfeb7c9a252b2344c713a9721c06ab6d5b5439ce4f8b98080413dd754d578274"
)
EXPECTED_EPOCH_ZERO_BASED = 14
EXPECTED_COMPLETED_EPOCH = 15
EXPECTED_SPLIT_COUNTS = {"train": 21269, "val": 2768, "test": 1515}
EXPECTED_NAMES = {
    0: "person",
    1: "vehicle",
    2: "bicycle_motorcycle",
    3: "pole",
    4: "tree",
    5: "stairs",
    6: "curb",
    7: "barrier",
    8: "cone",
    9: "dog",
    10: "signboard",
    11: "door",
    12: "slope",
    13: "overhead_obstacle",
    14: "generic_obstacle",
}
ARCHIVE_NAMES = {"cane-v1-training-640.tar"}
CHECKPOINT_NAMES = {
    "latest-last.pt",
    "latest-best.pt",
    "last.pt",
    "best.pt",
    "cane-v1-resume-epoch15.pt",
    "cane-v1-last-epoch15.pt",
}

print(
    {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "vram_mb": int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024),
        "host": "kaggle",
        "purpose": "resume Cane V1 fine-tune only",
    }
)"""
        ),
        nbf.v4.new_markdown_cell("## 2. Locate and checksum the pinned archive and epoch-15 checkpoint"),
        nbf.v4.new_code_cell(
            r"""def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in root.rglob("*") if path.is_file())


def find_by_names(root: Path, names: set[str]) -> List[Path]:
    return sorted(path for path in iter_files(root) if path.name in names)


def find_data_yaml(root: Path) -> List[Path]:
    hits = []
    for path in iter_files(root):
        if path.name == "data.yaml" and (path.parent / "images" / "train").is_dir():
            hits.append(path)
    return sorted(hits)


search_roots = [INPUT_ROOT, WORKING]
archive_hits: List[Path] = []
checkpoint_hits: List[Path] = []
yaml_hits: List[Path] = []
for root in search_roots:
    archive_hits.extend(find_by_names(root, ARCHIVE_NAMES))
    checkpoint_hits.extend(find_by_names(root, CHECKPOINT_NAMES))
    yaml_hits.extend(find_data_yaml(root))

# Prefer /kaggle/input over working copies of the same name.
def prefer_input(paths: List[Path]) -> List[Path]:
    unique: Dict[str, Path] = {}
    for path in paths:
        key = path.name + ":" + str(path.stat().st_size)
        if key in unique and str(unique[key]).startswith("/kaggle/input"):
            continue
        unique[key] = path
    return list(unique.values())

archive_hits = prefer_input(archive_hits)
checkpoint_hits = prefer_input(checkpoint_hits)

print(
    {
        "archives": [str(path) for path in archive_hits],
        "checkpoints": [str(path) for path in checkpoint_hits],
        "extracted_yaml": [str(path) for path in yaml_hits],
    }
)
if not archive_hits and not yaml_hits:
    raise FileNotFoundError(
        "Attach cane-v1-training-640.tar (or its extracted folder) under /kaggle/input"
    )
if not checkpoint_hits:
    raise FileNotFoundError(
        "Attach the epoch-15 latest-last.pt checkpoint under /kaggle/input"
    )"""
        ),
        nbf.v4.new_code_cell(
            r"""def extract_archive(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        destination_root = destination.resolve()
        members = [member for member in handle.getmembers() if member.name]
        for member in members:
            target = (destination / member.name).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise ValueError(f"unsafe archive path: {member.name}")
        roots = sorted({Path(member.name).parts[0] for member in members})
        extract_kwargs = {}
        if sys.version_info >= (3, 12):
            extract_kwargs["filter"] = "data"
        handle.extractall(destination, **extract_kwargs)
    if len(roots) != 1:
        raise ValueError(f"dataset archive must contain exactly one root, found {roots}")
    data_yaml = destination / roots[0] / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    return rewrite_data_yaml(data_yaml)


def rewrite_data_yaml(source_yaml: Path) -> Path:
    # Working YAML must use an absolute path:, not cwd-relative "."
    dataset_root = source_yaml.parent.resolve()
    lines = source_yaml.read_text(encoding="utf-8").splitlines()
    rewritten = []
    replaced = False
    for line in lines:
        if line.strip().startswith("path:"):
            rewritten.append(f"path: {dataset_root}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        rewritten.insert(0, f"path: {dataset_root}")
    destination = WORKING / "cane-v1-data.yaml"
    destination.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return destination


def count_split_images(dataset_root: Path) -> Dict[str, int]:
    counts = {}
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        counts[split] = (
            sum(1 for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if image_dir.is_dir()
            else 0
        )
    return counts


data_yaml: Optional[Path] = None
dataset_archive_meta = None
if archive_hits:
    archive = archive_hits[0]
    actual_archive = sha256(archive)
    if actual_archive != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            f"dataset archive checksum mismatch: {actual_archive} from {archive}"
        )
    extracted_root = DATASETS / "cane-v1-training-640"
    extracted_yaml = extracted_root / "data.yaml"
    if extracted_yaml.exists():
        data_yaml = rewrite_data_yaml(extracted_yaml)
    else:
        data_yaml = extract_archive(archive, DATASETS)
    dataset_archive_meta = {
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": actual_archive,
    }
else:
    source_yaml = yaml_hits[0]
    data_yaml = rewrite_data_yaml(source_yaml)

dataset_root = Path(
    next(
        line.split(":", 1)[1].strip()
        for line in data_yaml.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("path:")
    )
)
split_counts = count_split_images(dataset_root)
if split_counts != EXPECTED_SPLIT_COUNTS:
    raise RuntimeError(f"unexpected split counts: {split_counts}")

print(
    {
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "split_counts": split_counts,
        "archive": dataset_archive_meta,
    }
)"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Install the protected epoch-15 weights as Ultralytics `last.pt`

The checkpoint stores Colab paths under `/content/...`. Those paths are rewritten
to this Kaggle run directory **before** `resume=True`, otherwise Ultralytics
would look for a machine that no longer exists. Optimizer state, epoch, and
fitness are left untouched."""
        ),
        nbf.v4.new_code_cell(
            r"""def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_report(path: Path, ckpt: dict) -> dict:
    model = ckpt.get("ema") or ckpt.get("model")
    names = {int(index): str(name) for index, name in getattr(model, "names", {}).items()}
    if names != EXPECTED_NAMES:
        raise ValueError(f"unexpected checkpoint class taxonomy: {names}")
    epoch_zero_based = int(ckpt.get("epoch", -1))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "epoch_zero_based": epoch_zero_based,
        "completed_epoch": epoch_zero_based + 1,
        "best_fitness": float(ckpt.get("best_fitness", 0.0)),
        "has_optimizer": ckpt.get("optimizer") is not None,
        "class_names": names,
    }


resumable = []
for candidate in checkpoint_hits:
    loaded = load_checkpoint(candidate)
    report = checkpoint_report(candidate, loaded)
    if not report["has_optimizer"]:
        continue
    if report["epoch_zero_based"] < EXPECTED_EPOCH_ZERO_BASED:
        continue
    if (
        report["sha256"] != EXPECTED_CHECKPOINT_SHA256
        and report["epoch_zero_based"] == EXPECTED_EPOCH_ZERO_BASED
    ):
        continue
    resumable.append((report["epoch_zero_based"], str(candidate), candidate, loaded, report))

if not resumable:
    raise RuntimeError(
        "no attached file matched the epoch-15 SHA-256; upload latest-last.pt"
    )
resumable.sort()
_, _, input_ckpt_path, input_ckpt, input_report = resumable[-1]
if not input_report["has_optimizer"]:
    raise RuntimeError("checkpoint has no optimizer state and cannot be resumed")
if input_report["epoch_zero_based"] < EXPECTED_EPOCH_ZERO_BASED:
    raise RuntimeError(
        f"refusing to resume from epoch {input_report['completed_epoch']}; expected >= 15"
    )

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
PROTECTED.mkdir(parents=True, exist_ok=True)
last_path = WEIGHTS_DIR / "last.pt"
best_path = WEIGHTS_DIR / "best.pt"

# Rewrite only host paths. Do not reset epoch, optimizer, or fitness.
train_args = dict(input_ckpt.get("train_args") or {})
train_args["data"] = str(data_yaml)
train_args["project"] = str(PROJECT)
train_args["name"] = RUN_NAME
train_args["save_dir"] = str(RUN_DIR)
train_args["model"] = str(last_path)
train_args["resume"] = str(last_path)
train_args["device"] = "0"
input_ckpt["train_args"] = train_args
torch.save(input_ckpt, last_path)
# Epoch 15 last.pt was also best.pt. Do not clobber a later Kaggle best.pt.
if not best_path.exists() or (
    input_report["sha256"] == EXPECTED_CHECKPOINT_SHA256
    and input_report["completed_epoch"] == EXPECTED_COMPLETED_EPOCH
):
    shutil.copy2(last_path, best_path)

installed = checkpoint_report(last_path, load_checkpoint(last_path))
print(
    {
        "source_checkpoint": input_report,
        "installed_last": installed,
        "train_args_data": train_args["data"],
        "train_args_save_dir": train_args["save_dir"],
        "next_epoch": installed["completed_epoch"] + 1,
    }
)"""
        ),
        nbf.v4.new_markdown_cell("## 4. Resume training — do not start a new run"),
        nbf.v4.new_code_cell(
            r"""model = YOLO(str(last_path))
train_result = model.train(resume=True)
run_dir = Path(train_result.save_dir)
best_path = run_dir / "weights" / "best.pt"
last_path = run_dir / "weights" / "last.pt"
if not best_path.exists():
    raise FileNotFoundError(best_path)
print(
    {
        "training_mode": "resumed",
        "save_dir": str(run_dir),
        "best": str(best_path),
        "last": str(last_path),
        "best_bytes": best_path.stat().st_size,
    }
)"""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. Held-out val/test at the training resolution (416)

This is the trainer's own split check. The architecture-table quality number is
still the laptop script `scripts/evaluate_model_comparison.py` at **640** on the
full 1,515-image test split. Do not treat the next cell as the paper M2 result."""
        ),
        nbf.v4.new_code_cell(
            r"""def jsonable_metrics(metrics) -> Dict[str, float]:
    import numbers

    return {
        str(name): float(value)
        for name, value in getattr(metrics, "results_dict", {}).items()
        if isinstance(value, numbers.Real)
    }


def normalized_detection_metrics(metrics) -> Dict[str, float]:
    raw = jsonable_metrics(metrics)
    aliases = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    normalized = {name: raw[key] for name, key in aliases.items() if key in raw}
    normalized["fitness"] = float(getattr(metrics, "fitness", 0.0))
    return normalized


best_model = YOLO(str(best_path))
validation_results: Dict[str, Dict[str, object]] = {}
for split in ("val", "test"):
    try:
        result = best_model.val(
            data=str(data_yaml),
            imgsz=416,
            device=0,
            split=split,
            plots=False,
            verbose=False,
        )
        validation_results[split] = {
            "status": "complete",
            "normalized": normalized_detection_metrics(result),
            "raw": jsonable_metrics(result),
        }
    except Exception as exc:
        validation_results[split] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
print(json.dumps(validation_results, indent=2, sort_keys=True))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 6. Export ONNX and FP16 NCNN from `best.pt`

Same Colab smoke exports: NCNN uses `end2end=False` (one-to-many head + runtime
NMS). The six Pi backends at 640/FP32 are still produced later from the
immutable final `best.pt` on the laptop/Linux path. A failed NCNN export does
not discard the trained weights."""
        ),
        nbf.v4.new_code_cell(
            r"""def export_model(best: Path) -> Dict[str, Dict[str, object]]:
    exports: Dict[str, Dict[str, object]] = {}
    for export_format in ("onnx", "ncnn"):
        options = {
            "format": export_format,
            "imgsz": 416,
            "batch": 1,
            "device": "cpu",
        }
        if export_format == "ncnn":
            options.update({"end2end": False, "quantize": 16})
        try:
            exported = Path(YOLO(str(best)).export(**options))
            if not exported.exists():
                raise FileNotFoundError(str(exported))
            artifact_files = (
                [exported]
                if exported.is_file()
                else sorted(path for path in exported.rglob("*") if path.is_file())
            )
            exports[export_format] = {
                "status": "complete",
                "path": str(exported),
                "files": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in artifact_files
                ],
            }
        except Exception as exc:
            exports[export_format] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return exports


exports = export_model(best_path)
print(json.dumps(exports, indent=2, sort_keys=True))"""
        ),
        nbf.v4.new_markdown_cell("## 7. Package artifacts for download"),
        nbf.v4.new_code_cell(
            r"""final_best = checkpoint_report(best_path, load_checkpoint(best_path))
final_last = checkpoint_report(last_path, load_checkpoint(last_path))
parameter_count = sum(
    parameter.numel() for parameter in best_model.model.parameters()
)
summary = {
    "schema_version": 2,
    "purpose": "Cane V1 fine-tune resume on Kaggle; not Raspberry Pi performance evidence",
    "scope": "fine-tune resume only; dataset, fusion, and Pi contracts unchanged",
    "host": "kaggle",
    "base_model": "yolo26n.pt",
    "data": str(data_yaml),
    "dataset_archive": dataset_archive_meta,
    "epochs_requested": 30,
    "patience": 8,
    "imgsz": 416,
    "batch": 32,
    "workers": 4,
    "seed": 42,
    "training_mode": "resumed",
    "resume_from_completed_epoch": input_report["completed_epoch"],
    "resume_from_sha256": input_report["sha256"],
    "cuda_available": True,
    "hardware": torch.cuda.get_device_name(0),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "ultralytics": ultralytics.__version__,
    "parameter_count": parameter_count,
    "best_weights": final_best,
    "last_weights": final_last,
    "quality_metrics": validation_results,
    "exports": exports,
}
summary_path = RUN_DIR / "validation-summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

PROTECTED.mkdir(parents=True, exist_ok=True)
for source, name in (
    (best_path, "best.pt"),
    (last_path, "last.pt"),
    (summary_path, "validation-summary.json"),
    (RUN_DIR / "results.csv", "results.csv"),
    (RUN_DIR / "args.yaml", "args.yaml"),
):
    if source.exists():
        shutil.copy2(source, PROTECTED / name)

archive_path = Path(
    shutil.make_archive(str(WORKING / "yolo-pi-kaggle-artifacts"), "zip", PROJECT)
)
with zipfile.ZipFile(archive_path) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise ValueError(f"corrupt ZIP member: {bad}")
    names = archive.namelist()
required = ("weights/best.pt", "weights/last.pt", "validation-summary.json")
missing = [marker for marker in required if not any(name.endswith(marker) for name in names)]
if missing:
    raise RuntimeError(f"final ZIP is missing required members: {missing}")

summary["artifact_archive"] = {
    "path": str(archive_path),
    "bytes": archive_path.stat().st_size,
    "sha256": sha256(archive_path),
    "zip_test": "passed",
    "members": len(names),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(WORKING / "validation-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("RESULT_JSON=" + json.dumps(summary, sort_keys=True))
print("Download /kaggle/working/yolo-pi-kaggle-artifacts.zip and /kaggle/working/protected/")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Checks and next steps

- [ ] GPU cell printed `ultralytics==8.4.132` and a CUDA device
- [ ] Archive / checkpoint hashes matched the table above
- [ ] Training started at **16/30** (or later if this is a Kaggle re-resume)
- [ ] `protected/best.pt` and `yolo-pi-kaggle-artifacts.zip` are in the notebook output
- [ ] Download those files into `artifacts/training/` on the laptop
- [ ] Run `scripts/evaluate_model_comparison.py` at imgsz 640 on the full test split
- [ ] Export the six Pi backends from that immutable `best.pt`

Kaggle metrics are not Pi latency. Dataset and fusion code were not modified by
this notebook."""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    nbf.validate(nbf.read(OUTPUT, as_version=4))
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
