from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def resize_training_dataset(
    source_root: Path, output_root: Path, *, max_dimension: int = 640
) -> Dict[str, object]:
    if max_dimension <= 0:
        raise ValueError("max_dimension must be positive")
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to resize the training dataset") from exc

    temporary = output_root.with_name(output_root.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    counts: Counter = Counter()
    original_bytes = 0
    output_bytes = 0
    try:
        for split in ("train", "val", "test"):
            image_source = source_root / "images" / split
            label_source = source_root / "labels" / split
            image_output = temporary / "images" / split
            label_output = temporary / "labels" / split
            image_output.mkdir(parents=True, exist_ok=True)
            label_output.mkdir(parents=True, exist_ok=True)
            for image_path in sorted(image_source.glob("*.jpg")):
                destination = image_output / image_path.name
                original_bytes += image_path.stat().st_size
                with Image.open(image_path) as image:
                    width, height = image.size
                    if max(width, height) > max_dimension:
                        converted = image.convert("RGB")
                        converted.thumbnail((max_dimension, max_dimension))
                        converted.save(
                            destination,
                            format="JPEG",
                            quality=95,
                            optimize=True,
                            subsampling=0,
                        )
                        counts["images_resized"] += 1
                    else:
                        shutil.copy2(image_path, destination)
                        counts["images_copied_without_resize"] += 1
                output_bytes += destination.stat().st_size
                counts[f"images_{split}"] += 1
                label_path = label_source / f"{image_path.stem}.txt"
                if not label_path.exists():
                    raise FileNotFoundError(label_path)
                counts[f"labels_{_link_or_copy(label_path, label_output / label_path.name)}"] += 1
        for name in ("data.yaml", "build-report.json", "decontamination-report.json"):
            path = source_root / name
            if path.exists():
                shutil.copy2(path, temporary / name)
        report = {
            "schema_version": 1,
            "source_dataset": str(source_root),
            "output_dataset": str(output_root),
            "max_dimension": max_dimension,
            "jpeg_quality": 95,
            "jpeg_subsampling": 0,
            "annotation_transform": "none; normalized YOLO coordinates are invariant under aspect-preserving resize",
            "statistics": dict(sorted(counts.items())),
            "original_image_bytes": original_bytes,
            "output_image_bytes": output_bytes,
            "byte_ratio": output_bytes / original_bytes if original_bytes else 0.0,
            "intended_training_imgsz": 416,
            "warning": "This is a training-transfer optimization. The decontaminated source dataset remains the archival authority.",
        }
        (temporary / "resize-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output_root)
        return report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
