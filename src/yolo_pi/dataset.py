"""Dataset validation, taxonomy harmonization, de-duplication, and packaging."""

import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
ROBOFLOW_VARIANT_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{16,64}$", re.IGNORECASE)


class DatasetBuildError(ValueError):
    pass


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path
    names: Tuple[str, ...]
    class_map: Mapping[str, Optional[str]]
    license_name: str
    source_url: str


@dataclass(frozen=True)
class ParsedAnnotation:
    target_class: str
    center_x: float
    center_y: float
    width: float
    height: float

    def yolo_line(self, target_id: int) -> str:
        return (
            f"{target_id} {self.center_x:.8f} {self.center_y:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )


def _validate_unit(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise DatasetBuildError(f"{label} must be normalized to [0, 1], got {value}")


def _box_from_tokens(tokens: Sequence[str]) -> Tuple[float, float, float, float]:
    if len(tokens) == 5:
        center_x, center_y, width, height = [float(value) for value in tokens[1:]]
    elif len(tokens) >= 7 and (len(tokens) - 1) % 2 == 0:
        coordinates = [float(value) for value in tokens[1:]]
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
    else:
        raise DatasetBuildError(
            "Annotation must be YOLO detection (5 fields) or segmentation "
            "(class plus at least three x/y points)"
        )

    for value, name in (
        (center_x, "center_x"),
        (center_y, "center_y"),
        (width, "width"),
        (height, "height"),
    ):
        _validate_unit(value, name)
    if width <= 0.0 or height <= 0.0:
        raise DatasetBuildError("Annotation width and height must be positive")
    if center_x - width / 2 < -1e-6 or center_x + width / 2 > 1 + 1e-6:
        raise DatasetBuildError("Annotation extends outside normalized image width")
    if center_y - height / 2 < -1e-6 or center_y + height / 2 > 1 + 1e-6:
        raise DatasetBuildError("Annotation extends outside normalized image height")
    return center_x, center_y, width, height


def parse_label_lines(
    lines: Iterable[str], source: SourceSpec
) -> Tuple[List[ParsedAnnotation], int, int]:
    """Parse/remap detection or polygon labels.

    Returns retained annotations, dropped annotations, and original non-empty
    annotation count.
    """

    retained: List[ParsedAnnotation] = []
    dropped = 0
    original_count = 0
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        original_count += 1
        tokens = stripped.split()
        try:
            source_id = int(tokens[0])
        except (ValueError, IndexError) as exc:
            raise DatasetBuildError(f"Line {line_number}: invalid class id") from exc
        if source_id < 0 or source_id >= len(source.names):
            raise DatasetBuildError(
                f"Line {line_number}: class id {source_id} outside source taxonomy"
            )
        source_name = source.names[source_id]
        if source_name not in source.class_map:
            raise DatasetBuildError(
                f"Line {line_number}: source class {source_name!r} has no explicit mapping"
            )
        target_name = source.class_map[source_name]
        if target_name is None:
            dropped += 1
            continue
        try:
            center_x, center_y, width, height = _box_from_tokens(tokens)
        except (DatasetBuildError, ValueError) as exc:
            raise DatasetBuildError(f"Line {line_number}: {exc}") from exc
        retained.append(
            ParsedAnnotation(target_name, center_x, center_y, width, height)
        )
    return retained, dropped, original_count


def source_from_dict(payload: Mapping[str, object]) -> SourceSpec:
    return SourceSpec(
        name=str(payload["name"]),
        root=Path(str(payload["root"])),
        names=tuple(str(name) for name in payload["names"]),
        class_map={
            str(name): None if target is None else str(target)
            for name, target in dict(payload["class_map"]).items()
        },
        license_name=str(payload["license"]),
        source_url=str(payload["source_url"]),
    )


def load_build_config(path: Path) -> Tuple[Tuple[str, ...], List[SourceSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_names = tuple(str(name) for name in payload["target_names"])
    if not target_names or len(set(target_names)) != len(target_names):
        raise DatasetBuildError("target_names must be non-empty and unique")
    sources = [source_from_dict(item) for item in payload["sources"]]
    for source in sources:
        unknown_targets = {
            target
            for target in source.class_map.values()
            if target is not None and target not in target_names
        }
        if unknown_targets:
            raise DatasetBuildError(
                f"Source {source.name} maps to unknown targets: {sorted(unknown_targets)}"
            )
        missing = set(source.names) - set(source.class_map)
        extra = set(source.class_map) - set(source.names)
        if missing or extra:
            raise DatasetBuildError(
                f"Source {source.name} taxonomy mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
    return target_names, sources


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _discover_split(source: SourceSpec, split_dir_name: str) -> Iterable[Tuple[Path, Path]]:
    images_dir = source.root / split_dir_name / "images"
    labels_dir = source.root / split_dir_name / "labels"
    if not images_dir.exists():
        return []
    return [
        (image_path, labels_dir / f"{image_path.stem}.txt")
        for image_path in sorted(images_dir.rglob("*"))
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
    ]


def _lineage_key(source: SourceSpec, image_path: Path) -> str:
    """Return an ambiguous pre-export filename key for audit reporting only.

    Multiple source videos can reuse names such as ``frame_0s``. This key must
    never be treated as proof that images share a visual source frame.
    """

    original_stem = ROBOFLOW_VARIANT_SUFFIX.sub("", image_path.stem)
    return f"{source.name}:{original_stem.lower()}"


def build_dataset(
    target_names: Sequence[str],
    sources: Sequence[SourceSpec],
    output_root: Path,
) -> Dict[str, object]:
    """Create a de-duplicated YOLO dataset while preserving source splits."""

    target_id = {name: index for index, name in enumerate(target_names)}
    seen_hashes: Dict[str, str] = {}
    stats: Counter = Counter()
    class_counts: Counter = Counter()
    source_counts: Counter = Counter()

    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    candidates: List[Tuple[SourceSpec, str, Path, Path, str]] = []
    lineage_splits: Dict[str, set] = {}

    # Discover every candidate before copying. Pre-export names are retained for
    # audit reporting, but are not de-duplication evidence: unrelated videos can
    # reuse generic names such as frame_0s.
    for source in sources:
        if not source.root.exists():
            raise DatasetBuildError(f"Source root does not exist: {source.root}")
        discovered_any = False
        for source_split in ("test", "valid", "val", "train"):
            output_split = SPLIT_ALIASES[source_split]
            pairs = list(_discover_split(source, source_split))
            if not pairs:
                continue
            if source_split == "val" and (source.root / "valid" / "images").exists():
                continue
            discovered_any = True
            for image_path, label_path in pairs:
                lineage = _lineage_key(source, image_path)
                candidates.append((source, output_split, image_path, label_path, lineage))
                lineage_splits.setdefault(lineage, set()).add(output_split)
                stats["images_seen"] += 1
        if not discovered_any:
            raise DatasetBuildError(
                f"Source {source.name} has no train/valid/val/test images directories"
            )

    stats["ambiguous_cross_split_filename_keys"] = sum(
        1 for splits in lineage_splits.values() if len(splits) > 1
    )
    stats["images_with_ambiguous_cross_split_filename_keys"] = sum(
        1
        for _source, _split, _image, _label, lineage in candidates
        if len(lineage_splits[lineage]) > 1
    )

    # Evaluation splits are processed first so an exact duplicate appearing in
    # multiple sources cannot be retained in train while skipped from test/val.
    candidates.sort(
        key=lambda item: (
            -SPLIT_PRIORITY[item[1]],
            item[0].name,
            str(item[2]),
        )
    )
    for source, output_split, image_path, label_path, lineage in candidates:
        if not label_path.exists():
            stats["missing_label_files"] += 1
            continue
        image_hash = _sha256(image_path)
        if image_hash in seen_hashes:
            stats["duplicate_images_skipped"] += 1
            continue
        try:
            annotations, dropped, original_count = parse_label_lines(
                label_path.read_text(encoding="utf-8").splitlines(), source
            )
        except DatasetBuildError as exc:
            raise DatasetBuildError(f"{label_path}: {exc}") from exc
        stats["annotations_dropped_by_mapping"] += dropped
        if original_count > 0 and not annotations:
            # Treating a target-containing image as a negative after all
            # labels were dropped would introduce false-negative labels.
            stats["images_skipped_after_all_labels_dropped"] += 1
            continue

        safe_source = "".join(
            character if character.isalnum() else "-" for character in source.name
        ).strip("-").lower()
        output_stem = f"{safe_source}__{image_path.stem}__{image_hash[:10]}"
        output_image = output_root / "images" / output_split / (
            output_stem + image_path.suffix.lower()
        )
        output_label = output_root / "labels" / output_split / (output_stem + ".txt")
        shutil.copy2(image_path, output_image)
        output_label.write_text(
            "\n".join(
                annotation.yolo_line(target_id[annotation.target_class])
                for annotation in annotations
            )
            + ("\n" if annotations else ""),
            encoding="utf-8",
        )
        seen_hashes[image_hash] = str(image_path)
        stats[f"images_{output_split}"] += 1
        source_counts[source.name] += 1
        for annotation in annotations:
            class_counts[annotation.target_class] += 1
            stats["annotations_retained"] += 1

    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(target_names))
    (output_root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + names_yaml
        + "\n",
        encoding="utf-8",
    )
    report: Dict[str, object] = {
        "schema_version": 1,
        "target_names": list(target_names),
        "statistics": dict(sorted(stats.items())),
        "class_annotation_counts": dict(sorted(class_counts.items())),
        "source_image_counts": dict(sorted(source_counts.items())),
        "source_provenance": [
            {
                "name": source.name,
                "license": source.license_name,
                "source_url": source.source_url,
            }
            for source in sources
        ],
        "split_warning": (
            "Source-provided splits were preserved. Exact duplicates are retained "
            "only in the highest-priority split (test > val > train). Roboflow "
            "pre-export filename collisions are reported but never used as proof "
            "of shared lineage because multiple videos reuse names such as frame_0s. "
            "Review visual near-duplicates and adjacent source frames separately."
        ),
    }
    (output_root / "build-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
