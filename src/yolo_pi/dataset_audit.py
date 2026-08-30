from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(path: Path, hash_size: int = 8) -> int:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised only without Pillow
        raise RuntimeError("Pillow is required for perceptual dataset auditing") from exc

    with Image.open(path) as image:
        resized = image.convert("L").resize((hash_size + 1, hash_size))
        pixel_source = (
            resized.get_flattened_data()
            if hasattr(resized, "get_flattened_data")
            else resized.getdata()
        )
        pixels = list(pixel_source)
    value = 0
    row_width = hash_size + 1
    for row in range(hash_size):
        for column in range(hash_size):
            value = (value << 1) | int(
                pixels[row * row_width + column]
                > pixels[row * row_width + column + 1]
            )
    return value


def _source_name(path: Path) -> str:
    return path.name.split("__", 1)[0] if "__" in path.name else "unknown"


def _load_class_names(dataset_root: Path) -> List[str]:
    report_path = dataset_root / "build-report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        names = report.get("target_names")
        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            return names
    raise ValueError(
        f"{report_path} must contain the ordered target_names list for auditing"
    )


def _iter_images(dataset_root: Path) -> Iterable[Tuple[str, Path]]:
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        if not image_dir.exists():
            continue
        for path in sorted(image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield split, path


def _label_path(dataset_root: Path, split: str, image_path: Path) -> Path:
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def _validate_label_file(path: Path, class_count: int) -> Tuple[Counter, List[str]]:
    counts: Counter = Counter()
    errors: List[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 5:
            errors.append(f"line {line_number}: expected 5 tokens, found {len(tokens)}")
            continue
        try:
            class_id = int(tokens[0])
            x_center, y_center, width, height = (float(value) for value in tokens[1:])
        except ValueError:
            errors.append(f"line {line_number}: non-numeric YOLO field")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"line {line_number}: class {class_id} outside [0,{class_count})")
            continue
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            errors.append(f"line {line_number}: center outside normalized bounds")
            continue
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            errors.append(f"line {line_number}: box size outside normalized bounds")
            continue
        counts[class_id] += 1
    return counts, errors


def _near_duplicate_pairs(
    records: Sequence[Dict[str, object]], threshold: int
) -> List[Dict[str, object]]:
    if threshold < 0 or threshold >= 64:
        raise ValueError("perceptual Hamming threshold must be in [0, 63]")

    # Pigeonhole indexing: hashes within threshold bits must share at least one
    # exact partition when split into threshold+1 pieces.
    partitions = threshold + 1
    base_width, remainder = divmod(64, partitions)
    slices: List[Tuple[int, int]] = []
    offset = 0
    for index in range(partitions):
        width = base_width + int(index < remainder)
        slices.append((offset, width))
        offset += width

    buckets: List[Dict[int, List[int]]] = [defaultdict(list) for _ in slices]
    checked_pairs = set()
    candidates: List[Dict[str, object]] = []
    for right_index, right in enumerate(records):
        right_hash = int(right["dhash64"])
        possible = set()
        for partition_index, (shift, width) in enumerate(slices):
            key = (right_hash >> shift) & ((1 << width) - 1)
            possible.update(buckets[partition_index].get(key, []))
        for left_index in possible:
            pair = (left_index, right_index)
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            left = records[left_index]
            if left["split"] == right["split"] or left["sha256"] == right["sha256"]:
                continue
            distance = (int(left["dhash64"]) ^ right_hash).bit_count()
            if distance <= threshold:
                candidates.append(
                    {
                        "hamming_distance": distance,
                        "left_split": left["split"],
                        "left_path": left["path"],
                        "right_split": right["split"],
                        "right_path": right["path"],
                    }
                )
        for partition_index, (shift, width) in enumerate(slices):
            key = (right_hash >> shift) & ((1 << width) - 1)
            buckets[partition_index][key].append(right_index)
    return sorted(
        candidates,
        key=lambda item: (
            int(item["hamming_distance"]),
            str(item["left_path"]),
            str(item["right_path"]),
        ),
    )


def audit_dataset(
    dataset_root: Path,
    *,
    perceptual_threshold: int = 4,
    max_reported_candidates: int = 1000,
) -> Dict[str, object]:
    dataset_root = dataset_root.resolve()
    class_names = _load_class_names(dataset_root)
    started = time.time()
    records: List[Dict[str, object]] = []
    split_counts: Counter = Counter()
    source_counts: Counter = Counter()
    class_counts: Counter = Counter()
    missing_labels: List[str] = []
    invalid_labels: List[Dict[str, object]] = []
    unreadable_images: List[Dict[str, str]] = []
    paired_label_paths = set()

    for split, image_path in _iter_images(dataset_root):
        relative_path = image_path.relative_to(dataset_root).as_posix()
        label_path = _label_path(dataset_root, split, image_path)
        split_counts[split] += 1
        source_counts[_source_name(image_path)] += 1
        if not label_path.exists():
            missing_labels.append(relative_path)
        else:
            paired_label_paths.add(label_path.resolve())
            counts, errors = _validate_label_file(label_path, len(class_names))
            class_counts.update(counts)
            if errors:
                invalid_labels.append(
                    {
                        "path": label_path.relative_to(dataset_root).as_posix(),
                        "errors": errors,
                    }
                )
        try:
            records.append(
                {
                    "split": split,
                    "path": relative_path,
                    "sha256": _sha256(image_path),
                    "dhash64": _dhash(image_path),
                }
            )
        except Exception as exc:  # a corrupt image must not abort the entire audit
            unreadable_images.append({"path": relative_path, "error": str(exc)})

    orphan_labels: List[str] = []
    for split in SPLITS:
        label_dir = dataset_root / "labels" / split
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.glob("*.txt")):
            if path.resolve() not in paired_label_paths:
                orphan_labels.append(path.relative_to(dataset_root).as_posix())

    by_sha: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        by_sha[str(record["sha256"])].append(record)
    exact_groups = []
    for digest, group in sorted(by_sha.items()):
        if len(group) > 1:
            exact_groups.append(
                {
                    "sha256": digest,
                    "splits": sorted({str(item["split"]) for item in group}),
                    "paths": sorted(str(item["path"]) for item in group),
                }
            )
    cross_split_exact = [group for group in exact_groups if len(group["splits"]) > 1]
    near_pairs = _near_duplicate_pairs(records, perceptual_threshold)

    hard_error_count = (
        len(missing_labels)
        + len(invalid_labels)
        + len(unreadable_images)
        + len(orphan_labels)
        + len(cross_split_exact)
    )
    return {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "audit_scope": "read-only structural, exact-hash, and dHash candidate audit",
        "duration_seconds": round(time.time() - started, 3),
        "class_names": class_names,
        "image_counts_by_split": {split: split_counts[split] for split in SPLITS},
        "image_counts_by_source": dict(sorted(source_counts.items())),
        "annotation_counts_by_class": {
            name: class_counts[index] for index, name in enumerate(class_names)
        },
        "structural": {
            "missing_label_count": len(missing_labels),
            "missing_labels": missing_labels[:1000],
            "invalid_label_file_count": len(invalid_labels),
            "invalid_label_files": invalid_labels[:1000],
            "orphan_label_count": len(orphan_labels),
            "orphan_labels": orphan_labels[:1000],
            "unreadable_image_count": len(unreadable_images),
            "unreadable_images": unreadable_images[:1000],
        },
        "exact_duplicates": {
            "group_count": len(exact_groups),
            "cross_split_group_count": len(cross_split_exact),
            "cross_split_groups": cross_split_exact[:1000],
        },
        "perceptual_candidates": {
            "method": "64-bit difference hash (dHash)",
            "hamming_threshold": perceptual_threshold,
            "cross_split_candidate_count": len(near_pairs),
            "reported_count": min(len(near_pairs), max_reported_candidates),
            "truncated": len(near_pairs) > max_reported_candidates,
            "pairs": near_pairs[:max_reported_candidates],
            "interpretation": "Candidates require visual review; dHash similarity is not proof of shared lineage.",
        },
        "quality_gate": {
            "hard_error_count": hard_error_count,
            "structural_and_exact_gate_passed": hard_error_count == 0,
            "visual_review_required": bool(near_pairs),
        },
    }


def write_audit(report: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)


def write_candidate_contact_sheet(
    report: Dict[str, object],
    output_path: Path,
    *,
    max_pairs: int = 24,
    start_pair: int = 0,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to render the contact sheet") from exc

    root = Path(str(report["dataset_root"]))
    all_pairs = report["perceptual_candidates"]["pairs"]  # type: ignore[index]
    pairs = all_pairs[start_pair : start_pair + max_pairs]
    cell_width, image_height, label_height = 720, 220, 54
    sheet = Image.new("RGB", (cell_width, max(1, len(pairs)) * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, pair in enumerate(pairs):
        top = row * (image_height + label_height)
        for column, side in enumerate(("left", "right")):
            path = root / str(pair[f"{side}_path"])
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((cell_width // 2, image_height))
                x = column * (cell_width // 2) + (cell_width // 2 - image.width) // 2
                sheet.paste(image, (x, top + (image_height - image.height) // 2))
        draw.text(
            (8, top + image_height + 4),
            f"dHash distance={pair['hamming_distance']} | {pair['left_split']} vs {pair['right_split']}",
            fill="black",
        )
        draw.text(
            (8, top + image_height + 24),
            f"{Path(str(pair['left_path'])).name[:48]}  |  {Path(str(pair['right_path'])).name[:48]}",
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def decontaminate_dataset(
    source_root: Path,
    output_root: Path,
    audit_report: Dict[str, object],
) -> Dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    if Path(str(audit_report["dataset_root"])).resolve() != source_root:
        raise ValueError("audit report dataset_root does not match source dataset")

    priority = {"train": 0, "val": 1, "test": 2}
    removals: Dict[str, Dict[str, object]] = {}
    cross_source_ignored = 0
    pairs = audit_report["perceptual_candidates"]["pairs"]  # type: ignore[index]
    for pair in pairs:
        left_path = str(pair["left_path"])
        right_path = str(pair["right_path"])
        if _source_name(Path(left_path)) != _source_name(Path(right_path)):
            cross_source_ignored += 1
            continue
        left_split, right_split = str(pair["left_split"]), str(pair["right_split"])
        if priority[left_split] == priority[right_split]:
            continue
        removed_path, kept_path = (
            (left_path, right_path)
            if priority[left_split] < priority[right_split]
            else (right_path, left_path)
        )
        entry = removals.setdefault(
            removed_path,
            {
                "removed_path": removed_path,
                "kept_higher_priority_paths": [],
                "minimum_dhash_distance": int(pair["hamming_distance"]),
            },
        )
        entry["kept_higher_priority_paths"].append(kept_path)
        entry["minimum_dhash_distance"] = min(
            int(entry["minimum_dhash_distance"]), int(pair["hamming_distance"])
        )

    temporary_root = output_root.with_name(output_root.name + ".tmp")
    if temporary_root.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary_root}")
    link_modes: Counter = Counter()
    retained_counts: Counter = Counter()
    try:
        for split in SPLITS:
            (temporary_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (temporary_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for split, image_path in _iter_images(source_root):
            relative_image = image_path.relative_to(source_root).as_posix()
            if relative_image in removals:
                continue
            destination_image = temporary_root / relative_image
            link_modes[_link_or_copy(image_path, destination_image)] += 1
            label_path = _label_path(source_root, split, image_path)
            if label_path.exists():
                relative_label = label_path.relative_to(source_root)
                link_modes[_link_or_copy(label_path, temporary_root / relative_label)] += 1
            retained_counts[split] += 1
        for metadata_name in (
            "data.yaml",
            "build-report.json",
            "resize-report.json",
        ):
            metadata_path = source_root / metadata_name
            if metadata_path.exists():
                shutil.copy2(metadata_path, temporary_root / metadata_name)
        report = {
            "schema_version": 1,
            "source_dataset": str(source_root),
            "output_dataset": str(output_root),
            "policy": "same-source cross-split dHash candidate; retain test > val > train",
            "perceptual_method": audit_report["perceptual_candidates"]["method"],  # type: ignore[index]
            "hamming_threshold": audit_report["perceptual_candidates"]["hamming_threshold"],  # type: ignore[index]
            "removed_image_count": len(removals),
            "retained_image_counts_by_split": {
                split: retained_counts[split] for split in SPLITS
            },
            "cross_source_candidate_pairs_ignored": cross_source_ignored,
            "materialization": dict(sorted(link_modes.items())),
            "removals": sorted(removals.values(), key=lambda item: str(item["removed_path"])),
            "warning": "dHash candidates were conservatively decontaminated, not asserted as semantic duplicates. Source dataset remains unchanged.",
        }
        (temporary_root / "decontamination-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_root.replace(output_root)
        return report
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise
