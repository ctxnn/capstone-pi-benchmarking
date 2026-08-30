#!/usr/bin/env python3
"""Select a deterministic, class-covering held-out fixture for Pi timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Root used for portable manifest paths (default: current directory)",
    )
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    workspace_root = args.workspace_root.resolve()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    build = json.loads((args.dataset / "build-report.json").read_text())
    names: List[str] = build["target_names"]
    records: List[Tuple[Path, Set[int]]] = []
    for image in sorted((args.dataset / "images" / "test").glob("*.jpg")):
        label = args.dataset / "labels" / "test" / f"{image.stem}.txt"
        classes = {
            int(line.split()[0])
            for line in label.read_text().splitlines()
            if line.strip()
        }
        invalid_classes = sorted(class_id for class_id in classes if not 0 <= class_id < len(names))
        if invalid_classes:
            raise ValueError(f"{label} contains invalid class IDs: {invalid_classes}")
        records.append((image, classes))

    if len(records) < args.count:
        raise ValueError(
            f"requested {args.count} fixtures but test split contains only {len(records)} images"
        )

    populated = set().union(*(classes for _, classes in records))
    selected: List[Tuple[Path, Set[int]]] = []
    selected_paths = set()
    for class_id in sorted(populated):
        candidates = [record for record in records if class_id in record[1] and record[0] not in selected_paths]
        if candidates:
            choice = max(candidates, key=lambda record: (len(record[1]), -records.index(record)))
            selected.append(choice)
            selected_paths.add(choice[0])
    empty = [record for record in records if not record[1]]
    for record in empty[: min(3, max(0, args.count - len(selected)))]:
        selected.append(record)
        selected_paths.add(record[0])
    remaining = [record for record in records if record[0] not in selected_paths]
    slots = max(0, args.count - len(selected))
    if slots and remaining:
        step = len(remaining) / slots
        for index in range(slots):
            record = remaining[min(int(index * step), len(remaining) - 1)]
            if record[0] not in selected_paths:
                selected.append(record)
                selected_paths.add(record[0])
    selected = selected[: args.count]
    if len(selected) != args.count:
        raise RuntimeError(f"selected {len(selected)} fixtures, expected {args.count}")

    image_output = args.output / "images"
    image_output.mkdir(parents=True)
    entries: List[Dict[str, object]] = []
    for index, (source, classes) in enumerate(selected):
        destination = image_output / f"fixture-{index:03d}.jpg"
        shutil.copy2(source, destination)
        try:
            relative = destination.resolve().relative_to(workspace_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"fixture output {destination} must be inside workspace root {workspace_root}"
            ) from exc
        entries.append(
            {
                "id": destination.stem,
                "path": relative,
                "sha256": sha256(destination),
                "expected_visible_classes": [names[class_id] for class_id in sorted(classes)],
                "original_test_path": source.relative_to(args.dataset).as_posix(),
                "source_dataset": source.name.split("__", 1)[0],
                "conditions": ["held-out-test", "fixed-runtime-fixture"],
            }
        )
    manifest = {
        "schema_version": 1,
        "purpose": "Fixed held-out Cane V1 inputs for comparable Pi runtime/postprocessing measurements; not the accuracy evaluation itself",
        "dataset": str(args.dataset.resolve()),
        "selection": "one deterministic class-covering image per populated class, up to three empty-label images, then evenly spaced held-out test images",
        "input_count": len(entries),
        "covered_classes": [names[class_id] for class_id in sorted(set().union(*(classes for _, classes in selected)))],
        "inputs": entries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
