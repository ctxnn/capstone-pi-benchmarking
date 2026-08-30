#!/usr/bin/env python3
"""Build one contiguous, checksum-traceable training history from checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_from_results(results: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    if "epoch" not in results:
        raise ValueError("checkpoint train_results has no epoch column")
    columns = list(results)
    lengths = {column: len(results[column]) for column in columns}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"train_results columns have inconsistent lengths: {lengths}")
    rows: List[Dict[str, Any]] = []
    for index in range(lengths["epoch"]):
        row = {column: results[column][index] for column in columns}
        epoch = int(row["epoch"])
        if float(row["epoch"]) != epoch or epoch <= 0:
            raise ValueError(f"invalid completed epoch value: {row['epoch']}")
        row["epoch"] = epoch
        rows.append(row)
    return rows


def metric_rows_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare replayed epochs while ignoring elapsed time on a replacement VM."""

    keys = (set(left) | set(right)) - {"time", "source_checkpoint"}
    if set(left) - {"time", "source_checkpoint"} != set(right) - {
        "time",
        "source_checkpoint",
    }:
        return False
    for key in keys:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            if not math.isclose(float(left_value), float(right_value), rel_tol=1e-9, abs_tol=1e-12):
                return False
        elif left_value != right_value:
            return False
    return True


def merge_rows(
    sources: Iterable[Tuple[str, Sequence[Mapping[str, Any]]]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: Dict[int, Dict[str, Any]] = {}
    duplicates: List[Dict[str, Any]] = []
    for source, rows in sources:
        for raw in rows:
            row = dict(raw)
            epoch = int(row["epoch"])
            row["source_checkpoint"] = source
            previous = accepted.get(epoch)
            if previous is not None:
                if not metric_rows_match(previous, row):
                    raise ValueError(
                        f"epoch {epoch} has divergent metrics in "
                        f"{previous['source_checkpoint']} and {source}"
                    )
                duplicates.append(
                    {
                        "epoch": epoch,
                        "discarded_source": previous["source_checkpoint"],
                        "accepted_source": source,
                        "policy": "identical metrics; keep last occurrence",
                    }
                )
            accepted[epoch] = row
    if not accepted:
        raise ValueError("no training history rows found")
    ordered = [accepted[epoch] for epoch in sorted(accepted)]
    expected = list(range(1, ordered[-1]["epoch"] + 1))
    actual = [row["epoch"] for row in ordered]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise ValueError(f"training history is not contiguous; missing epochs: {missing}")
    return ordered, duplicates


def checkpoint_record(path: Path, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    model = checkpoint.get("ema") or checkpoint.get("model")
    names = {
        int(index): str(name)
        for index, name in getattr(model, "names", {}).items()
    }
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "completed_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "best_fitness": float(checkpoint.get("best_fitness", 0.0)),
        "has_optimizer": checkpoint.get("optimizer") is not None,
        "ultralytics_version": checkpoint.get("version"),
        "class_names": names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    import torch

    loaded = []
    sources = []
    expected_names = None
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        record = checkpoint_record(path, checkpoint)
        if expected_names is None:
            expected_names = record["class_names"]
        elif record["class_names"] != expected_names:
            raise ValueError(f"checkpoint taxonomy mismatch: {path}")
        loaded.append(record)
        sources.append((str(path.resolve()), rows_from_results(checkpoint["train_results"])))

    rows, duplicates = merge_rows(sources)
    columns = [key for key in rows[0] if key != "source_checkpoint"] + ["source_checkpoint"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(args.output)

    manifest = {
        "schema_version": 1,
        "purpose": "accepted contiguous training trajectory reconstructed from immutable checkpoints",
        "created_at_epoch_ns": time.time_ns(),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "accepted_epoch_start": rows[0]["epoch"],
        "accepted_epoch_end": rows[-1]["epoch"],
        "accepted_epoch_count": len(rows),
        "class_names": expected_names,
        "checkpoint_sources": loaded,
        "deduplicated_replays": duplicates,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_json.replace(args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
