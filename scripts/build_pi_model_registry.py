#!/usr/bin/env python3
"""Build the Pi registry from immutable training metrics and export manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def artifact_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
        and item.name != ".DS_Store"
    )


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    root = path if path.is_dir() else path.parent
    for item in artifact_files(path):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_export_path(entry: Dict[str, Any], artifact_root: Path) -> Path:
    """Resolve a manifest artifact after transfer from a different host root."""

    recorded = Path(str(entry["path"]))
    candidates = [recorded]
    if not recorded.is_absolute():
        candidates.append(artifact_root / recorded)
    candidates.append(artifact_root / recorded.name)
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            expected = entry.get("tree_sha256")
            actual = tree_sha256(resolved)
            if expected and actual != expected:
                raise ValueError(
                    f"transferred export tree checksum mismatch: {resolved}"
                )
            return resolved
    raise FileNotFoundError(
        f"export artifact is missing locally; recorded path was {recorded}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--quality-comparison", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("configs/pi-model-registry.json"))
    args = parser.parse_args()
    root = Path.cwd()
    training = json.loads(args.training_summary.read_text(encoding="utf-8"))
    comparison = json.loads(args.quality_comparison.read_text(encoding="utf-8"))
    exports = json.loads(args.export_manifest.read_text(encoding="utf-8"))
    if comparison.get("split") != "test" or comparison.get("fraction") != 1.0:
        raise ValueError("quality comparison must cover the complete held-out test split")
    if comparison.get("imgsz") != 640:
        raise ValueError("quality comparison must use the 640-pixel Pi matrix resolution")

    artifact_root = args.export_manifest.parent
    pytorch = artifact_root / "best.pt"
    baseline_sha = sha256(args.baseline)
    fine_tuned_sha = sha256(pytorch)
    expected_hashes = {
        "v0-pytorch-fp32": baseline_sha,
        "cane-v1-pytorch-fp32": fine_tuned_sha,
    }
    if training["best_weights"]["sha256"] != fine_tuned_sha:
        raise ValueError("exported best.pt does not match the final training summary")
    if exports["source_model"]["sha256"] != fine_tuned_sha:
        raise ValueError("export manifest was not built from the selected final best.pt")

    def quality(model_key: str) -> Dict[str, Any]:
        model = comparison["models"][model_key]
        if model.get("sha256") != expected_hashes[model_key]:
            raise ValueError(f"quality comparison checkpoint mismatch for {model_key}")
        normalized = model.get("normalized", {})
        required = {"precision", "recall", "map50", "map50_95"}
        missing = sorted(required - set(normalized))
        if missing:
            raise ValueError(f"quality comparison for {model_key} is missing {missing}")
        return {
            "split": "test",
            "imgsz": comparison["imgsz"],
            "fraction": comparison["fraction"],
            **normalized,
        }
    def exported(format_name: str) -> Dict[str, Any]:
        entry = exports["exports"][format_name]
        if entry["status"] != "complete":
            raise ValueError(f"required export is incomplete: {format_name}")
        path = resolve_export_path(entry, artifact_root)
        return {
            "path": relative(path, root),
            "tree_sha256": entry["tree_sha256"],
        }

    registry = {
        "schema_version": 1,
        "training_summary": relative(args.training_summary, root),
        "quality_comparison": relative(args.quality_comparison, root),
        "export_manifest": relative(args.export_manifest, root),
        "artifacts": {
            "v0-pytorch-fp32": {
                "path": relative(args.baseline, root),
                "sha256": baseline_sha,
                "quality_metrics": quality("v0-pytorch-fp32"),
            },
            "cane-v1-pytorch-fp32": {
                "path": relative(pytorch, root),
                "sha256": fine_tuned_sha,
                "quality_metrics": quality("cane-v1-pytorch-fp32"),
            },
            "cane-v1-onnx-fp32": exported("onnx"),
            "cane-v1-openvino-fp32": exported("openvino"),
            "cane-v1-mnn-fp32": exported("mnn"),
            "cane-v1-ncnn-fp32": exported("ncnn"),
            "cane-v1-litert-fp32": exported("litert"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
