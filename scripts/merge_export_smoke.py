#!/usr/bin/env python3
"""Merge checksum-matched cross-host smoke evidence into an export manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    entry = manifest["exports"][args.format]
    if evidence.get("status") != "complete":
        raise ValueError("smoke evidence is not complete")
    if evidence.get("format") != args.format:
        raise ValueError("smoke evidence format mismatch")
    if evidence.get("imgsz") != manifest.get("imgsz"):
        raise ValueError("smoke evidence resolution mismatch")
    if evidence.get("precision") != manifest.get("precision"):
        raise ValueError("smoke evidence precision mismatch")
    if evidence["artifact"]["tree_sha256"] != entry.get("tree_sha256"):
        raise ValueError("smoke evidence artifact tree checksum mismatch")
    expected_backend = {
        "onnx": "onnxruntime-",
        "openvino": "openvino-",
        "mnn": "mnn-",
        "ncnn": "ncnn-",
        "litert": "litert-",
    }[args.format]
    if not evidence["smoke"]["backend"].startswith(expected_backend):
        raise ValueError("smoke evidence resolved the wrong runtime")

    entry["status"] = "complete"
    entry["smoke"] = {
        **evidence["smoke"],
        "validation_environment": evidence["environment"],
        "evidence_path": str(args.evidence),
        "evidence_sha256": sha256(args.evidence),
    }
    output = args.output or args.manifest
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
