#!/usr/bin/env python3
"""Download fixed public smoke inputs and enforce their recorded checksums."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/evaluation/manifest.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.root.resolve()
    for item in payload["inputs"]:
        destination = (root / item["path"]).resolve()
        if root != destination and root not in destination.parents:
            raise ValueError(f"Input path escapes root: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha256(destination) != item["sha256"]:
            urllib.request.urlretrieve(item["url"], destination)
        actual = sha256(destination)
        if actual != item["sha256"]:
            raise ValueError(
                f"Checksum mismatch for {destination}: expected {item['sha256']}, got {actual}"
            )
        print({"id": item["id"], "path": str(destination), "sha256": actual})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
