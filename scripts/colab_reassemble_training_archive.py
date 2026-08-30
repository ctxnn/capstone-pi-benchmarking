"""Colab-side reassembly and verification for the Cane V1 training archive."""

from __future__ import annotations

import hashlib
from pathlib import Path


PART_GLOB = "/content/cane-v1-training-640.tar.part-*"
OUTPUT = Path("/content/cane-v1-training-640.tar")
EXPECTED_PARTS = 39
EXPECTED_SHA256 = "f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91"


parts = sorted(Path("/content").glob("cane-v1-training-640.tar.part-*"))
if len(parts) != EXPECTED_PARTS:
    raise RuntimeError(f"expected {EXPECTED_PARTS} chunks, found {len(parts)}")

digest = hashlib.sha256()
with OUTPUT.open("wb") as destination:
    for part in parts:
        with part.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                destination.write(block)
                digest.update(block)

actual = digest.hexdigest()
if actual != EXPECTED_SHA256:
    raise RuntimeError(f"archive checksum mismatch: {actual}")
print(
    {
        "archive": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "parts": len(parts),
        "sha256": actual,
        "verified": True,
    }
)
