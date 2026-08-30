"""Read-only Colab-side inventory of uploaded training archive chunks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


chunks = sorted(Path("/content").glob("cane-v1-training-640.tar.part-*"))
print(
    json.dumps(
        {
            "count": len(chunks),
            "chunks": [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in chunks
            ],
        },
        indent=2,
        sort_keys=True,
    )
)
