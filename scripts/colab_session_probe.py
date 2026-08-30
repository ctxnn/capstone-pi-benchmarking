"""Read-only Colab-side probe for assignment, filesystem, and detached trainer state."""

from __future__ import annotations

import json
from pathlib import Path


content = Path("/content")
pid_path = content / "cane-v1-resume.pid"
pid = None
if pid_path.exists():
    value = pid_path.read_text(encoding="utf-8").strip()
    pid = int(value) if value.isdigit() else None

paths = {
    "content": content,
    "dataset_archive": content / "cane-v1-training-640.tar",
    "resume_checkpoint": content / "cane-v1-resume-epoch15.pt",
    "run_last": content
    / "yolo-pi-runs/cane-v1-yolo26n-e30-img416/weights/last.pt",
    "run_best": content
    / "yolo-pi-runs/cane-v1-yolo26n-e30-img416/weights/best.pt",
    "training_log": content / "cane-v1-resume-training-pinned.log",
    "final_archive": content / "yolo-pi-colab-artifacts.zip",
}
report = {
    "paths": {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.is_file() else None,
        }
        for name, path in paths.items()
    },
    "trainer_pid": pid,
    "trainer_alive": bool(pid and Path(f"/proc/{pid}").exists()),
}
print(json.dumps(report, indent=2, sort_keys=True))
