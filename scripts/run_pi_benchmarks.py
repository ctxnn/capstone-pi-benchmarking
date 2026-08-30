from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_pi.pi_benchmark import run_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable Raspberry Pi benchmarks")
    parser.add_argument("--matrix", type=Path, default=Path("configs/pi-benchmark-matrix.json"))
    parser.add_argument("--registry", type=Path, default=Path("configs/pi-model-registry.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", action="append", dest="rows")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-non-pi", action="store_true")
    parser.add_argument("--power-jsonl", type=Path)
    parser.add_argument("--source", choices=("images", "camera"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--camera-fps", type=float)
    parser.add_argument("--camera-num", type=int)
    parser.add_argument("--camera-pixel-format")
    args = parser.parse_args()
    camera_config = {
        key: value
        for key, value in {
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
            "camera_num": args.camera_num,
            "pixel_format": args.camera_pixel_format,
        }.items()
        if value is not None
    }
    outcomes = run_matrix(
        args.matrix,
        args.registry,
        args.output,
        row_ids=args.rows,
        force=args.force,
        allow_non_pi=args.allow_non_pi,
        power_jsonl=args.power_jsonl,
        source_type=args.source,
        camera_config=camera_config,
        threads=args.threads,
    )
    print(json.dumps(outcomes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
