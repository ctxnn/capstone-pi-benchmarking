#!/usr/bin/env python3
"""Build the combined V1 dataset from downloaded YOLO-format exports."""

import argparse
import json
from pathlib import Path

from yolo_pi.dataset import build_dataset, load_build_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dataset-sources.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/cane-v1"))
    args = parser.parse_args()
    target_names, sources = load_build_config(args.config)
    report = build_dataset(target_names, sources, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
