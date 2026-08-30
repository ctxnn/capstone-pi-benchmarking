from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_pi.training_dataset import resize_training_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize a YOLO dataset for bounded GPU transfer")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dimension", type=int, default=640)
    args = parser.parse_args()
    print(
        json.dumps(
            resize_training_dataset(
                args.dataset, args.output, max_dimension=args.max_dimension
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
