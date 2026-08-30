from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_pi.pi_benchmark import compile_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile saved Pi runs into CSV and Markdown")
    parser.add_argument("--matrix", type=Path, default=Path("configs/pi-benchmark-matrix.json"))
    parser.add_argument("--registry", type=Path, default=Path("configs/pi-model-registry.json"))
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_tables(args.matrix, args.registry, args.runs, args.output), indent=2))


if __name__ == "__main__":
    main()
