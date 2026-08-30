from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_pi.dataset_audit import decontaminate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a higher-split-priority dataset from reviewed dHash candidates"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    report = decontaminate_dataset(args.dataset, args.output, audit)
    print(json.dumps({key: value for key, value in report.items() if key != "removals"}, indent=2))


if __name__ == "__main__":
    main()
