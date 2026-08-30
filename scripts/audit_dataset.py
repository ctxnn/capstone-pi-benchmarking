from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_pi.dataset_audit import audit_dataset, write_audit, write_candidate_contact_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a harmonized YOLO dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--hamming-threshold", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--max-sheet-pairs", type=int, default=24)
    parser.add_argument("--sheet-offset", type=int, default=0)
    args = parser.parse_args()

    report = audit_dataset(
        args.dataset,
        perceptual_threshold=args.hamming_threshold,
        max_reported_candidates=args.max_candidates,
    )
    write_audit(report, args.output)
    if args.contact_sheet and report["perceptual_candidates"]["pairs"]:
        write_candidate_contact_sheet(
            report,
            args.contact_sheet,
            max_pairs=args.max_sheet_pairs,
            start_pair=args.sheet_offset,
        )
    print(json.dumps(report["quality_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
