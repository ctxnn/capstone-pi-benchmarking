import json
import tempfile
import unittest
from pathlib import Path

from yolo_pi.dataset import (
    DatasetBuildError,
    SourceSpec,
    build_dataset,
    parse_label_lines,
)


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SourceSpec(
            name="fixture",
            root=Path("unused"),
            names=("Person", "Ignore"),
            class_map={"Person": "person", "Ignore": None},
            license_name="fixture",
            source_url="https://example.test",
        )

    def test_parses_detection_and_segmentation(self) -> None:
        annotations, dropped, original = parse_label_lines(
            [
                "0 0.5 0.5 0.2 0.4",
                "0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4",
                "1 0.5 0.5 0.1 0.1",
            ],
            self.source,
        )
        self.assertEqual(len(annotations), 2)
        self.assertEqual(dropped, 1)
        self.assertEqual(original, 3)
        self.assertAlmostEqual(annotations[1].center_x, 0.2)
        self.assertAlmostEqual(annotations[1].height, 0.3)

    def test_rejects_unmapped_or_out_of_bounds_labels(self) -> None:
        incomplete = SourceSpec(
            "bad",
            Path("unused"),
            ("Person",),
            {},
            "fixture",
            "https://example.test",
        )
        with self.assertRaises(DatasetBuildError):
            parse_label_lines(["0 0.5 0.5 0.2 0.2"], incomplete)
        with self.assertRaises(DatasetBuildError):
            parse_label_lines(["0 1.0 0.5 0.2 0.2"], self.source)

    def test_builds_dataset_and_removes_exact_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            for split in ("train", "valid"):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir(parents=True)
            (root / "train" / "images" / "a.jpg").write_bytes(b"same-image")
            (root / "train" / "labels" / "a.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            (root / "valid" / "images" / "b.jpg").write_bytes(b"same-image")
            (root / "valid" / "labels" / "b.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            source = SourceSpec(
                name="fixture",
                root=root,
                names=self.source.names,
                class_map=self.source.class_map,
                license_name="fixture",
                source_url="https://example.test",
            )
            report = build_dataset(("person",), [source], output)
            self.assertEqual(report["statistics"]["images_val"], 1)
            self.assertEqual(report["statistics"]["duplicate_images_skipped"], 1)
            self.assertEqual(report["class_annotation_counts"]["person"], 1)
            self.assertTrue((output / "data.yaml").exists())
            saved = json.loads((output / "build-report.json").read_text())
            self.assertEqual(saved["source_image_counts"]["fixture"], 1)

    def test_reports_ambiguous_filename_key_without_discarding_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            for split in ("train", "valid", "test"):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir(parents=True)
            variants = {
                "train": "frame_10s_jpg.rf.11111111111111111111111111111111",
                "valid": "frame_10s_jpg.rf.22222222222222222222222222222222",
                "test": "frame_10s_jpg.rf.33333333333333333333333333333333",
            }
            for split, stem in variants.items():
                (root / split / "images" / f"{stem}.jpg").write_bytes(
                    f"{split}-variant".encode()
                )
                (root / split / "labels" / f"{stem}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
            source = SourceSpec(
                name="fixture",
                root=root,
                names=self.source.names,
                class_map=self.source.class_map,
                license_name="fixture",
                source_url="https://example.test",
            )
            report = build_dataset(("person",), [source], output)
            statistics = report["statistics"]
            self.assertEqual(statistics["ambiguous_cross_split_filename_keys"], 1)
            self.assertEqual(
                statistics["images_with_ambiguous_cross_split_filename_keys"], 3
            )
            self.assertEqual(statistics["images_test"], 1)
            self.assertEqual(statistics["images_train"], 1)
            self.assertEqual(statistics["images_val"], 1)


if __name__ == "__main__":
    unittest.main()
