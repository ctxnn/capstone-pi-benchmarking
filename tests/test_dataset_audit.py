import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yolo_pi.dataset_audit import audit_dataset, decontaminate_dataset, write_audit


class DatasetAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for split in ("train", "val", "test"):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
        (self.root / "build-report.json").write_text(
            json.dumps({"target_names": ["person", "obstacle"]}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _image(self, split: str, name: str, color: tuple) -> Path:
        path = self.root / "images" / split / name
        Image.new("RGB", (16, 16), color).save(path)
        (self.root / "labels" / split / f"{path.stem}.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )
        return path

    def test_clean_dataset_passes_structural_gate(self) -> None:
        self._image("train", "source-a__one.png", (255, 0, 0))
        self._image("val", "source-b__two.png", (0, 255, 0))
        report = audit_dataset(self.root, perceptual_threshold=0)
        self.assertTrue(report["quality_gate"]["structural_and_exact_gate_passed"])
        self.assertEqual(report["annotation_counts_by_class"]["person"], 2)

    def test_cross_split_exact_duplicate_fails_gate(self) -> None:
        train = self._image("train", "source-a__one.png", (10, 20, 30))
        duplicate = self.root / "images" / "test" / "source-a__duplicate.png"
        duplicate.write_bytes(train.read_bytes())
        (self.root / "labels" / "test" / "source-a__duplicate.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )
        report = audit_dataset(self.root, perceptual_threshold=0)
        self.assertFalse(report["quality_gate"]["structural_and_exact_gate_passed"])
        self.assertEqual(report["exact_duplicates"]["cross_split_group_count"], 1)

    def test_atomic_json_output(self) -> None:
        self._image("train", "source-a__one.png", (255, 0, 0))
        report = audit_dataset(self.root, perceptual_threshold=0)
        output = self.root / "reports" / "audit.json"
        write_audit(report, output)
        self.assertEqual(json.loads(output.read_text())["schema_version"], 1)
        self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_decontamination_keeps_higher_priority_split(self) -> None:
        train = self._image("train", "source-a__one.png", (10, 20, 30))
        validation = self._image("val", "source-a__two.png", (10, 20, 31))
        audit = {
            "dataset_root": str(self.root.resolve()),
            "perceptual_candidates": {
                "method": "64-bit difference hash (dHash)",
                "hamming_threshold": 4,
                "pairs": [
                    {
                        "hamming_distance": 1,
                        "left_split": "train",
                        "left_path": train.relative_to(self.root).as_posix(),
                        "right_split": "val",
                        "right_path": validation.relative_to(self.root).as_posix(),
                    }
                ],
            },
        }
        output = self.root.parent / f"{self.root.name}-clean"
        try:
            report = decontaminate_dataset(self.root, output, audit)
            self.assertEqual(report["removed_image_count"], 1)
            self.assertFalse((output / train.relative_to(self.root)).exists())
            self.assertTrue((output / validation.relative_to(self.root)).exists())
        finally:
            if output.exists():
                import shutil

                shutil.rmtree(output)


if __name__ == "__main__":
    unittest.main()
