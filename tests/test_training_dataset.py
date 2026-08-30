import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yolo_pi.training_dataset import resize_training_dataset


class TrainingDatasetTests(unittest.TestCase):
    def test_resizes_images_and_preserves_normalized_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            output = Path(temporary) / "output"
            for split in ("train", "val", "test"):
                (source / "images" / split).mkdir(parents=True)
                (source / "labels" / split).mkdir(parents=True)
                Image.new("RGB", (100, 50), "red").save(source / "images" / split / "one.jpg")
                (source / "labels" / split / "one.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (source / "data.yaml").write_text("names: [object]\n")
            (source / "build-report.json").write_text(json.dumps({"target_names": ["object"]}))

            report = resize_training_dataset(source, output, max_dimension=64)
            with Image.open(output / "images" / "train" / "one.jpg") as image:
                self.assertEqual(image.size, (64, 32))
            self.assertEqual(
                (output / "labels" / "train" / "one.txt").read_text(),
                "0 0.5 0.5 0.2 0.2\n",
            )
            self.assertEqual(report["statistics"]["images_resized"], 3)


if __name__ == "__main__":
    unittest.main()
