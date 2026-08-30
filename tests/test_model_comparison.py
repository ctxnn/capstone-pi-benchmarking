import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_model_comparison.py"
SPEC = importlib.util.spec_from_file_location("evaluate_model_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ModelComparisonTests(unittest.TestCase):
    def test_data_yaml_root_is_made_absolute_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "dataset" / "data.yaml"
            source.parent.mkdir()
            original = "path: .\ntrain: images/train\ntest: images/test\nnames: [person]\n"
            source.write_text(original, encoding="utf-8")
            destination = root / "runtime.yaml"

            MODULE.absolute_data_yaml(source, destination)

            self.assertEqual(source.read_text(encoding="utf-8"), original)
            self.assertIn(f"path: {source.parent.resolve()}", destination.read_text())

    def test_mapping_uses_names_not_incompatible_numeric_ids(self) -> None:
        self.assertEqual(MODULE.COCO_TO_CANE["car"], "vehicle")
        self.assertEqual(MODULE.COCO_TO_CANE["bicycle"], "bicycle_motorcycle")
        self.assertNotIn("chair", MODULE.COCO_TO_CANE)
        self.assertEqual(MODULE.names_dict(["person", "vehicle"]), {0: "person", 1: "vehicle"})


if __name__ == "__main__":
    unittest.main()
