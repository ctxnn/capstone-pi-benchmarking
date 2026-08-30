import unittest

from yolo_pi.sectors import assign_sectors, horizontal_sector
from yolo_pi.types import BoundingBox, Detection, Sector


class SectorTests(unittest.TestCase):
    def test_equal_thirds(self) -> None:
        self.assertEqual(horizontal_sector(10, 300), Sector.LEFT)
        self.assertEqual(horizontal_sector(150, 300), Sector.CENTER)
        self.assertEqual(horizontal_sector(290, 300), Sector.RIGHT)

    def test_boundary_values_belong_to_center(self) -> None:
        self.assertEqual(horizontal_sector(100, 300), Sector.CENTER)
        self.assertEqual(horizontal_sector(200, 300), Sector.CENTER)

    def test_assigns_without_mutating_original_detection(self) -> None:
        original = Detection(0, "person", 0.8, BoundingBox(0, 0, 30, 30))
        assigned = assign_sectors([original], frame_width=300)
        self.assertEqual(original.sector, Sector.UNKNOWN)
        self.assertEqual(assigned[0].sector, Sector.LEFT)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            horizontal_sector(1, 0)
        with self.assertRaises(ValueError):
            horizontal_sector(1, 10, center_fraction=1.0)


if __name__ == "__main__":
    unittest.main()
