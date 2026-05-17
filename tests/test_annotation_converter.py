import unittest
import numpy as np
from src.preprocessing.annotation_converter import get_polygon_from_mask
from pathlib import Path
import cv2

class TestAnnotation(unittest.TestCase):
    def test_polygon_normalization(self):
        """Logic Check: Ensure all polygon coordinates are between 0 and 1."""
        # Create a dummy 100x100 black mask with a 50x50 white square
        dummy_mask = np.zeros((100, 100), dtype=np.uint8)
        dummy_mask[25:75, 25:75] = 255
        mask_path = "test_mask.png"
        cv2.imwrite(mask_path, dummy_mask)

        try:
            polygons = get_polygon_from_mask(mask_path)
            for poly in polygons:
                for coord in poly:
                    self.assertTrue(0.0 <= coord <= 1.0, f"Coordinate {coord} is not normalized!")
            print("✅ Annotation Normalization Logic: PASSED")
        finally:
            if Path(mask_path).exists():
                Path(mask_path).unlink()

if __name__ == "__main__":
    unittest.main()