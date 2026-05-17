import unittest
import os
from pathlib import Path
from src.postprocessing.bbox_drawer import LegoBBoxDrawer

class TestBBoxDrawer(unittest.TestCase):
    def test_output_directory_creation(self):
        """Logic Check: Ensure the results directory is initialized properly."""
        # We don't need a real model for this structural test
        try:
            drawer = LegoBBoxDrawer(model_path="non_existent.pt")
        except FileNotFoundError:
            # We expect this error, but we want to see if the directory logic triggered
            pass
        
        res_dir = Path("data/results")
        self.assertTrue(res_dir.exists(), "Results directory was not created!")
        print("✅ BBox Drawer Directory Logic: PASSED")

if __name__ == "__main__":
    unittest.main()