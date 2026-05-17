import unittest
import os
import shutil
from pathlib import Path
from src.postprocessing.bbox_drawer import LegoBBoxDrawer

class TestBBoxDrawer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Runs once before any tests. 
        Ensures a clean environment for testing directory logic.
        """
        cls.target_dir = Path("outputs/predictions")
        cls.dummy_model = "outputs/checkpoints/best.pt"
        
        # Ensure checkpoint directory exists so initialization doesn't crash 
        # on missing folders before it even tries to create the prediction folder.
        Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)

    def setUp(self):
        """Runs before each individual test to ensure the target dir is gone."""
        if self.target_dir.exists():
            shutil.rmtree(self.target_dir)

    def test_output_directory_initialization(self):
        """
        Check: Does the class automatically create the prediction folder on init?
        """
        # We wrap in try/except because we are testing the __init__ logic,
        # not the actual YOLO model loading which might fail on a local CPU.
        try:
            LegoBBoxDrawer(model_path=self.dummy_model)
        except Exception:
            # We ignore model-loading errors to focus on the directory creation logic
            pass
        
        self.assertTrue(self.target_dir.exists(), "FAIL: Prediction directory was not created during initialization!")
        print("✅ Directory Initialization Logic: PASSED")

    def test_invalid_path_handling(self):
        """
        Check: Does the program handle a completely wrong model path correctly?
        """
        invalid_path = "wrong/path/to/model.pt"
        with self.assertRaises(FileNotFoundError):
            LegoBBoxDrawer(model_path=invalid_path)
        print("✅ Error Handling (Invalid Path): PASSED")

    def test_singleton_directory_logic(self):
        """
        Check: If the directory already exists, does the program handle it without error?
        """
        self.target_dir.mkdir(parents=True, exist_ok=True)
        try:
            LegoBBoxDrawer(model_path=self.dummy_model)
            error_occurred = False
        except Exception:
            error_occurred = True
        
        # It shouldn't crash just because the folder already exists
        self.assertFalse(error_occurred, "FAIL: System crashed when output directory already existed!")
        print("✅ Existing Directory Resilience: PASSED")

    @classmethod
    def tearDownClass(cls):
        """Cleanup: Optional - removes test artifacts after tests are done."""
        # Only uncomment the line below if you want the test to delete your folders after finishing.
        # shutil.rmtree(cls.target_dir, ignore_errors=True)
        pass

if __name__ == "__main__":
    unittest.main()
