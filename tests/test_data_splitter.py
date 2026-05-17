import unittest
import os
import shutil
from pathlib import Path
# 🟢 Logic: Correctly import the existing function from your source file
from src.preprocessing.test_data_splitter import split_processed_data

class TestSplitter(unittest.TestCase):
    def setUp(self):
        """Logic: Setup a fake 'all' folder with dummy images/labels for testing."""
        self.test_root = Path("test_data_temp")
        self.input_dir = self.test_root / "all"
        
        # Create dummy directories
        (self.input_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.input_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Create 10 dummy files to test the 70/20/10 split
        for i in range(10):
            (self.input_dir / "images" / f"img_{i}.png").touch()
            (self.input_dir / "labels" / f"img_{i}.txt").touch()

    def test_split_ratios(self):
        """Logic Check: Verify 70% goes to train (7 out of 10 images)."""
        # 🟢 Logic: Call your actual function with the temp test paths
        split_processed_data(input_dir=str(self.input_dir), output_dir=str(self.test_root))
        
        train_imgs = list((self.test_root / "train" / "images").glob("*.png"))
        
        # Assert that the math is correct (10 images * 0.7 = 7)
        self.assertEqual(len(train_imgs), 7, f"Expected 7 images in train, but found {len(train_imgs)}")
        print(f"\n✅ Data Splitting Logic: PASSED (Found {len(train_imgs)} images in train split)")

    def tearDown(self):
        """Logic: Cleanup the temporary test folder after test finishes."""
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

if __name__ == "__main__":
    unittest.main()