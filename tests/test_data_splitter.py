import unittest
import shutil
from pathlib import Path
from src.preprocessing.test_data_splitter import split_processed_data

class TestSplitter(unittest.TestCase):
    def setUp(self):
        """Setup a controlled dummy environment with 10 pairs of files."""
        self.test_root = Path("test_data_temp")
        self.input_dir = self.test_root / "all"
        
        # Create dummy directories
        (self.input_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.input_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Create 10 dummy file pairs
        for i in range(10):
            (self.input_dir / "images" / f"img_{i}.png").touch()
            (self.input_dir / "labels" / f"img_{i}.txt").touch()

    def test_split_ratios_and_integrity(self):
        """Verify 70/20/10 split math and image-label pairing."""
        split_processed_data(input_dir=str(self.input_dir), output_dir=str(self.test_root))
        
        # 1. Check Train Split (70%)
        train_imgs = list((self.test_root / "train" / "images").glob("*.png"))
        self.assertEqual(len(train_imgs), 7, f"Expected 7 images in train, found {len(train_imgs)}")
        
        # 2. Check Val Split (20%)
        val_imgs = list((self.test_root / "val" / "images").glob("*.png"))
        self.assertEqual(len(val_imgs), 2, f"Expected 2 images in val, found {len(val_imgs)}")

        # 3. Check Referential Integrity (Does every image have a label?)
        for split in ['train', 'val', 'test']:
            imgs = list((self.test_root / split / "images").glob("*.png"))
            labels = list((self.test_root / split / "labels").glob("*.txt"))
            self.assertEqual(len(imgs), len(labels), f"Mismatched image-label count in {split} split!")
        
        print(f"✅ Data Splitting Logic & Integrity: PASSED")

    def tearDown(self):
        """Cleanup temporary files."""
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

if __name__ == "__main__":
    unittest.main()
