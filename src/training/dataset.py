import yaml
import os
from pathlib import Path

class LegoDataset:
    """
    Complete logic for dataset management, validation, and health checks.
    Ensures the training pipeline only starts if the data is structurally sound.
    """

    def __init__(self, dataset_config_path="configs/dataset.yaml"):
        """
        Initializes the dataset manager by loading the YAML configuration.
        """
        self.config_path = Path(dataset_config_path)
        self.config = self._load_config()
        self.base_path = Path(self.config.get('path', ''))

    def _load_config(self):
        """Loads the YAML file with error handling."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"❌ Config file not found at: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def validate_structure(self):
        """
        Performs a deep integrity check of the processed data.
        Verifies:
        1. Folders exist.
        2. Image and Label counts match.
        3. Classes are defined.
        """
        print(f"🔍 Validating Dataset at: {self.base_path.resolve()}")
        
        splits = ['train', 'val', 'test']
        report = {}

        for split in splits:
            img_path = self.base_path / self.config.get(split, "")
            # Assuming labels are in a sibling folder to images as per YOLO standard
            lbl_path = img_path.parent / "labels"

            if not img_path.exists():
                print(f"⚠️  Missing split: {split} (expected at {img_path})")
                continue

            images = list(img_path.glob("*.png"))
            labels = list(lbl_path.glob("*.txt"))

            if len(images) != len(labels):
                print(f"🚨 WARNING: {split} mismatch! Images: {len(images)}, Labels: {len(labels)}")
            
            report[split] = len(images)

        # Class verification
        nc = self.config.get('nc', 0)
        names = self.config.get('names', [])
        
        if nc != len(names):
            print(f"🚨 ERROR: nc ({nc}) does not match length of names ({len(names)})")
            return False

        print("📊 Dataset Summary:")
        for split, count in report.items():
            print(f"   - {split.capitalize()}: {count} samples")
            
        return all(split in report for split in ['train', 'val'])

    def get_config(self):
        """Returns the raw config dictionary."""
        return self.config

    def get_split_paths(self):
        """Returns resolved absolute paths for all data splits."""
        return {
            split: (self.base_path / self.config[split]).resolve()
            for split in ['train', 'val', 'test']
        }

if __name__ == "__main__":
    # Standalone execution to test dataset health
    try:
        ds_manager = LegoDataset()
        is_healthy = ds_manager.validate_structure()
        if is_healthy:
            print("✅ Dataset logic is 100% connected and verified.")
        else:
            print("❌ Dataset logic failed validation.")
    except Exception as e:
        print(f"💥 Runtime Error: {e}")