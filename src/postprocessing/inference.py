import os
import sys
from pathlib import Path
from ultralytics import YOLO

# Ensure project root is accessible
sys.path.append(str(Path(__file__).resolve().parents[2]))

class LegoPredictor:
    """
    Complete logic for running inference using the trained LEGO model.
    Handles loading, prediction, and result extraction.
    """
    def __init__(self, model_path="outputs/train/lego_run/weights/best.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Trained model not found at {model_path}. Did you finish training?")
        
        self.model = YOLO(model_path)
        print(f"🧠 Model loaded and ready for inference.")

    def predict(self, source, conf=0.25):
        """
        Runs the model on a single image or folder.
        :param source: Path to image or folder.
        :param conf: Confidence threshold (0.0 to 1.0).
        :return: List of Ultralytics Result objects.
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            save=False, # We handle saving manually in bbox_drawer
            stream=False
        )
        return results

if __name__ == "__main__":
    # Quick test logic
    predictor = LegoPredictor()
    # Replace with a real test image path from your data/processed/test/images folder
    test_img = "data/processed/test/images"
    if os.path.exists(test_img):
        res = predictor.predict(test_img)
        print(f"✅ Found {len(res)} results.")