import os
from ultralytics import YOLO
from pathlib import Path

class LegoModel:
    """
    Complete logic for managing the YOLO Segmentation model life cycle,
    including initialization, weight loading, and architecture selection.
    """
    
    def __init__(self, model_cfg):
        """
        Initializes the model based on the provided configuration dictionary.
        :param model_cfg: Dictionary containing 'model' (path or variant name)
        """
        self.model_name = model_cfg.get('model', 'yolo11n-seg.pt')
        self.model = self._initialize_model()

    def _initialize_model(self):
        """
        Logic to determine if we are starting fresh or resuming.
        If a 'best.pt' or 'last.pt' exists in the weights folder, it can be loaded.
        """
        print(f"🏗️  Initializing Architecture: {self.model_name}")
        
        try:
            # Check if the model_name is a local path to a weights file or a cloud variant
            if os.path.exists(self.model_name):
                print(f"🔄 Loading local weights from: {self.model_name}")
                return YOLO(self.model_name)
            else:
                print(f"🌐 Downloading/Loading pretrained weights: {self.model_name}")
                return YOLO(self.model_name)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def get_model(self):
        """Returns the actual Ultralytics YOLO object."""
        return self.model

    def export_model(self, format="onnx"):
        """
        Logic for post-training deployment. 
        Converts the .pt file to other formats (onnx, engine, tflite).
        """
        print(f"📦 Exporting model to {format} format...")
        return self.model.export(format=format)

    @staticmethod
    def get_latest_checkpoint(output_dir="outputs/train/lego_segmentation/weights"):
        """
        Static logic to find the last saved checkpoint to resume training.
        """
        last_weights = Path(output_dir) / "last.pt"
        if last_weights.exists():
            return str(last_weights)
        return None

# Example of how this logic connects:
if __name__ == "__main__":
    # This part is for standalone testing of the model logic
    fake_config = {"model": "yolo11n-seg.pt"}
    lego_ai = LegoModel(fake_config)
    print("✅ Model logic verified.")