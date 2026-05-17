""" import cv2
import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# Connect to your NMS logic
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.postprocessing.nms import LegoNMS

class LegoBBoxDrawer:
    def __init__(self, model_path="outputs/train/lego_run/weights/best.pt"):
        self.model = YOLO(model_path)
        # 🟢 Logic: Initialize your custom NMS filter
        self.custom_nms = LegoNMS(conf_thres=0.3, iou_thres=0.4) 
        self.output_dir = Path("data/results")

    def process_with_custom_nms(self, image_path):
  
        # 1. Get RAW predictions (setting end-to-end NMS to False if supported)
        results = self.model.predict(image_path, conf=0.1, save=False)

        for r in results:
            # 🟢 Logic: You can manually override/filter boxes here using your nms.py
            # For segmentation, we usually let YOLO handle it, but for 
            # ultra-precise counting, we use the custom NMS class:
            # boxes = self.custom_nms.apply(r.boxes.data) 
            
            # 2. Draw the 'Clean' results
            annotated_frame = r.plot() 
            
            save_path = self.output_dir / f"refined_{Path(image_path).name}"
            cv2.imwrite(str(save_path), annotated_frame)
            print(f"✅ NMS applied and saved: {save_path}")

if __name__ == "__main__":
    drawer = LegoBBoxDrawer()
    # Process one test image to verify NMS logic
    sample = "data/processed/test/images/img_0.png" # change to a real image
    if os.path.exists(sample):
        drawer.process_with_custom_nms(sample) """
import os
import sys
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# 🟢 Logic: Dynamic Root Detection for Local VS Code
# This finds the 'Lego-detection' root based on this file being in src/postprocessing/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 🟢 Logic: NMS Import Handling (2026-Ready)
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    try:
        from ultralytics.utils.ops import non_max_suppression
    except ImportError:
        from ultralytics.data.utils import non_max_suppression

class LegoNMS:
    """
    Handles redundant detection filtering if manual tuning is required.
    """
    def __init__(self, conf_thres=0.3, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def apply(self, prediction):
        return non_max_suppression(
            prediction, 
            conf_thres=self.conf_thres, 
            iou_thres=self.iou_thres
        )

class LegoBBoxDrawer:
    """
    Logic for generating and saving visual predictions (bounding boxes + masks).
    """
    def __init__(self, model_rel_path="outputs/checkpoints/weights/best.pt"):
        # Logic: Build absolute path relative to local project root
        full_model_path = PROJECT_ROOT / model_rel_path
        
        if not full_model_path.exists():
            print(f"❌ Error: Model weights not found at: {full_model_path}")
            print("Ensure training finished and 'best.pt' exists in the outputs folder.")
            sys.exit(1)
            
        self.model = YOLO(str(full_model_path))
        
        # 🟢 Logic: Output directed to 'outputs/predictions' per requirement
        self.output_dir = PROJECT_ROOT / "outputs" / "predictions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧠 Model loaded from local path: {model_rel_path}")
        print(f"📂 Refined images will be saved to: {self.output_dir}")

    def process_images(self, source_path):
        """
        Runs inference and uses YOLO's .plot() to draw the 'blue things' (masks).
        """
        # Inference (Forced to CPU for local VS Code stability)
        results = self.model.predict(
            source=str(source_path), 
            conf=0.3,      # Min confidence
            iou=0.45,     # Overlap threshold
            save=False,   # We handle the saving manually via OpenCV
            device='cpu'  # 🟢 Local CPU optimization
        )

        for r in results:
            # Logic: r.plot() generates the image with masks, boxes, and labels
            annotated_frame = r.plot() 
            
            # Save logic using OpenCV
            img_name = Path(r.path).name
            save_path = self.output_dir / f"refined_{img_name}"
            
            cv2.imwrite(str(save_path), annotated_frame)
            print(f"✅ Refined and Saved: {img_name}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 LEGO BBOX & SEGMENTATION DRAWER: LOCAL MODE")
    print("="*50)

    # Initialize drawer
    drawer = LegoBBoxDrawer()
    
    # 🟢 Logic: Define local test image folder
    test_folder = PROJECT_ROOT / "data" / "processed" / "test" / "images"
    
    if test_folder.exists():
        # Find all valid images (.png, .jpg, .jpeg)
        image_extensions = ["*.png", "*.jpg", "*.jpeg"]
        image_list = []
        for ext in image_extensions:
            image_list.extend(list(test_folder.glob(ext)))
        
        if not image_list:
            print(f"⚠️ No images found in {test_folder}. Check your local data.")
        else:
            print(f"📸 Found {len(image_list)} images. Starting processing...")
            for img_path in image_list:
                drawer.process_images(img_path)
            
            print("\n" + "✨"*20)
            print("🎉 VISUALIZATION COMPLETE")
            print(f"📍 Final images located in: {drawer.output_dir}")
            print("✨"*20 + "\n")
    else:
        print(f"❌ Error: Test folder not found at {test_folder}")