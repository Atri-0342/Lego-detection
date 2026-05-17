import cv2
import numpy as np
from pathlib import Path

class LegoVisualizer:
    """
    Complete logic for manual verification of YOLO labels and images.
    """
    @staticmethod
    def preview_labeled_image(image_path, label_path):
        """
        Draws the YOLO normalized polygons onto the raw image for visual QC.
        """
        img = cv2.imread(str(image_path))
        if img is None: return
        h, w, _ = img.shape

        if not Path(label_path).exists():
            print(f"❌ Label not found: {label_path}")
            return

        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if not parts: continue
                
                # Class is the first number, rest are x1 y1 x2 y2...
                points = np.array(parts[1:]).reshape(-1, 2)
                points[:, 0] *= w # Denormalize X
                points[:, 1] *= h # Denormalize Y
                points = points.astype(np.int32)

                # Draw Polygon mask outline
                cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(img, "LEGO Brick", (points[0][0], points[0][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Dataset Verification", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Logic to test the visualizer on the first training sample
    img_dir = Path("data/processed/train/images")
    if img_dir.exists():
        sample_img = next(img_dir.glob("*.png"))
        sample_lbl = Path("data/processed/train/labels") / f"{sample_img.stem}.txt"
        LegoVisualizer.preview_labeled_image(sample_img, sample_lbl)


'''
import cv2
import numpy as np
import PIL.Image  # 🟢 ADD THIS LINE: Explicitly load the Image attribute
from pathlib import Path
from google.colab.patches import cv2_imshow

class LegoVisualizer:
    """
    Complete logic for manual verification of YOLO labels and images.
    """
    @staticmethod
    def preview_labeled_image(image_path, label_path):
        """
        Draws the YOLO normalized polygons onto the raw image for visual QC.
        """
        img = cv2.imread(str(image_path))
        if img is None: 
            print(f"❌ Could not read image: {image_path}")
            return
            
        h, w, _ = img.shape

        if not Path(label_path).exists():
            print(f"❌ Label not found: {label_path}")
            return

        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if not parts: continue
                
                # Class is the first number, rest are polygons
                points = np.array(parts[1:]).reshape(-1, 2)
                points[:, 0] *= w 
                points[:, 1] *= h 
                points = points.astype(np.int32)

                # Draw Polygon mask outline
                cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(img, "LEGO Brick", (points[0][0], points[0][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"🖼️ Displaying: {image_path.name}")
        # Convert BGR (OpenCV) to RGB (PIL) for correct colors in Colab
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2_imshow(img) 

if __name__ == "__main__":
    ROOT = Path("/content/Lego-detection")
    img_dir = ROOT / "data/processed/train/images"
    
    if img_dir.exists():
        try:
            sample_img = next(img_dir.glob("*.png"))
            sample_lbl = ROOT / "data/processed/train/labels" / f"{sample_img.stem}.txt"
            LegoVisualizer.preview_labeled_image(sample_img, sample_lbl)
        except StopIteration:
            print("❌ No images found.")
    else:
        print(f"❌ Directory not found: {img_dir}")
        
'''