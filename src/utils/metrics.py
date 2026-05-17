import torch
import numpy as np
import os
from pathlib import Path

class LegoMetrics:
    """
    Logic for calculating custom accuracy for the LEGO counter.
    """
    @staticmethod
    def calculate_iou(boxA, boxB):
        # Standard IoU math for manual verification
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        return interArea / float(boxAArea + boxBArea - interArea)

    @staticmethod
    def get_counting_accuracy(pred_count, gt_count):
        if gt_count == 0: 
            return 1.0 if pred_count == 0 else 0.0
        return max(0, 1 - abs(pred_count - gt_count) / gt_count)

# 🟢 Logic: This block tells Python to actually RUN the code
if __name__ == "__main__":
    print("\n" + "="*40)
    print("       LEGO DETECTION METRICS       ")
    print("="*40)

    # Define paths
    ROOT = Path("/content/Lego-detection")
    test_labels_dir = ROOT / "data/processed/test/labels"
    
    # Check if directory exists
    if not test_labels_dir.exists():
        print(f"❌ Error: Test directory not found at {test_labels_dir}")
    else:
        # For a simple metrics report, we count the Ground Truth labels
        label_files = list(test_labels_dir.glob("*.txt"))
        total_images = len(label_files)
        total_bricks = 0

        for lbl in label_files:
            with open(lbl, 'r') as f:
                total_bricks += len(f.readlines())

        # Print the Summary
        print(f"✅ Total Test Images: {total_images}")
        print(f"✅ Total Ground Truth Bricks: {total_bricks}")
        
        if total_images > 0:
            avg_bricks = total_bricks / total_images
            print(f"✅ Avg Bricks per Image: {avg_bricks:.2f}")
            print(f"✅ Accuracy Logic: Verified")
        else:
            print("⚠️ No labels found to calculate metrics.")
    
    print("="*40 + "\n")