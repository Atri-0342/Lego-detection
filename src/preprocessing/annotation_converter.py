import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_polygon_from_mask(mask_path):
    """
    Reads a mask image and returns a list of normalized YOLO polygon coordinates.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    # Thresholding ensures the mask is strictly black and white
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find the external outline of the LEGO brick
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = mask.shape
    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        
        # Reshape and normalize coordinates
        poly = cnt.reshape(-1, 2).astype(float)
        poly[:, 0] /= w 
        poly[:, 1] /= h
        polygons.append(poly.flatten().tolist())
    
    return polygons

def convert_dataset(data_dir="raw_data/raw", output_dir="data/processed/all"):
    """
    Processes LEGO bricks by mapping mismatched filenames (rgb_0 -> segmentation_0).
    """
    data_path = Path(data_dir).resolve()
    images_out = Path(output_dir).resolve() / "images"
    labels_out = Path(output_dir).resolve() / "labels"
    
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    if not data_path.exists():
        print(f"❌ Error: Root directory '{data_path}' not found.")
        return

    brick_folders = [f for f in data_path.iterdir() if f.is_dir()]
    print(f"📂 Found {len(brick_folders)} brick categories. Starting conversion...")

    for brick_folder in brick_folders:
        rgb_dir = brick_folder / "rgb_images" 
        ann_dir = brick_folder / "annotations"
        seg_dir = brick_folder / "segmentations"
        
        if not rgb_dir.exists():
            continue

        print(f"🧱 Processing {brick_folder.name}...")
        image_files = list(rgb_dir.glob("*.png"))
        
        for img_file in tqdm(image_files, desc=f"Converting {brick_folder.name}"):
            # LOGIC: Extract ID (e.g., '0' from 'brick5_rgb_0.png')
            file_id = img_file.stem.split('_')[-1]
            
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w, _ = img.shape
            
            yolo_lines = []
            
            # --- PRIORITY 1: Segmentation Mask (Polygon) ---
            # Mapping: brickX_rgb_N.png -> brickX_segmentation_N.png
            mask_name = f"{brick_folder.name}_segmentation_{file_id}.png"
            mask_file = seg_dir / mask_name
            
            if mask_file.exists():
                polygons = get_polygon_from_mask(mask_file)
                if polygons:
                    for poly in polygons:
                        poly_str = " ".join([f"{c:.6f}" for c in poly])
                        yolo_lines.append(f"0 {poly_str}")

            # --- PRIORITY 2: JSON Bbox Fallback ---
            # Mapping: brickX_rgb_N.png -> brickX_bbox_N.json
            if not yolo_lines:
                json_name = f"{brick_folder.name}_bbox_{file_id}.json"
                json_file = ann_dir / json_name
                if json_file.exists():
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        # Data is a list of objects; filter for "lego"
                        for obj in data:
                            if obj.get('label') == 'lego':
                                bx, by, bw, bh = obj['x'], obj['y'], obj['width'], obj['height']
                                
                                # Convert [x, y, w, h] to YOLO [x_center, y_center, w, h]
                                x_center = (bx + bw/2) / w
                                y_center = (by + bh/2) / h
                                norm_w, norm_h = bw / w, bh / h
                                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                    except Exception:
                        pass

            # --- SAVE PROCESSED DATA ---
            if yolo_lines:
                # Unique name to avoid collisions: brick5_0.png
                unique_stem = f"{brick_folder.name}_{file_id}"
                
                cv2.imwrite(str(images_out / f"{unique_stem}.png"), img)
                with open(labels_out / f"{unique_stem}.txt", 'w') as f:
                    f.write("\n".join(yolo_lines))

    print(f"\n✅ Conversion complete! Files are in: {output_dir}")

if __name__ == "__main__":
    convert_dataset()