import os
from pathlib import Path

def inspect_raw_data(base_dir="raw_data/raw"):
    base_path = Path(base_dir)
    print(f"🔍 Scanning: {base_path.resolve()}")
    
    # We look for folders like brick5, brick10, brick15
    for brick_folder in base_path.iterdir():
        if not brick_folder.is_dir():
            continue
            
        rgb = list((brick_folder / "rgb_images").glob("*.png"))
        ann = list((brick_folder / "annotations").glob("*.json"))
        seg = list((brick_folder / "segmentations").glob("*.png"))
        
        print(f"\n📦 Folder: {brick_folder.name}")
        print(f"   🖼️  Images found: {len(rgb)}")
        print(f"   📝 JSONs found:  {len(ann)}")
        print(f"   🎭 Masks found:  {len(seg)}")

if __name__ == "__main__":
    inspect_raw_data()