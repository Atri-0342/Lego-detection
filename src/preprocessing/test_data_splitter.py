import os
import shutil
import random
from pathlib import Path

def split_processed_data(input_dir="data/processed/all", output_dir="data/processed"):
    """
    Splits images and labels from the 'all' folder into train, val, and test sets.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 1. Gather all images that have a corresponding label
    images = list((input_path / "images").glob("*.png"))
    
    if not images:
        print(f"❌ No images found in {input_path / 'images'}. Did the convertor run correctly?")
        return

    # Shuffle for randomness
    random.seed(42)  # Set seed for reproducible results
    random.shuffle(images)
    
    # 2. Define Split Points (70% Train, 20% Val, 10% Test)
    total = len(images)
    train_end = int(total * 0.7)
    val_end = int(total * 0.9)
    
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    print(f"📊 Total files found: {total}")
    
    # 3. Move the files
    for split_name, split_files in splits.items():
        img_dest = output_path / split_name / "images"
        lbl_dest = output_path / split_name / "labels"
        
        # Create folders for each split
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        
        print(f"   📂 Moving {len(split_files)} files to '{split_name}'...")
        
        for img_file in split_files:
            # Move Image
            shutil.move(str(img_file), str(img_dest / img_file.name))
            
            # Move corresponding Label (.txt)
            label_file = input_path / "labels" / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.move(str(label_file), str(lbl_dest / label_file.name))
            else:
                print(f"   ⚠️ Warning: Label missing for {img_file.name}")

    print("\n✅ Data Splitting Complete!")
    
    # Optional: Clean up the empty 'all' folder
    # shutil.rmtree(input_dir)

if __name__ == "__main__":
    split_processed_data()