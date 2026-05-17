import os
from pathlib import Path

def flashlight():
    base = Path("raw_data")
    print(f"--- 🔦 Checking Path: {base.resolve()} ---")
    
    if not base.exists():
        print("❌ The folder 'raw_data' does not exist in this directory!")
        print(f"I see these folders instead: {os.listdir('.')}")
        return

    # Look deep into the folders
    for root, dirs, files in os.walk(base):
        level = root.replace(str(base), '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}📂 {os.path.basename(root)}/")
        
        # Show first 2 files if they exist
        sub_files = [f for f in files if f.endswith('.png')]
        for f in sub_files[:2]:
            print(f"{indent}    📄 {f}")

if __name__ == "__main__":
    flashlight()