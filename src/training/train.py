import yaml
import os
import sys
import shutil
from pathlib import Path

# 🟢 Logic: Dynamic Root Detection
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.model import LegoModel
from src.training.dataset import LegoDataset

class LegoTrainer:
    """
    Orchestrates the training process and automatically organizes 
    outputs into /checkpoints and /logs.
    """
    
    def __init__(self, train_cfg_path="configs/train_config.yaml", ds_cfg_path="configs/dataset.yaml"):
        self.ROOT = PROJECT_ROOT
        self.train_cfg_path = self.ROOT / train_cfg_path
        self.ds_cfg_path = self.ROOT / ds_cfg_path
        
        # Load hyperparams
        self.train_params = self._load_yaml(self.train_cfg_path)
        
        # Initialize Dataset and Model logic
        self.dataset_manager = LegoDataset(dataset_config_path=str(self.ds_cfg_path))
        self.model_manager = LegoModel(model_cfg=self.train_params)

    def _load_yaml(self, path):
        if not path.exists():
            raise FileNotFoundError(f"❌ Configuration not found at {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def run(self, resume=False):
        print("\n" + "="*50)
        print("🏗️  LEGO DETECTION: ORGANIZED TRAINING PIPELINE")
        print("="*50)

        # 1. Validate Dataset
        if not self.dataset_manager.validate_structure():
            print("🛑 Training aborted: Dataset validation failed.")
            return

        # 2. Setup Final Target Folders
        CHECKPOINT_DIR = self.ROOT / "outputs" / "checkpoints"
        LOGS_DIR = self.ROOT / "outputs" / "logs"
        TEMP_DIR = self.ROOT / "outputs" / "temp_run" # Temporary holding spot

        # Create directories
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # 3. Get Model
        yolo_model = self.model_manager.get_model()

        # 4. Run Training
        print(f"🔥 Training starting. Target Batch: {self.train_params.get('batch', 4)}")
        
        yolo_model.train(
            data=str(self.ds_cfg_path),
            epochs=self.train_params.get('epochs', 25),
            imgsz=self.train_params.get('imgsz', 640),
            batch=self.train_params.get('batch', 4),
            device='cpu',
            optimizer=self.train_params.get('optimizer', 'AdamW'),
            lr0=self.train_params.get('lr0', 0.01),
            patience=self.train_params.get('patience', 5),
            workers=2,
            project=str(self.ROOT / "outputs"),
            name="temp_run",  # Save to temp folder first
            exist_ok=True,
            save=True,
            plots=True
        )

        # 🟢 Logic: Automatic File Organization (The "Logger" Logic)
        print("\n📦 Organizing artifacts into /checkpoints and /logs...")
        
        if TEMP_DIR.exists():
            # Move Weights (.pt files)
            weights_src = TEMP_DIR / "weights"
            if weights_src.exists():
                (CHECKPOINT_DIR / "weights").mkdir(exist_ok=True)
                for pt_file in weights_src.glob("*.pt"):
                    shutil.move(str(pt_file), str(CHECKPOINT_DIR / "weights" / pt_file.name))

            # Move Logs (CSV, PNG curves, YAML configs)
            log_extensions = ["*.csv", "*.png", "*.yaml"]
            for ext in log_extensions:
                for log_file in TEMP_DIR.glob(ext):
                    # Prevent moving weights folder as a file
                    if log_file.is_file():
                        shutil.move(str(log_file), str(LOGS_DIR / log_file.name))

            # Clean up the temp folder
            shutil.rmtree(TEMP_DIR)

        print("\n" + "✨" * 20)
        print("TRAINING & LOGGING COMPLETE")
        print(f"📍 Weights: {CHECKPOINT_DIR}/weights/best.pt")
        print(f"📍 Logs:    {LOGS_DIR}/results.png")
        print("✨" * 20 + "\n")

if __name__ == "__main__":
    try:
        trainer = LegoTrainer()
        trainer.run(resume=False)
    except Exception as e:
        print(f"💥 Critical Training Error: {e}")