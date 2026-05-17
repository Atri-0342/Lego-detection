#!/bin/bash
# 🟢 Logic: Set path and trigger the Training Orchestrator.
echo "------------------------------------------"
echo "🚀 STARTING LEGO MODEL TRAINING"
echo "------------------------------------------"

# Ensure the root directory is in the python path for imports
export PYTHONPATH=$PYTHONPATH:.

# Run the training script
python src/training/train.py

echo "🏁 Training Process Finished."