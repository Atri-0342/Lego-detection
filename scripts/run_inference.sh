#!/bin/bash
# 🟢 Logic: Execute post-processing on the 'test' dataset split.
echo "------------------------------------------"
echo "🔍 STARTING INFERENCE & VISUALIZATION"
echo "------------------------------------------"

export PYTHONPATH=$PYTHONPATH:.

# Run the BBox Drawer which uses the NMS logic internally
python src/postprocessing/bbox_drawer.py

echo "🖼️  Check 'data/results/' for the output images."
echo "✅ Inference Complete!"