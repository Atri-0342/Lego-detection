#!/bin/bash
# 🟢 Logic: Clean start, Convert, then Split.
echo "------------------------------------------"
echo "🛠️  STARTING PREPROCESSING PIEPLINE"
echo "------------------------------------------"

# Step 1: Clean previous processed data
echo "🧹 Cleaning old processed files..."
rm -rf data/processed/*

# Step 2: Run Annotation Converter
echo "🔄 Step 1: Converting Masks to YOLO Polygons..."
python src/preprocessing/annotation_converter.py

# Step 3: Run Data Splitter
echo "📊 Step 2: Splitting data into Train/Val/Test (70/20/10)..."
python src/preprocessing/data_splitter.py

echo "✅ Preprocessing Complete!"