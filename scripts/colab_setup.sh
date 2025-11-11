#!/bin/bash

# Quick Colab Setup Script
# Run this after mounting Google Drive

set -e  # Exit on error

echo "=========================================="
echo "Colab Quick Setup"
echo "=========================================="
echo ""

# Verify we're in Colab
if [ ! -d "/content" ]; then
    echo "❌ Error: Not running in Google Colab"
    exit 1
fi

# Verify Drive is mounted
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "❌ Error: Google Drive not mounted!"
    echo ""
    echo "Please run this in a Colab cell first:"
    echo "  from google.colab import drive"
    echo "  drive.mount('/content/drive')"
    echo ""
    exit 1
fi

echo "✅ Google Drive is mounted"

# Clone repo if needed
cd /content
if [ ! -d "csce585-midi" ]; then
    echo "Cloning repository..."
    git clone https://github.com/csce585-mlsystems/csce585-midi.git
    echo "✅ Repository cloned"
else
    echo "✅ Repository already exists"
fi

cd csce585-midi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q miditok symusic tqdm matplotlib
echo "✅ Dependencies installed"

# Create checkpoint directory
CHECKPOINT_DIR="/content/drive/MyDrive/csce585_model_checkpoints"
mkdir -p "$CHECKPOINT_DIR"
echo "✅ Checkpoint directory ready: $CHECKPOINT_DIR"

# Download dataset
echo ""
echo "Downloading dataset..."
if [ ! -d "data/nottingham-dataset-master" ]; then
    cd data
    git clone https://github.com/jukedeck/nottingham-dataset.git nottingham-dataset-master
    cd ..
    echo "✅ Dataset downloaded"
else
    echo "✅ Dataset already exists"
fi

# Preprocess data
echo ""
echo "Preprocessing data..."
echo "  1. Naive preprocessing..."
python utils/preprocess_naive.py

echo "  2. MIDITok preprocessing..."
python utils/preprocess_miditok.py

echo "  3. Measure dataset..."
python utils/measure_dataset.py

echo "  4. Augmented dataset..."
python utils/augment_miditok.py

echo ""
echo "=========================================="
echo "✅ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "To train a model, run:"
echo ""
echo "  cd /content/csce585-midi"
echo "  python training/train_generator.py \\"
echo "    --dataset miditok_augmented \\"
echo "    --model_type lstm \\"
echo "    --epochs 20 \\"
echo "    --batch_size 256 \\"
echo "    --hidden_size 1024 \\"
echo "    --num_layers 4 \\"
echo "    --checkpoint_dir /content/drive/MyDrive/csce585_model_checkpoints"
echo ""
