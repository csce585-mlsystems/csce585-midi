#!/bin/bash

# Google Colab A100 Training Script with Google Drive Checkpoints
# This script is designed to run on Google Colab with an A100 GPU
#
# IMPORTANT: Before running this script in Colab, you must:
# 1. Mount Google Drive using: from google.colab import drive; drive.mount('/content/drive')
# 2. Then run this script: !bash scripts/train_colab_a100.sh

echo "=========================================="
echo "Google Colab A100 Setup & Training"
echo "=========================================="
echo ""

# Check if running in Colab
if [ ! -d "/content" ]; then
    echo "⚠️  Warning: This script is designed for Google Colab"
    echo "   If running locally, checkpoints will save to ./models/"
    CHECKPOINT_DIR="./models/generators/checkpoints"
else
    echo "✅ Running in Google Colab environment"
    
    
    # Create checkpoint directory
    CHECKPOINT_DIR="/content/drive/MyDrive/csce585_model_checkpoints"
    mkdir -p "$CHECKPOINT_DIR"
    echo "✅ Google Drive mounted successfully"
    echo "✅ Checkpoint directory ready: $CHECKPOINT_DIR"
fi

echo ""
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Check GPU
echo "Checking GPU..."
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB' if torch.cuda.is_available() else '')"
echo ""

# Clone repository if needed
if [ ! -d "csce585-midi" ]; then
    echo "Cloning repository..."
    git clone https://github.com/csce585-mlsystems/csce585-midi.git
    cd csce585-midi
else
    echo "Repository already cloned"
    cd csce585-midi
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q miditok symusic tqdm matplotlib

# Download and prepare data
echo ""
echo "=========================================="
echo "Preparing Dataset"
echo "=========================================="
echo ""

if [ ! -d "data/nottingham-dataset-master" ]; then
    echo "Downloading Nottingham dataset..."
    cd data/
    git clone https://github.com/jukedeck/nottingham-dataset.git nottingham-dataset-master
    cd ..
else
    echo "Dataset already downloaded"
fi

# Run preprocessing
echo ""
echo "Running preprocessing..."
echo "  1. Naive preprocessing..."
python utils/preprocess_naive.py

echo "  2. MIDITok preprocessing..."
python utils/preprocess_miditok.py

echo "  3. Creating measure dataset..."
python utils/measure_dataset.py

echo "  4. Creating augmented dataset..."
python utils/augment_miditok.py

echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""
echo "Models will be saved to: $CHECKPOINT_DIR"
echo ""

# ============================================
# LARGE LSTM MODEL
# ============================================

echo ""
echo "1. Training Large LSTM (1024 hidden, 4 layers)..."
echo "   Estimated time: 1.5-2 hours (with A100 optimizations)"
echo ""

python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 20 \
    --batch_size 512 \
    --lr 0.001 \
    --hidden_size 1024 \
    --num_layers 4 \
    --dropout 0.4 \
    --val_split 0.05 \
    --patience 5 \
    --device cuda \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo "✅ LSTM training complete!"
echo ""

# ============================================
# LARGE GRU MODEL
# ============================================

echo ""
echo "2. Training Large GRU (1024 hidden, 4 layers)..."
echo "   Estimated time: 1-1.5 hours (with A100 optimizations)"
echo ""

python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type gru \
    --epochs 20 \
    --batch_size 512 \
    --lr 0.001 \
    --hidden_size 1024 \
    --num_layers 4 \
    --dropout 0.4 \
    --val_split 0.05 \
    --patience 5 \
    --device cuda \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo "✅ GRU training complete!"
echo ""

# ============================================
# LARGE TRANSFORMER MODEL  
# ============================================

echo ""
echo "3. Training Large Transformer (1024 d_model, 8 layers)..."
echo "   Estimated time: 3-4 hours (with A100 optimizations)"
echo ""

python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type transformer \
    --epochs 20 \
    --batch_size 256 \
    --lr 0.0001 \
    --d_model 1024 \
    --nhead 16 \
    --transformer_layers 8 \
    --dim_feedforward 4096 \
    --dropout 0.3 \
    --val_split 0.05 \
    --patience 5 \
    --device cuda \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo "✅ Transformer training complete!"
echo ""

# ============================================
# COMPLETION
# ============================================

echo ""
echo "=========================================="
echo "✅ ALL TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "Model checkpoints saved to:"
echo "  $CHECKPOINT_DIR"
echo ""
echo "To list your saved models:"
echo "  ls -lh $CHECKPOINT_DIR/miditok_augmented/"
echo ""
