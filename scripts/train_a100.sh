#!/bin/bash

# Training script optimized for Google Colab A100 GPU
# This script trains larger, more powerful models that take advantage of A100's capabilities

cd data/
git clone https://github.com/jukedeck/nottingham-dataset.git nottingham-dataset-master
cd ..

python utils/preprocess_naive.py
python utils/preprocess_miditok.py
python utils/measure_dataset.py
python utils/augment_miditok.py

echo "=========================================="
echo "A100 GPU Training Pipeline"
echo "=========================================="
echo ""
echo "Optimizations for A100:"
echo "  • Larger batch sizes (128-256)"
echo "  • Bigger models (512-1024 hidden units)"
echo "  • More layers (3-4)"
echo "  • Faster training with GPU acceleration"
echo ""

# Check GPU
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# ============================================
# LARGE LSTM MODELS
# ============================================

echo "=========================================="
echo "Training Large LSTM Models"
echo "=========================================="
echo ""

# Large LSTM (512 hidden, 3 layers)
echo "1. Large LSTM (512 hidden, 3 layers)..."
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.001 \
    --hidden_size 512 \
    --num_layers 3 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda

echo ""

# Extra Large LSTM (1024 hidden, 4 layers)
echo "2. Extra Large LSTM (1024 hidden, 4 layers)..."
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 20 \
    --batch_size 256 \
    --lr 0.001 \
    --hidden_size 1024 \
    --num_layers 4 \
    --dropout 0.4 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda

echo ""

# ============================================
# LARGE GRU MODELS
# ============================================

echo "=========================================="
echo "Training Large GRU Models"
echo "=========================================="
echo ""

# Large GRU (512 hidden, 3 layers)
echo "3. Large GRU (512 hidden, 3 layers)..."
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type gru \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.001 \
    --hidden_size 512 \
    --num_layers 3 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda

echo ""

# Extra Large GRU (1024 hidden, 4 layers)
echo "4. Extra Large GRU (1024 hidden, 4 layers)..."
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type gru \
    --epochs 20 \
    --batch_size 256 \
    --lr 0.001 \
    --hidden_size 1024 \
    --num_layers 4 \
    --dropout 0.4 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda

echo ""

# ============================================
# LARGE TRANSFORMER MODELS
# ============================================

echo "=========================================="
echo "Training Large Transformer Models"
echo "=========================================="
echo ""

# Large Transformer (512 d_model, 8 heads, 6 layers)
echo "5. Large Transformer (512 d_model, 8 heads, 6 layers)..."
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type transformer \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.0001 \
    --d_model 512 \
    --nhead 8 \
    --transformer_layers 6 \
    --dim_feedforward 2048 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda

echo ""

# Extra Large Transformer (1024 d_model, 16 heads, 8 layers)
echo "6. Extra Large Transformer (1024 d_model, 16 heads, 8 layers)..."
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
    --val_split 0.1 \
    --patience 5 \
    --device cuda

echo ""

echo "=========================================="
echo "All models trained!"
echo "=========================================="
echo ""
echo "Check logs/generators/miditok_augmented/ for results"
echo ""
