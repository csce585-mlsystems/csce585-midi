#!/bin/bash

# Script to preprocess MIDI data with augmentation and train models
# This helps reduce overfitting by creating more diverse training data

echo "=========================================="
echo "MIDITok Data Augmentation Pipeline"
echo "=========================================="
echo ""

# Step 1: Run data augmentation
echo "Step 1: Preprocessing with data augmentation..."
echo "This will create 7x more training data via transposition"
echo ""

python utils/augment_miditok.py \
    --transpositions="-5,-3,-1,0,1,3,5" \
    --output_dir="data/miditok_augmented"

if [ $? -ne 0 ]; then
    echo "Error: Preprocessing failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Training models on augmented data"
echo "=========================================="
echo ""

# Train LSTM model
echo "Training LSTM model..."
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 15 \
    --batch_size 64 \
    --lr 0.001 \
    --hidden_size 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 4

echo ""

# Train GRU model
echo "Training GRU model..."
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type gru \
    --epochs 15 \
    --batch_size 64 \
    --lr 0.001 \
    --hidden_size 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 4

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Check logs/generators/miditok_augmented/ for results"
echo ""
