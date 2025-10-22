#!/bin/bash
# Quick experiment script for presentation

echo "=== TRAINING QUICK MODELS FOR PRESENTATION ==="

echo "Training Naive model (small)..."
python train.py --dataset naive --epochs 3 --batch_size 16 --hidden_size 128 --max_batches 10

echo "Training Naive model (large)..."
python train.py --dataset naive --epochs 3 --batch_size 16 --hidden_size 256 --max_batches 10

echo "Training MidiTok model (small)..."
python train.py --dataset miditok --epochs 3 --batch_size 16 --hidden_size 128 --max_batches 10

echo "Training MidiTok model (large)..."
python train.py --dataset miditok --epochs 3 --batch_size 16 --hidden_size 256 --max_batches 10

echo "=== GENERATING SAMPLE OUTPUTS ==="

# Find the latest models
NAIVE_MODEL=$(ls -t models/naive/*.pth | head -1)
MIDITOK_MODEL=$(ls -t models/miditok/*.pth | head -1)

echo "Generating from Naive model: $NAIVE_MODEL"
python generate.py --model_path "$NAIVE_MODEL" --length 200 --temperature 0.8

echo "Generating from MidiTok model: $MIDITOK_MODEL"
python generate.py --model_path "$MIDITOK_MODEL" --length 200 --temperature 0.8

echo "=== CREATING COMPARISON PLOTS ==="
python create_presentation_plots.py

echo "=== PRESENTATION MATERIALS READY! ==="
echo "Check these folders for outputs:"
echo "- outputs/midi/ (generated MIDI files)"
echo "- outputs/training_loss/ (loss curves)"
echo "- outputs/presentation_comparison.png (comparison plots)"
echo "- logs/ (detailed training metrics)"