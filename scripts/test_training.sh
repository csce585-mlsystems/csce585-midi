#!/bin/bash
# Quick test to verify training works

set -e

source .venv/bin/activate

echo "=========================================="
echo "Quick Training Test"
echo "=========================================="
echo ""

echo "Test 1: LSTM with 10 batches only..."
python training/train_generator.py \
  --model_type lstm \
  --dataset miditok \
  --epochs 2 \
  --batch_size 32 \
  --max_batches 10 \
  --hidden_size 64 \
  --num_layers 1

echo ""
echo "✅ Test 1 passed!"
echo ""

echo "Test 2: GRU with 10% of data..."
python training/train_generator.py \
  --model_type gru \
  --dataset miditok \
  --epochs 2 \
  --batch_size 64 \
  --subsample_ratio 0.1 \
  --hidden_size 64 \
  --num_layers 1

echo ""
echo "✅ Test 2 passed!"
echo ""

echo "=========================================="
echo "All tests passed! Training system works."
echo "=========================================="
