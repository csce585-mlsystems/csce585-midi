#!/bin/bash
# Quick test to verify the optimizations work before running full training

set -e

source .venv/bin/activate

echo "=========================================="
echo "Testing Optimized Training System"
echo "=========================================="
echo ""

echo "Test 1: Quick training test (10 batches, no subsample)"
echo "------------------------------------------------------"
python training/train_generator.py \
    --model_type lstm \
    --dataset miditok \
    --epochs 2 \
    --max_batches 10 \
    --batch_size 32 \
    --val_split 0.1 \
    --patience 5

echo ""
echo "✅ Test 1 passed!"
echo ""

echo "Test 2: Subsampled dataset test (50% data, 20 batches)"
echo "--------------------------------------------------------"
python training/train_generator.py \
    --model_type gru \
    --dataset miditok \
    --epochs 2 \
    --max_batches 20 \
    --batch_size 32 \
    --subsample_ratio 0.5 \
    --val_split 0.1 \
    --patience 5

echo ""
echo "✅ Test 2 passed!"
echo ""

echo "=========================================="
echo "All tests passed!"
echo "=========================================="
echo ""
echo "The optimizations are working correctly."
echo "You can now run the full training with:"
echo "  bash scripts/train_miditok_optimized.sh"
echo ""
