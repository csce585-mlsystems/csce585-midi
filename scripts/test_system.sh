#!/bin/bash
# Quick test script to verify all systems are working

echo "======================================"
echo "Testing MIDI Music Generation System"
echo "======================================"
echo ""

# Activate virtual environment
source .venv/bin/activate

echo "1. Testing Generator Architectures..."
python scripts/test_generators.py
if [ $? -eq 0 ]; then
    echo "✅ Generator tests PASSED"
else
    echo "❌ Generator tests FAILED"
    exit 1
fi

echo ""
echo "2. Training LSTM Generator (quick test)..."
python training/train_generator.py --model_type lstm --epochs 1 --max_batches 3
if [ $? -eq 0 ]; then
    echo "✅ LSTM training PASSED"
else
    echo "❌ LSTM training FAILED"
    exit 1
fi

echo ""
echo "3. Training MLP Discriminator (quick test)..."
python training/train_discriminator.py --model_type mlp --label_mode chords --epochs 1 --context_measures 4
if [ $? -eq 0 ]; then
    echo "✅ Discriminator training PASSED"
else
    echo "❌ Discriminator training FAILED"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ ALL TESTS PASSED!"
echo "======================================"
echo ""
echo "System is ready for full training runs:"
echo "  - Generator: python training/train_generator.py --model_type [lstm|gru|transformer]"
echo "  - Discriminator: python training/train_discriminator.py --model_type [mlp|lstm|transformer]"
echo ""
