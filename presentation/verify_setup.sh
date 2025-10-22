#!/bin/bash
# Pre-training verification script

echo "🔍 Verifying directory structure before starting comprehensive training..."

# Check if all necessary directories exist
DIRS_TO_CHECK=(
    "models/naive"
    "models/miditok" 
    "logs/naive"
    "logs/miditok"
    "outputs/naive"
    "outputs/miditok"
    "data/naive"
    "data/miditok"
)

echo "Checking directory structure..."
for dir in "${DIRS_TO_CHECK[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir exists"
    else
        echo "❌ $dir missing - creating..."
        mkdir -p "$dir"
    fi
done

# Check if training data exists
echo ""
echo "Checking training data..."
if [ -f "data/naive/sequences.npy" ]; then
    echo "✅ Naive dataset ready"
else
    echo "❌ Naive dataset missing!"
fi

if [ -f "data/miditok/sequences.npy" ]; then
    echo "✅ MidiTok dataset ready"
else  
    echo "❌ MidiTok dataset missing!"
fi

# Check if train.py exists and is executable
echo ""
echo "Checking training script..."
if [ -f "train.py" ]; then
    echo "✅ train.py exists"
    # Test a quick run
    echo "Testing training script with minimal parameters..."
    python train.py --dataset naive --epochs 1 --max_batches 1 >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Training script works correctly"
    else
        echo "❌ Training script has issues - check manually"
    fi
else
    echo "❌ train.py missing!"
fi

echo ""
echo "🚀 Ready to start comprehensive training!"
echo "Run: ./comprehensive_training.sh"
echo ""
echo "Estimated time: 2-3 hours"
echo "Will create organized outputs in:"
echo "  - models/naive/ & models/miditok/"
echo "  - outputs/naive/midi/ & outputs/miditok/midi/"
echo "  - logs/naive/ & logs/miditok/"