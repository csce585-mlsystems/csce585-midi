#!/bin/bash

# Quick verification script to check if all required files exist for the experiment

echo "Verifying experiment setup..."
echo ""

ERRORS=0

# Check generator model
GENERATOR="models/generators/checkpoints/nottingham-dataset-final_experiments_naive/lstm_20251129_200244.pth"
if [ -f "$GENERATOR" ]; then
    echo "✓ Generator model found: $GENERATOR"
else
    echo "✗ Generator model NOT found: $GENERATOR"
    ERRORS=$((ERRORS + 1))
fi

# Check discriminator model
DISCRIMINATOR="models/discriminators/checkpoints/lstm/lstm_labelpitches_ctx4_ep10.pt"
if [ -f "$DISCRIMINATOR" ]; then
    echo "✓ Discriminator model found: $DISCRIMINATOR"
else
    echo "✗ Discriminator model NOT found: $DISCRIMINATOR"
    ERRORS=$((ERRORS + 1))
fi

# Check data directory
DATA_DIR="data/nottingham-dataset-final_experiments_naive"
if [ -d "$DATA_DIR" ]; then
    echo "✓ Data directory found: $DATA_DIR"
    
    # Check specific data files
    if [ -f "$DATA_DIR/sequences.npy" ]; then
        echo "  ✓ sequences.npy exists"
    else
        echo "  ✗ sequences.npy NOT found"
        ERRORS=$((ERRORS + 1))
    fi
    
    if [ -f "$DATA_DIR/note_to_int.pkl" ]; then
        echo "  ✓ note_to_int.pkl exists"
    else
        echo "  ✗ note_to_int.pkl NOT found"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "✗ Data directory NOT found: $DATA_DIR"
    ERRORS=$((ERRORS + 1))
fi

# Check experiment script
EXPERIMENT_SCRIPT="experiments/harmonic_consistency_experiment.py"
if [ -f "$EXPERIMENT_SCRIPT" ]; then
    echo "✓ Experiment script found: $EXPERIMENT_SCRIPT"
else
    echo "✗ Experiment script NOT found: $EXPERIMENT_SCRIPT"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "========================================"
if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed! Ready to run experiment."
    echo "========================================"
    echo ""
    echo "To run the experiment:"
    echo "  ./experiments/run_experiment1.sh"
    echo ""
    echo "Or manually:"
    echo "  python experiments/harmonic_consistency_experiment.py \\"
    echo "    --generator_model $GENERATOR \\"
    echo "    --discriminator_model $DISCRIMINATOR \\"
    echo "    --data_dir $DATA_DIR"
    exit 0
else
    echo "✗ Found $ERRORS error(s). Please fix before running experiment."
    echo "========================================"
    exit 1
fi
