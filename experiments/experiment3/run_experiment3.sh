#!/bin/bash
# Experiment 3: Tokenization Strategy Comparison
# ===============================================
# This script runs the full tokenization comparison experiment
# comparing naive tokenization, MidiTok REMI, and MidiTok REMI + augmentation

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

echo "=============================================="
echo "Experiment 3: Tokenization Strategy Comparison"
echo "=============================================="
echo ""
echo "This experiment compares 3 baseline models:"
echo "  1. Naive tokenization (simple note representation)"
echo "  2. MidiTok REMI (structured tokenization)"
echo "  3. MidiTok REMI + Augmentation (transposed training data)"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check that all required models exist
echo "Checking for required model checkpoints..."

NAIVE_MODEL="models/generators/checkpoints/nottingham-dataset-final_experiments_naive/lstm_20251129_200244.pth"
MIDITOK_MODEL="models/generators/checkpoints/nottingham-dataset-final_experiments_miditok/lstm_20251130_091025.pth"
MIDITOK_AUG_MODEL="models/generators/checkpoints/nottingham-dataset-final_experiments_miditok_augmented/lstm_20251201_170956.pth"

if [ ! -f "$NAIVE_MODEL" ]; then
    echo "ERROR: Naive model not found at $NAIVE_MODEL"
    exit 1
fi

if [ ! -f "$MIDITOK_MODEL" ]; then
    echo "ERROR: MidiTok model not found at $MIDITOK_MODEL"
    exit 1
fi

if [ ! -f "$MIDITOK_AUG_MODEL" ]; then
    echo "ERROR: MidiTok Augmented model not found at $MIDITOK_AUG_MODEL"
    exit 1
fi

echo "✓ All model checkpoints found"
echo ""

# Parse arguments
NUM_SAMPLES=${1:-10}  # first arg has default of 10
GENERATE_LENGTH=${2:-200}  # second arg has default of 200

echo "Configuration:"
echo "  Samples per setting: $NUM_SAMPLES"
echo "  Generate length: $GENERATE_LENGTH tokens"
echo ""

# Run the experiment
echo "Starting experiment..."
echo ""

python experiments/experiment3/tokenization_comparison_experiment.py \
    --output_dir experiments/experiment3/results \
    --num_samples "$NUM_SAMPLES" \
    --generate_length "$GENERATE_LENGTH" \
    --seq_length 50

echo ""
echo "=============================================="
echo "Experiment completed!"
echo "=============================================="
echo ""
echo "Results saved to: experiments/experiment3/results/"
echo ""

# Generate plots
echo "Generating visualization plots..."
python experiments/experiment3/plot_results.py \
    --results_dir experiments/experiment3/results

echo ""
echo "Plots saved to: experiments/experiment3/results/plots/"
echo ""
echo "Key files:"
echo "  - experiments/experiment3/results/experiment_results.json (summary)"
echo "  - experiments/experiment3/results/detailed_results.csv (all metrics)"
echo "  - experiments/experiment3/results/plots/ (visualizations)"
