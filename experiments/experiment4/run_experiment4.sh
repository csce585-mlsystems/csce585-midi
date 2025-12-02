#!/bin/bash
# Experiment 4: Neural Network Architecture Comparison
# =====================================================
# This script runs the full architecture comparison experiment
# comparing LSTM, GRU, and Transformer on naive tokenization

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

echo "=============================================="
echo "Experiment 4: Architecture Comparison"
echo "=============================================="
echo ""
echo "This experiment compares 3 architectures:"
echo "  1. LSTM (baseline) - 941,620 params"
echo "  2. GRU - 711,220 params"
echo "  3. Transformer (small) - 2,133,556 params"
echo ""
echo "All trained on naive tokenization to avoid confounders."
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check that all required models exist
echo "Checking for required model checkpoints..."

LSTM_MODEL="models/generators/checkpoints/nottingham-dataset-final_experiments_naive/lstm_20251129_200244.pth"
GRU_MODEL="models/generators/checkpoints/nottingham-dataset-final_experiments_naive/gru_20251201_171806.pth"
TRANSFORMER_MODEL="models/generators/checkpoints/nottingham-dataset-final_experiments_naive/transformer_20251201_114247.pth"

if [ ! -f "$LSTM_MODEL" ]; then
    echo "ERROR: LSTM model not found at $LSTM_MODEL"
    exit 1
fi

if [ ! -f "$GRU_MODEL" ]; then
    echo "ERROR: GRU model not found at $GRU_MODEL"
    exit 1
fi

if [ ! -f "$TRANSFORMER_MODEL" ]; then
    echo "ERROR: Transformer model not found at $TRANSFORMER_MODEL"
    exit 1
fi

echo "✓ All model checkpoints found"
echo ""

# Check for training loss files
echo "Checking for training loss data..."

LSTM_LOSS="logs/generators/nottingham-dataset-final_experiments_naive/models/train_losses_lstm_20251129_200244.npy"
GRU_LOSS="logs/generators/nottingham-dataset-final_experiments_naive/models/train_losses_gru_20251201_171806.npy"
TRANSFORMER_LOSS="logs/generators/nottingham-dataset-final_experiments_naive/models/train_losses_transformer_20251201_114247.npy"

if [ -f "$LSTM_LOSS" ] && [ -f "$GRU_LOSS" ] && [ -f "$TRANSFORMER_LOSS" ]; then
    echo "✓ All training loss files found"
else
    echo "Warning: Some training loss files missing - loss curve plots may be incomplete"
fi
echo ""

# Parse command line arguments
NUM_SAMPLES=${1:-10}
GENERATE_LENGTH=${2:-200}
OUTPUT_DIR=${3:-"experiments/experiment4/results"}

echo "Configuration:"
echo "  Samples per setting: $NUM_SAMPLES"
echo "  Tokens per sample: $GENERATE_LENGTH"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Ask for confirmation
read -p "Press Enter to start the experiment (Ctrl+C to cancel)..."
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the experiment
echo "=============================================="
echo "Running Architecture Comparison Experiment"
echo "=============================================="
echo ""

python experiments/experiment4/architecture_comparison_experiment.py \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --generate_length "$GENERATE_LENGTH" \
    --seq_length 50

echo ""
echo "=============================================="
echo "Generating Plots"
echo "=============================================="
echo ""

python experiments/experiment4/plot_results.py \
    --results_dir "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key outputs:"
echo "  - $OUTPUT_DIR/experiment_results.json"
echo "  - $OUTPUT_DIR/detailed_results.csv"
echo "  - $OUTPUT_DIR/plots/"
echo ""
echo "To view plots, check the plots/ directory for:"
echo "  - training_loss_curves.png (convergence comparison)"
echo "  - architecture_overview.png (comprehensive comparison)"
echo "  - summary_statistics.md (formatted results table)"
echo ""
