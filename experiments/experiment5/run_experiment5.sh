#!/bin/bash
# Experiment 5: Discriminator Architecture Comparison
# =====================================================
# 
# This script runs the discriminator architecture comparison experiment,
# testing how MLP, LSTM, and Transformer discriminators affect guided
# music generation quality with a fixed baseline LSTM generator.

set -e  # Exit on error

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "=============================================="
echo "Experiment 5: Discriminator Architecture Comparison"
echo "=============================================="
echo ""
echo "Generator: LSTM (naive tokenization baseline)"
echo "Discriminators: None (baseline), MLP, LSTM, Transformer"
echo ""

# Check if discriminator checkpoints exist
echo "Checking discriminator checkpoints..."

DISC_MLP="models/discriminators/checkpoints/mlp/mlp_labelpitches_ctx4_ep15.pt"
DISC_LSTM="models/discriminators/checkpoints/lstm/lstm_labelpitches_ctx4_ep10.pt"
DISC_TRANS="models/discriminators/checkpoints/transformer/transformer_labelchords_ctx4_ep6.pt"
GEN_PATH="models/generators/checkpoints/nottingham-dataset-final_experiments_naive/lstm_20251129_200244.pth"

missing_models=0

if [ ! -f "$GEN_PATH" ]; then
    echo "WARNING: Generator not found: $GEN_PATH"
    missing_models=1
fi

if [ ! -f "$DISC_MLP" ]; then
    echo "WARNING: MLP discriminator not found: $DISC_MLP"
    echo "  Available MLP models:"
    ls -la models/discriminators/checkpoints/mlp/ 2>/dev/null || echo "  None found"
    missing_models=1
fi

if [ ! -f "$DISC_LSTM" ]; then
    echo "WARNING: LSTM discriminator not found: $DISC_LSTM"
    echo "  Available LSTM models:"
    ls -la models/discriminators/checkpoints/lstm/ 2>/dev/null || echo "  None found"
    missing_models=1
fi

if [ ! -f "$DISC_TRANS" ]; then
    echo "WARNING: Transformer discriminator not found: $DISC_TRANS"
    echo "  Available Transformer models:"
    ls -la models/discriminators/checkpoints/transformer/ 2>/dev/null || echo "  None found"
    missing_models=1
fi

if [ $missing_models -eq 1 ]; then
    echo ""
    echo "Some models are missing. The experiment will skip those conditions."
    echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
    sleep 5
fi

echo "All checks passed. Starting experiment..."
echo ""

# Create results directory
mkdir -p experiments/experiment5/results

# Run the experiment
echo "Running discriminator comparison experiment..."
echo "This will generate samples with each discriminator architecture."
echo ""

python experiments/experiment5/discriminator_comparison_experiment.py \
    --output_dir experiments/experiment5/results \
    --num_samples 10 \
    --generate_length 200 \
    --seq_length 50 \
    --guidance_strength 0.5

echo ""
echo "=============================================="
echo "Experiment 5 Complete!"
echo "=============================================="
echo ""
echo "Results saved to: experiments/experiment5/results/"
echo ""
echo "To generate plots, run:"
echo "  python experiments/experiment5/plot_results.py --results_dir experiments/experiment5/results"
echo ""
echo "To analyze existing results:"
echo "  python experiments/experiment5/discriminator_comparison_experiment.py --analyze experiments/experiment5/results/experiment_results.json"
