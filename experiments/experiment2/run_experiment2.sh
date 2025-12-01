#!/bin/bash

# Experiment 2: Sampling Strategy Comparison
# This script compares different sampling strategies on a trained generator model

set -e  # Exit on error

echo "========================================================================"
echo "Experiment 2: Sampling Strategy Comparison"
echo "========================================================================"
echo ""
echo "This experiment compares sampling strategies:"
echo "  1. Greedy (always pick highest probability)"
echo "  2. Top-k (k=5, k=10)"
echo "  3. Top-p / Nucleus (p=0.9, p=0.95)"
echo "  4. Temperature sampling (T=0.5, T=1.0, T=1.5)"
echo ""
echo "Metrics measured:"
echo "  - Note Density"
echo "  - Pitch Range"  
echo "  - Average Polyphony"
echo "  - Scale Consistency"
echo "  - Pitch Entropy"
echo ""
echo "Expected findings:"
echo "  - Greedy: repetitive, low diversity"
echo "  - Top-p: best musical diversity"
echo "  - High temperature: more creative but potentially less coherent"
echo ""

# Configuration
GENERATOR_MODEL="models/generators/checkpoints/nottingham-dataset-final_experiments_naive/lstm_20251129_200244.pth"
DATA_DIR="data/nottingham-dataset-final_experiments_naive"
OUTPUT_DIR="experiments/experiment2/results"
NUM_SAMPLES=30
GENERATE_LENGTH=200

echo "Configuration:"
echo "  Generator model: $GENERATOR_MODEL"
echo "  Data directory:  $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Samples per strategy: $NUM_SAMPLES"
echo "  Notes per sample: $GENERATE_LENGTH"
echo ""
echo "Total generations: $((NUM_SAMPLES * 8)) MIDI files (8 strategies)"
echo "Estimated time: ~45-60 minutes"
echo ""

# Check if model exists
if [ ! -f "$GENERATOR_MODEL" ]; then
    echo "ERROR: Generator model not found at $GENERATOR_MODEL"
    echo ""
    echo "Available models:"
    find models/generators/checkpoints -name "*.pth" -type f 2>/dev/null || echo "No models found"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found at $DATA_DIR"
    exit 1
fi

echo "All files verified. Starting experiment..."
echo ""

# Run the experiment
python experiments/experiment2/sampling_experiment.py \
    --generator_model "$GENERATOR_MODEL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --generate_length "$GENERATE_LENGTH"

echo ""
echo "Generating plots..."
python experiments/experiment2/plot_sampling_results.py \
    --results_path "$OUTPUT_DIR/experiment_results.json" \
    --output_dir "$OUTPUT_DIR/plots"

echo ""
echo "========================================================================"
echo "Experiment completed!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - experiment_results.json (raw data)"
echo "  - plots/ (visualizations)"
echo ""
echo "Generated MIDI files organized by strategy:"
ls -d "$OUTPUT_DIR"/*/ 2>/dev/null | grep -v plots || echo "  (run experiment first)"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/experiment_results.json | python -m json.tool | head -100"
echo ""
echo "To view plots:"
echo "  open $OUTPUT_DIR/plots/"
echo ""
