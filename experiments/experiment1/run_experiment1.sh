#!/bin/bash

# Experiment 1: Harmonic Consistency - Generator Alone vs Generator + Discriminator
# This script runs the harmonic consistency experiment with recommended settings

set -e  # Exit on error

echo "========================================================================"
echo "Experiment 1: Harmonic Consistency Comparison"
echo "========================================================================"
echo ""
echo "This experiment compares:"
echo "  1. Generator alone (baseline)"
echo "  2. Generator + Discriminator conditioning (guided)"
echo ""
echo "Metric: Harmonic consistency across measures"
echo ""

# Configuration
# naive model
#GENERATOR_MODEL="models/generators/checkpoints/nottingham-dataset-final_experiments_naive/lstm_20251129_200244.pth"
DISCRIMINATOR_MODEL="models/discriminators/checkpoints/lstm/lstm_labelpitches_ctx4_ep10.pt"
GENERATOR_MODEL="models/generators/checkpoints/nottingham-dataset-final_experiments_miditok/lstm_20251130_091025.pth"
DATA_DIR="data/nottingham-dataset-final_experiments_miditok"
OUTPUT_DIR="experiments/experiment1/results_miditok/comprehensive"
NUM_SAMPLES=50
GENERATE_LENGTH=200
GUIDANCE_STRENGTH=0.5
CONTEXT_MEASURES=4

echo "Configuration:"
echo "  Generator model:       $GENERATOR_MODEL"
echo "  Discriminator model:   $DISCRIMINATOR_MODEL"
echo "  Data directory:        $DATA_DIR"
echo "  Output directory:      $OUTPUT_DIR"
echo "  Samples per condition: $NUM_SAMPLES (for robust statistics)"
echo "  Notes per sample:      $GENERATE_LENGTH"
echo "  Guidance strength:     $GUIDANCE_STRENGTH"
echo "  Context measures:      $CONTEXT_MEASURES"
echo ""
echo "Total generations: $((NUM_SAMPLES * 2)) MIDI files"
echo "Estimated time: ~30-45 minutes"
echo ""

# Check if models exist
if [ ! -f "$GENERATOR_MODEL" ]; then
    echo "ERROR: Generator model not found at $GENERATOR_MODEL"
    exit 1
fi

if [ ! -f "$DISCRIMINATOR_MODEL" ]; then
    echo "ERROR: Discriminator model not found at $DISCRIMINATOR_MODEL"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found at $DATA_DIR"
    exit 1
fi

echo "All files verified. Starting experiment..."
echo ""

# Run the experiment
python experiments/experiment1/comprehensive_eval_experiment.py \
  --generator_model "$GENERATOR_MODEL" \
  --discriminator_model "$DISCRIMINATOR_MODEL" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --generate_length "$GENERATE_LENGTH" \
  --guidance_strength "$GUIDANCE_STRENGTH" \
  --context_measures "$CONTEXT_MEASURES"

echo ""
echo "Generating plots..."
python experiments/experiment1/plot_comprehensive_results.py

echo ""
echo "========================================================================"
echo "Experiment completed!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Plots saved to: $OUTPUT_DIR/plots"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/experiment_results.json | python -m json.tool"
echo ""
echo "To listen to generated MIDI files:"
echo "  open $OUTPUT_DIR/baseline_generator_alone/"
echo "  open $OUTPUT_DIR/guided_generator_discriminator/"
echo ""
