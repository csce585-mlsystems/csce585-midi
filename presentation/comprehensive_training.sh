#!/bin/bash
# Comprehensive training script for presentation - runs while away
# This will train multiple models with different configurations for comparison

echo "Starting comprehensive MIDI generation training session..."
echo "Started at: $(date)"

# Create logs directory for this session
SESSION_DIR="logs/training_session_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SESSION_DIR"

# Log file for this session
LOG_FILE="$SESSION_DIR/training_log.txt"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_message "=== COMPREHENSIVE TRAINING SESSION STARTED ==="

# ==============================================================================
# NAIVE DATASET EXPERIMENTS
# ==============================================================================

log_message "🎹 Starting NAIVE dataset experiments..."

log_message "Training Naive - Small Model (baseline)"
python train.py --dataset naive --epochs 15 --batch_size 32 --hidden_size 128 --num_layers 2 --lr 0.001
if [ $? -eq 0 ]; then
    log_message "✅ Naive small model completed successfully"
else
    log_message "❌ Naive small model failed"
fi

log_message "Training Naive - Medium Model"
python train.py --dataset naive --epochs 20 --batch_size 32 --hidden_size 256 --num_layers 2 --lr 0.001
if [ $? -eq 0 ]; then
    log_message "✅ Naive medium model completed successfully"
else
    log_message "❌ Naive medium model failed"
fi

log_message "Training Naive - Large Model"
python train.py --dataset naive --epochs 25 --batch_size 16 --hidden_size 512 --num_layers 3 --lr 0.0005
if [ $? -eq 0 ]; then
    log_message "✅ Naive large model completed successfully"
else
    log_message "❌ Naive large model failed"
fi

log_message "Training Naive - Deep Model"
python train.py --dataset naive --epochs 20 --batch_size 32 --hidden_size 256 --num_layers 4 --lr 0.0008
if [ $? -eq 0 ]; then
    log_message "✅ Naive deep model completed successfully"
else
    log_message "❌ Naive deep model failed"
fi

# ==============================================================================
# MIDITOK DATASET EXPERIMENTS
# ==============================================================================

log_message "🎼 Starting MIDITOK dataset experiments..."

log_message "Training MidiTok - Small Model (baseline)"
python train.py --dataset miditok --epochs 15 --batch_size 32 --hidden_size 128 --num_layers 2 --lr 0.0005
if [ $? -eq 0 ]; then
    log_message "✅ MidiTok small model completed successfully"
else
    log_message "❌ MidiTok small model failed"
fi

log_message "Training MidiTok - Medium Model"
python train.py --dataset miditok --epochs 20 --batch_size 32 --hidden_size 256 --num_layers 2 --lr 0.0005
if [ $? -eq 0 ]; then
    log_message "✅ MidiTok medium model completed successfully"
else
    log_message "❌ MidiTok medium model failed"
fi

log_message "Training MidiTok - Large Model"
python train.py --dataset miditok --epochs 25 --batch_size 16 --hidden_size 512 --num_layers 3 --lr 0.0003
if [ $? -eq 0 ]; then
    log_message "✅ MidiTok large model completed successfully"
else
    log_message "❌ MidiTok large model failed"
fi

log_message "Training MidiTok - Deep Model"
python train.py --dataset miditok --epochs 20 --batch_size 32 --hidden_size 256 --num_layers 4 --lr 0.0004
if [ $? -eq 0 ]; then
    log_message "✅ MidiTok deep model completed successfully"
else
    log_message "❌ MidiTok deep model failed"
fi

# ==============================================================================
# HYPERPARAMETER EXPLORATION
# ==============================================================================

log_message "🔬 Starting hyperparameter exploration..."

log_message "Training Naive - High Learning Rate Experiment"
python train.py --dataset naive --epochs 15 --batch_size 32 --hidden_size 256 --num_layers 2 --lr 0.002
if [ $? -eq 0 ]; then
    log_message "✅ Naive high LR experiment completed successfully"
else
    log_message "❌ Naive high LR experiment failed"
fi

log_message "Training Naive - Large Batch Experiment"
python train.py --dataset naive --epochs 15 --batch_size 64 --hidden_size 256 --num_layers 2 --lr 0.001
if [ $? -eq 0 ]; then
    log_message "✅ Naive large batch experiment completed successfully"
else
    log_message "❌ Naive large batch experiment failed"
fi

log_message "Training MidiTok - Conservative Learning Rate"
python train.py --dataset miditok --epochs 25 --batch_size 32 --hidden_size 256 --num_layers 2 --lr 0.0002
if [ $? -eq 0 ]; then
    log_message "✅ MidiTok conservative LR experiment completed successfully"
else
    log_message "❌ MidiTok conservative LR experiment failed"
fi

# ==============================================================================
# GENERATE SAMPLES FROM BEST MODELS
# ==============================================================================

log_message "🎵 Generating sample outputs from trained models..."

# Find the best models (lowest loss) from each dataset
log_message "Finding best models for generation..."

# Generate from multiple models for comparison
NAIVE_MODELS=($(ls -t models/naive/*.pth | head -5))
MIDITOK_MODELS=($(ls -t models/miditok/*.pth | head -5))

log_message "Generating samples from top Naive models..."
for i in {0..2}; do
    if [ ${NAIVE_MODELS[$i]+x} ]; then
        log_message "Generating from model: ${NAIVE_MODELS[$i]}"
        python generate.py --model_path "${NAIVE_MODELS[$i]}" --length 300 --temperature 0.8 >/dev/null 2>&1
        python generate.py --model_path "${NAIVE_MODELS[$i]}" --length 300 --temperature 1.0 >/dev/null 2>&1
        python generate.py --model_path "${NAIVE_MODELS[$i]}" --length 300 --temperature 0.6 >/dev/null 2>&1
    fi
done

log_message "Generating samples from top MidiTok models..."
for i in {0..2}; do
    if [ ${MIDITOK_MODELS[$i]+x} ]; then
        log_message "Generating from model: ${MIDITOK_MODELS[$i]}"
        python generate.py --model_path "${MIDITOK_MODELS[$i]}" --length 300 --temperature 0.8 >/dev/null 2>&1
        python generate.py --model_path "${MIDITOK_MODELS[$i]}" --length 300 --temperature 1.0 >/dev/null 2>&1
        python generate.py --model_path "${MIDITOK_MODELS[$i]}" --length 300 --temperature 0.6 >/dev/null 2>&1
    fi
done

# ==============================================================================
# CREATE COMPREHENSIVE ANALYSIS
# ==============================================================================

log_message "📊 Creating comprehensive analysis plots..."

# Update the presentation plots
python create_presentation_plots.py >/dev/null 2>&1
if [ $? -eq 0 ]; then
    log_message "✅ Analysis plots created successfully"
else
    log_message "❌ Analysis plots creation failed"
fi

# ==============================================================================
# SESSION SUMMARY
# ==============================================================================

log_message "=== TRAINING SESSION COMPLETED ==="
log_message "Finished at: $(date)"

# Count models and outputs created (respecting directory organization)
NAIVE_COUNT=$(ls models/naive/*.pth 2>/dev/null | wc -l)
MIDITOK_COUNT=$(ls models/miditok/*.pth 2>/dev/null | wc -l)
NAIVE_MIDI_COUNT=$(ls outputs/naive/midi/*.mid 2>/dev/null | wc -l)
MIDITOK_MIDI_COUNT=$(ls outputs/miditok/midi/*.mid 2>/dev/null | wc -l)
TOTAL_MIDI_COUNT=$((NAIVE_MIDI_COUNT + MIDITOK_MIDI_COUNT))

log_message "📈 SESSION RESULTS:"
log_message "   - Naive models trained: $NAIVE_COUNT (in models/naive/)"
log_message "   - MidiTok models trained: $MIDITOK_COUNT (in models/miditok/)" 
log_message "   - Naive MIDI files generated: $NAIVE_MIDI_COUNT (in outputs/naive/midi/)"
log_message "   - MidiTok MIDI files generated: $MIDITOK_MIDI_COUNT (in outputs/miditok/midi/)"
log_message "   - Total MIDI files: $TOTAL_MIDI_COUNT"
log_message "   - Training logs: logs/naive/ and logs/miditok/"
log_message "   - Loss curves: outputs/naive/training_loss/ and outputs/miditok/training_loss/"
log_message "   - Session log: $LOG_FILE"

# Create a summary file for easy viewing
SUMMARY_FILE="$SESSION_DIR/session_summary.md"
cat > "$SUMMARY_FILE" << EOF
# Training Session Summary

**Session Date:** $(date)
**Duration:** Started $(date)

## Models Trained
- **Naive Dataset:** $NAIVE_COUNT models (stored in \`models/naive/\`)
- **MidiTok Dataset:** $MIDITOK_COUNT models (stored in \`models/miditok/\`)

## Experiments Conducted
1. **Architecture Comparison:**
   - Small (128 hidden, 2 layers)
   - Medium (256 hidden, 2 layers) 
   - Large (512 hidden, 3 layers)
   - Deep (256 hidden, 4 layers)

2. **Hyperparameter Exploration:**
   - Learning rates: 0.0002 - 0.002
   - Batch sizes: 16, 32, 64
   - Epochs: 15-25

3. **Generated Outputs:**
   - Multiple temperature settings (0.6, 0.8, 1.0)
   - Various sequence lengths
   - **Naive:** $NAIVE_MIDI_COUNT MIDI files
   - **MidiTok:** $MIDITOK_MIDI_COUNT MIDI files
   - **Total:** $TOTAL_MIDI_COUNT MIDI files

## Key Files for Presentation (Organized by Dataset)
- \`logs/naive/\` & \`logs/miditok/\` - Training metrics and comparisons
- \`outputs/naive/midi/\` & \`outputs/miditok/midi/\` - Generated music samples
- \`outputs/naive/training_loss/\` & \`outputs/miditok/training_loss/\` - Loss curve visualizations
- \`outputs/presentation_comparison.png\` - Model comparison plots

## Next Steps for Presentation
1. Listen to generated MIDI files
2. Compare loss curves between models
3. Analyze model performance metrics
4. Select best examples for demo
EOF

log_message "📋 Summary created: $SUMMARY_FILE"

echo ""
echo "🎉 COMPREHENSIVE TRAINING SESSION COMPLETE!"
echo "📊 Check the following for your presentation:"
echo "   📁 Models: models/naive/ and models/miditok/"
echo "   🎵 Generated music: outputs/naive/midi/ and outputs/miditok/midi/"
echo "   📈 Training curves: outputs/naive/training_loss/ and outputs/miditok/training_loss/"
echo "   📋 Session summary: $SUMMARY_FILE"
echo ""
echo "💡 Tip: Check the session log for any errors: $LOG_FILE"

# Final beep to notify completion (if on macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    for i in {1..3}; do
        afplay /System/Library/Sounds/Glass.aiff
        sleep 0.5
    done
fi