#!/bin/bash
# Comprehensive training script for all generator and discriminator architectures
# This will run multiple training experiments while you're away

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

# Create timestamp for this training session
SESSION_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/training_session_${SESSION_TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "MIDI Music Generation Training Session"
echo "Started: $(date)"
echo "Session ID: $SESSION_TIMESTAMP"
echo "=========================================="
echo ""

# Function to log with timestamp
log() {
    echo "[$(date +"%H:%M:%S")] $1"
}

# Function to train and log
train_model() {
    local model_type=$1
    local script=$2
    local args=$3
    local log_file="$LOG_DIR/${model_type}_${script##*/}.log"
    
    log "Starting: $model_type ($script)"
    echo "Command: python $script $args" | tee "$log_file"
    
    if python $script $args 2>&1 | tee -a "$log_file"; then
        log "✅ Completed: $model_type"
        return 0
    else
        log "❌ Failed: $model_type (check $log_file)"
        return 1
    fi
}

# Track successes and failures
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_START=$(date +%s)

echo "=========================================="
echo "PHASE 1: GENERATOR TRAINING"
echo "=========================================="
echo ""

# LSTM Generator - baseline (20 epochs, ~15-20 minutes)
log "Training LSTM Generator..."
if train_model "lstm_generator" "training/train_generator.py" \
    "--model_type lstm --epochs 20 --batch_size 64 --lr 0.001 --hidden_size 256 --num_layers 2"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# GRU Generator - faster alternative (20 epochs, ~12-15 minutes)
log "Training GRU Generator..."
if train_model "gru_generator" "training/train_generator.py" \
    "--model_type gru --epochs 20 --batch_size 64 --lr 0.001 --hidden_size 256 --num_layers 2"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# Transformer Generator - more powerful (15 epochs, ~20-25 minutes)
log "Training Transformer Generator..."
if train_model "transformer_generator" "training/train_generator.py" \
    "--model_type transformer --epochs 15 --batch_size 32 --lr 0.0001 --d_model 256 --nhead 8 --num_layers 4"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# LSTM Generator - larger model (15 epochs, ~20-25 minutes)
log "Training Large LSTM Generator..."
if train_model "lstm_large_generator" "training/train_generator.py" \
    "--model_type lstm --epochs 15 --batch_size 32 --lr 0.001 --hidden_size 512 --num_layers 3"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

echo "=========================================="
echo "PHASE 2: DISCRIMINATOR TRAINING (Chords)"
echo "=========================================="
echo ""

# MLP Discriminator with chords (15 epochs, ~5 minutes)
log "Training MLP Discriminator (chords)..."
if train_model "mlp_discriminator_chords" "training/train_discriminator.py" \
    "--model_type mlp --label_mode chords --epochs 15 --context_measures 4 --batch_size 64 --lr 0.001"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# LSTM Discriminator with chords (15 epochs, ~10 minutes)
log "Training LSTM Discriminator (chords)..."
if train_model "lstm_discriminator_chords" "training/train_discriminator.py" \
    "--model_type lstm --label_mode chords --epochs 15 --context_measures 4 --batch_size 64 --lr 0.001"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# Transformer Discriminator with chords (15 epochs, ~15 minutes)
log "Training Transformer Discriminator (chords)..."
if train_model "transformer_discriminator_chords" "training/train_discriminator.py" \
    "--model_type transformer --label_mode chords --epochs 15 --context_measures 4 --batch_size 32 --lr 0.0001"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

echo "=========================================="
echo "PHASE 3: DISCRIMINATOR TRAINING (Pitches)"
echo "=========================================="
echo ""

# MLP Discriminator with pitches (15 epochs, ~5 minutes)
log "Training MLP Discriminator (pitches)..."
if train_model "mlp_discriminator_pitches" "training/train_discriminator.py" \
    "--model_type mlp --label_mode pitches --epochs 15 --context_measures 4 --batch_size 64 --lr 0.001"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# LSTM Discriminator with pitches (15 epochs, ~10 minutes)
log "Training LSTM Discriminator (pitches)..."
if train_model "lstm_discriminator_pitches" "training/train_discriminator.py" \
    "--model_type lstm --label_mode pitches --epochs 15 --context_measures 4 --batch_size 64 --lr 0.001"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

echo "=========================================="
echo "PHASE 4: ADDITIONAL EXPERIMENTS"
echo "=========================================="
echo ""

# Different context sizes for discriminators
log "Training MLP Discriminator (context=6, chords)..."
if train_model "mlp_discriminator_ctx6" "training/train_discriminator.py" \
    "--model_type mlp --label_mode chords --epochs 15 --context_measures 6 --batch_size 64 --lr 0.001"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# GRU Generator with different hyperparameters
log "Training Large GRU Generator..."
if train_model "gru_large_generator" "training/train_generator.py" \
    "--model_type gru --epochs 20 --batch_size 32 --lr 0.001 --hidden_size 512 --num_layers 3"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# Calculate total time
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_TIME / 3600))
TOTAL_MINS=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "=========================================="
echo "TRAINING SESSION COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Total Time: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "Total: $((SUCCESS_COUNT + FAIL_COUNT))"
echo ""
echo "Logs saved to: $LOG_DIR"
echo "Models saved to: models/"
echo "Training logs: logs/discriminators/train_summary.csv"
echo "                logs/naive/models.csv"
echo "=========================================="
echo ""

# Create summary report
SUMMARY_FILE="$LOG_DIR/summary.txt"
{
    echo "Training Session Summary"
    echo "========================"
    echo "Session ID: $SESSION_TIMESTAMP"
    echo "Started: $(date)"
    echo "Total Time: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
    echo ""
    echo "Results:"
    echo "  Successful: $SUCCESS_COUNT"
    echo "  Failed: $FAIL_COUNT"
    echo "  Total: $((SUCCESS_COUNT + FAIL_COUNT))"
    echo ""
    echo "Models Trained:"
    echo "  Generators: 5 (LSTM, GRU, Transformer, LSTM-Large, GRU-Large)"
    echo "  Discriminators: 6 (MLP/LSTM/Transformer × chords/pitches, MLP-ctx6)"
    echo ""
    echo "Check individual log files in $LOG_DIR for details"
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

# Exit with appropriate code
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi
