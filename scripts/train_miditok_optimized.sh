#!/bin/bash
# Optimized training script for MIDITok models with early stopping and lazy loading
# Uses subsample_ratio to train faster on large datasets

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

# Create timestamp for this training session
SESSION_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/training_miditok_optimized_${SESSION_TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "MIDITok Generator Training (Optimized)"
echo "Started: $(date)"
echo "Session ID: $SESSION_TIMESTAMP"
echo "=========================================="
echo "Key improvements:"
echo "  - Lazy dataset loading (no memory explosion)"
echo "  - Early stopping (stops when overfitting)"
echo "  - Validation split (10%)"
echo "  - Reduced epochs (max 15, stops earlier if overfitting)"
echo "  - Optional subsampling for even faster training"
echo "=========================================="
echo ""

# Function to log with timestamp
log() {
    echo "[$(date +"%H:%M:%S")] $1"
}

train_model() {
    local model_name=$1
    local args=$2
    local log_file="$LOG_DIR/${model_name}.log"
    local timeout_seconds=$((3 * 3600))  # 3 hours in seconds
    
    log "Starting: $model_name"
    echo "Command: python training/train_generator.py $args" | tee "$log_file"
    echo "Timeout: 3 hours" | tee -a "$log_file"
    echo "Started at: $(date)" | tee -a "$log_file"
    
    # Start the training process in background
    python training/train_generator.py $args 2>&1 | tee -a "$log_file" &
    local PID=$!
    
    # Create a timeout killer in background
    (
        sleep $timeout_seconds
        if kill -0 $PID 2>/dev/null; then
            log "⏱️  TIMEOUT: $model_name (exceeded 3 hours) - killing PID $PID"
            # Kill the entire process group
            pkill -P $PID 2>/dev/null || true
            kill -9 $PID 2>/dev/null || true
            echo "TIMEOUT - Process killed at $(date)" | tee -a "$log_file"
        fi
    ) &
    local TIMEOUT_PID=$!
    
    # Wait for the training to complete
    wait $PID
    local EXIT_CODE=$?
    
    # Kill the timeout killer if training finished first
    kill $TIMEOUT_PID 2>/dev/null || true
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "✅ Completed: $model_name"
        echo "Completed at: $(date)" | tee -a "$log_file"
        return 0
    else
        log "❌ Failed: $model_name (exit code: $EXIT_CODE)"
        echo "Exit code: $EXIT_CODE at $(date)" | tee -a "$log_file"
        return 1
    fi
}

# Track successes and failures
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_START=$(date +%s)

# Common arguments for all models
COMMON_ARGS="--dataset miditok --val_split 0.1 --patience 3"

echo "=========================================="
echo "PHASE 1: BASELINE MODELS (Full Dataset)"
echo "=========================================="
echo ""


# GRU Generator
log "Training GRU Generator..."
if train_model "miditok_gru" \
    "$COMMON_ARGS --model_type gru --epochs 15 --batch_size 64 --lr 0.001 --hidden_size 256 --num_layers 2"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# Transformer Generator - reduced capacity for speed
log "Training Transformer Generator..."
if train_model "miditok_transformer" \
    "$COMMON_ARGS --model_type transformer --epochs 15 --batch_size 16 --lr 0.0001 --d_model 256 --nhead 8 --num_layers 3"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

echo "=========================================="
echo "PHASE 2: LARGER MODELS"
echo "=========================================="
echo ""

# LSTM Large
log "Training Large LSTM Generator..."
if train_model "miditok_lstm_large" \
    "$COMMON_ARGS --model_type lstm --epochs 12 --batch_size 32 --lr 0.001 --hidden_size 512 --num_layers 3"; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# GRU Large
log "Training Large GRU Generator..."
if train_model "miditok_gru_large" \
    "$COMMON_ARGS --model_type gru --epochs 12 --batch_size 32 --lr 0.001 --hidden_size 512 --num_layers 3"; then
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
echo "Models saved to: models/generators/checkpoints/miditok/"
echo "Training logs: logs/generators/miditok/models.csv"
echo "=========================================="
echo ""

# Create summary report
SUMMARY_FILE="$LOG_DIR/summary.txt"
{
    echo "MIDITok Generator Training Summary (Optimized)"
    echo "=============================================="
    echo "Session ID: $SESSION_TIMESTAMP"
    echo "Total Time: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
    echo ""
    echo "Results:"
    echo "  Successful: $SUCCESS_COUNT"
    echo "  Failed: $FAIL_COUNT"
    echo ""
    echo "Models Trained:"
    echo "  - LSTM (256 hidden, 2 layers)"
    echo "  - GRU (256 hidden, 2 layers)"
    echo "  - Transformer (256 d_model, 8 heads, 3 layers)"
    echo "  - LSTM Large (512 hidden, 3 layers)"
    echo "  - GRU Large (512 hidden, 3 layers)"
    echo ""
    echo "Improvements Used:"
    echo "  ✓ Lazy dataset loading (no memory issues)"
    echo "  ✓ Early stopping with patience=3"
    echo "  ✓ 10% validation split"
    echo "  ✓ Reduced epochs (15 max, stops earlier if overfitting)"
    echo "  ✓ 3-hour timeout per model"
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