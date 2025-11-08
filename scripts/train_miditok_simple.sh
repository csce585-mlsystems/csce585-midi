#!/bin/bash
# Simple training script for MIDITok models
# No early stopping, no validation split - just train for the specified epochs

set -e

source .venv/bin/activate

SESSION_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/training_miditok_simple_${SESSION_TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "MIDITok Generator Training (Simple)"
echo "Started: $(date)"
echo "Session ID: $SESSION_TIMESTAMP"
echo "=========================================="
echo "Configuration:"
echo "  - NO early stopping"
echo "  - NO validation split"
echo "  - Train on 100% of data"
echo "  - Full epochs (no shortcuts)"
echo "=========================================="
echo ""

log() {
    echo "[$(date +"%H:%M:%S")] $1"
}

train_model() {
    local model_name=$1
    shift
    local log_file="$LOG_DIR/${model_name}.log"
    
    log "Starting: $model_name"
    echo "Command: python training/train_generator.py $@" | tee "$log_file"
    echo "Started at: $(date)" | tee -a "$log_file"
    
    python training/train_generator.py "$@" 2>&1 | tee -a "$log_file"
    local EXIT_CODE=$?
    
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

SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_START=$(date +%s)

echo "=========================================="
echo "TRAINING MODELS"
echo "=========================================="
echo ""

# LSTM Generator
log "Training LSTM Generator..."
if train_model "lstm" \
    --model_type lstm \
    --dataset miditok \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.001 \
    --hidden_size 256 \
    --num_layers 2; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# GRU Generator
log "Training GRU Generator..."
if train_model "gru" \
    --model_type gru \
    --dataset miditok \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.001 \
    --hidden_size 256 \
    --num_layers 2; then
    ((SUCCESS_COUNT++))
else
    ((FAIL_COUNT++))
fi
echo ""

# Transformer Generator
log "Training Transformer Generator..."
if train_model "transformer" \
    --model_type transformer \
    --dataset miditok \
    --epochs 15 \
    --batch_size 16 \
    --lr 0.0001 \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3; then
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
echo "TRAINING COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Total Time: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""
echo "Logs: $LOG_DIR"
echo "Models: models/generators/checkpoints/miditok/"
echo "=========================================="

[ $FAIL_COUNT -eq 0 ] && exit 0 || exit 1
