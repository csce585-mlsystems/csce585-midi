#!/bin/bash
# Quick start script for completing P2 milestone
# Run this to begin your final experiments

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     CSCE 585 Project P2 - Quick Start                       ║"
echo "║     Complete Experimental Evaluation                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/train_all.py" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Create directories for results
echo "📁 Creating results directories..."
mkdir -p docs/figures/training_curves
mkdir -p docs/figures/comparisons
mkdir -p docs/figures/ablations
mkdir -p results/experiments

# Start training (this will take 4-6 hours)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Starting Full Training Suite                                ║"
echo "║  Estimated time: 4-6 hours                                   ║"
echo "║                                                              ║"
echo "║  Training:                                                   ║"
echo "║    - 4 Discriminator models                                  ║"
echo "║    - 2 Generator models                                      ║"
echo "║                                                              ║"
echo "║  You can monitor progress in another terminal:              ║"
echo "║    tail -f logs/training_marathon.log                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

read -p "⚠️  This will take 4-6 hours. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled. Run this script when ready for full training."
    exit 0
fi

# Run training in background with logging
echo "🚀 Starting training (running in background)..."
nohup python scripts/train_all.py > logs/training_marathon.log 2>&1 &
TRAIN_PID=$!

echo "✅ Training started with PID: $TRAIN_PID"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f logs/training_marathon.log"
echo ""
echo "⏹️  Stop training if needed:"
echo "   kill $TRAIN_PID"
echo ""
echo "🎯 While training runs, you can:"
echo "   1. Start writing your final report (docs/FINAL_REPORT.md)"
echo "   2. Prepare visualization scripts"
echo "   3. Draft presentation slides"
echo "   4. Review PROJECT_RUBRIC_CHECKLIST.md"
echo ""
echo "✨ Training complete notification will be in logs/training_marathon.log"
echo ""

# Save PID for later reference
echo $TRAIN_PID > .training_pid
echo "📝 Training PID saved to .training_pid"
