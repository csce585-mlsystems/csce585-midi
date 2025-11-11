#!/usr/bin/env python3
"""
Quick test to verify the progress bar improvements work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing progress bar improvements...")
print("\nThis test will train a tiny model for 2 epochs to verify:")
print("  1. Batch-level progress bars appear during training")
print("  2. Validation progress bars work")
print("  3. Epoch summaries are clear and non-duplicated")
print("  4. Training configuration info is displayed")
print("\nStarting test...\n")

import subprocess
result = subprocess.run([
    "python", "training/train_generator.py",
    "--dataset", "miditok",  # Use existing dataset
    "--model_type", "lstm",
    "--epochs", "2",
    "--batch_size", "64",
    "--max_batches", "50",  # Only 50 batches for quick test
    "--hidden_size", "64",   # Small model
    "--num_layers", "1",
    "--val_split", "0.1",
    "--patience", "10"  # High patience so it doesn't stop early
], cwd=Path(__file__).parent.parent)

if result.returncode == 0:
    print("\n" + "="*60)
    print("✅ Progress bar test completed successfully!")
    print("="*60)
    print("\nYou should have seen:")
    print("  ✓ Training configuration summary at start")
    print("  ✓ Progress bar during training batches")
    print("  ✓ Progress bar during validation")
    print("  ✓ Clear epoch summaries with train/val loss")
    print("  ✓ No duplicate print statements")
else:
    print("\n❌ Test failed with return code:", result.returncode)
    sys.exit(1)
