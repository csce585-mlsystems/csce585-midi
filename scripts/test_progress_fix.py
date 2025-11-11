#!/usr/bin/env python3
"""
Quick test to verify progress bars work correctly without printing multiple bars.
"""

print("Testing progress bar fix...")
print("This will train for 1 epoch with 200 batches to verify:")
print("  1. Single progress bar that updates in place")
print("  2. Shows both current loss and average loss")
print("  3. Validation bar doesn't clutter output")
print("\nStarting test...\n")

import subprocess
import sys
from pathlib import Path

result = subprocess.run([
    "python", "training/train_generator.py",
    "--dataset", "miditok_augmented",
    "--model_type", "lstm",
    "--epochs", "1",
    "--batch_size", "128",
    "--max_batches", "200",
    "--hidden_size", "256",
    "--num_layers", "2",
    "--val_split", "0.1",
    "--device", "cuda" if __import__("torch").cuda.is_available() else "mps" if __import__("torch").backends.mps.is_available() else "cpu"
], cwd=Path(__file__).parent.parent)

print("\n" + "="*60)
if result.returncode == 0:
    print("✅ Test completed!")
    print("="*60)
    print("\nYou should have seen:")
    print("  ✓ ONE training progress bar that updated in place")
    print("  ✓ Loss values showing current and average")
    print("  ✓ ONE validation progress bar (disappeared after)")
    print("  ✓ Clean epoch summary at the end")
    print("\n💡 If you saw multiple progress bars spamming,")
    print("   the fix didn't work. Otherwise, you're good!")
else:
    print("❌ Test failed with return code:", result.returncode)
    sys.exit(1)
