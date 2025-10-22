#!/usr/bin/env python3
"""
Quick Transformer discriminator training
"""

import subprocess
import time
from datetime import datetime

def run_transformer_training():
    print("🤖 Transformer Discriminator Training")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Transformer configurations - fast and reasonable
    configs = [
        # (epochs, batch_size, lr, context, description)
        (5, 32, 1e-4, 4, "Transformer Quick (5 epochs)"),
        (8, 24, 1e-4, 4, "Transformer Standard (8 epochs)"),
        (10, 16, 5e-5, 4, "Transformer Extended (10 epochs, smaller batch)"),
        (6, 16, 1e-4, 6, "Transformer Large Context (6 measures)"),
    ]
    
    for i, (epochs, batch_size, lr, context, description) in enumerate(configs, 1):
        print(f"\n🚀 Configuration {i}/{len(configs)}: {description}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        
        cmd = [
            "python", "train_discriminator.py",
            "--model_type", "transformer",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--context", str(context),
            "--device", "auto",
            "--data_dir", "data/measures"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True)
            duration = time.time() - start_time
            print(f"✅ SUCCESS: {description} ({duration/60:.1f} minutes)")
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"❌ FAILED: {description} ({duration/60:.1f} minutes)")
            print(f"Error: {e}")
        
        time.sleep(2)  # Brief pause
    
    print(f"\n🎉 Transformer training complete!")
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
    print("\n📊 Check results:")
    print("  • Models: models/discriminators/transformer/")
    print("  • Logs: logs/discriminators/train_summary.csv")

if __name__ == "__main__":
    run_transformer_training()