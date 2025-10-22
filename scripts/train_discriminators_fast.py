#!/usr/bin/env python3
"""
Fast discriminator training - reasonable epoch counts and batch sizes
"""

import subprocess
import time
import sys
import os
from datetime import datetime
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors gracefully"""
    print(f"\n{'='*50}")
    print(f"Starting: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        
        print(f"SUCCESS: {description}")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        
        # Print last few lines of output
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print("Last output:")
            for line in lines[-3:]:  # Last 3 lines
                print(f"  {line}")
                
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"FAILED: {description}")
        print(f"Duration: {duration:.1f}s")
        print(f"Error: {e}")
        return False

def main():
    print("Fast Discriminator Training (Reasonable Configs)")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Change to project root directory (parent of scripts)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Ensure data exists
    if not Path("data/measures/measure_sequences.npy").exists():
        print("Measure dataset not found.")
        sys.exit(1)
    
    total_start = time.time()
    success_count = 0
    total_count = 0
    
    # FAST Training configurations - much more reasonable
    configs = [
        # MLP Models - Fast baseline
        ("mlp", 10, 128, 1e-3, 4, "MLP Baseline (4 measures, 10 epochs)"),
        ("mlp", 15, 64, 1e-3, 4, "MLP Extended (4 measures, 15 epochs)"),
        
        # LSTM Models - Reasonable epochs and batch sizes
        ("lstm", 8, 64, 1e-3, 4, "LSTM Baseline (4 measures, 8 epochs)"),
        ("lstm", 10, 32, 1e-3, 4, "LSTM Extended (4 measures, 10 epochs)"),
        ("lstm", 6, 16, 5e-4, 6, "LSTM Large Context (6 measures, 6 epochs, small batch)"),
        
        # Transformer Models - Small and fast
        ("transformer", 5, 32, 1e-4, 4, "Transformer Quick (4 measures, 5 epochs)"),
        ("transformer", 8, 16, 1e-4, 4, "Transformer Extended (4 measures, 8 epochs)"),
    ]
    
    print(f"Training {len(configs)} FAST configurations:")
    for i, (model, epochs, batch, lr, ctx, desc) in enumerate(configs, 1):
        print(f"  {i:2d}. {desc}")
    
    print(f"\nStarting FAST training...")
    
    for i, (model_type, epochs, batch_size, lr, context, description) in enumerate(configs, 1):
        total_count += 1
        
        print(f"\nConfiguration {i}/{len(configs)}")
        
        cmd = [
            "python", "train_discriminator.py",
            "--model_type", model_type,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--context", str(context),
            "--device", "auto",
            "--data_dir", "data/measures"
        ]
        
        if run_command(cmd, description):
            success_count += 1
        else:
            print(f"Continuing with next model...")
        
        # Brief pause
        time.sleep(3)
    
    # Final summary
    total_duration = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"FAST TRAINING COMPLETE!")
    print(f"{'='*50}")
    print(f"Total time: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)")
    print(f"Success rate: {success_count}/{total_count}")
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
    
    print(f"\nCheck results:")
    print(f"  • Models: models/discriminators/*/")
    print(f"  • Logs: logs/discriminators/train_summary.csv")

if __name__ == "__main__":
    main()