#!/usr/bin/env python3
"""
Comprehensive discriminator training script - trains MLP, LSTM, and Transformer models
Run this script and go away - it will train all models with multiple configurations
"""

import subprocess
import time
import sys
import os
from datetime import datetime
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors gracefully"""
    print(f"\n{'='*60}")
    print(f"Starting: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        
        print(f"SUCCESS: {description}")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        
        # Print last few lines of output
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print("Output:")
            for line in lines[-5:]:  # Last 5 lines
                print(f"  {line}")
                
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"FAILED: {description}")
        print(f"Duration: {duration:.1f}s")
        print(f"Error: {e}")
        print(f"Stderr: {e.stderr}")
        return False

def main():
    print("Discriminator Training Marathon Starting!")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to project root directory (parent of scripts)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}")
    
    # Ensure we're in the right directory
    if not Path("train_discriminator.py").exists():
        print("train_discriminator.py not found. Check project structure.")
        sys.exit(1)
    
    # Ensure data exists
    if not Path("data/measures/measure_sequences.npy").exists():
        print("Measure dataset not found. Run 'python utils/measure_dataset.py' first.")
        sys.exit(1)
    
    total_start = time.time()
    success_count = 0
    total_count = 0
    
    # Training configurations
    # Format: (model_type, epochs, batch_size, lr, context, description)
    configs = [
        # MLP Models - Fast training, good baseline
        ("mlp", 15, 128, 1e-3, 4, "MLP Baseline (4 measures context)"),
        ("mlp", 20, 64, 5e-4, 8, "MLP Extended (8 measures context)"),
        ("mlp", 25, 128, 1e-3, 4, "MLP Long Training (4 measures)"),
        
        # LSTM Models - Sequential modeling
        ("lstm", 20, 64, 1e-3, 4, "LSTM Baseline (4 measures context)"),
        ("lstm", 25, 32, 5e-4, 8, "LSTM Extended (8 measures context)"),
        ("lstm", 30, 64, 1e-3, 4, "LSTM Long Training (4 measures)"),
        
        # Transformer Models - Attention-based (most resource intensive)
        ("transformer", 15, 32, 1e-4, 4, "Transformer Baseline (4 measures)"),
        ("transformer", 20, 16, 5e-5, 8, "Transformer Extended (8 measures)"),
        ("transformer", 25, 32, 1e-4, 4, "Transformer Long Training (4 measures)"),
    ]
    
    print(f"Training {len(configs)} different configurations:")
    for i, (model, epochs, batch, lr, ctx, desc) in enumerate(configs, 1):
        print(f"  {i:2d}. {desc}")
    
    print(f"\nStarting training marathon...")
    
    for i, (model_type, epochs, batch_size, lr, context, description) in enumerate(configs, 1):
        total_count += 1
        
        print(f"\n📍 Configuration {i}/{len(configs)}")
        
        cmd = [
            "python", "train_discriminator.py",
            "--model_type", model_type,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--context", str(context),
            "--device", "auto",  # Auto-detect best device
            "--data_dir", "data/measures"
        ]
        
        if run_command(cmd, description):
            success_count += 1
        else:
            print(f"⚠️  Continuing with next model despite failure...")
        
        # Brief pause between models
        print(f"Brief pause before next model...")
        time.sleep(5)
    
    # Final summary
    total_duration = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TRAINING MARATHON COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes, {total_duration/3600:.1f} hours)")
    print(f"Success rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Results summary
    print(f"\nCheck your results:")
    print(f"  • Model files: models/discriminators/*/")
    print(f"  • Training logs: logs/discriminators/train_summary.csv")
    
    if success_count == total_count:
        print(f"\nPerfect run! All models trained successfully!")
    elif success_count > 0:
        print(f"\nPartial success. {success_count} models completed.")
    else:
        print(f"\nNo models completed successfully. Check errors above.")

if __name__ == "__main__":
    main()