import subprocess
import time
from pathlib import Path
import sys

def run_command(cmd, description):
    """Run a command and log output"""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout per model
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS - Completed in {duration/60:.1f} minutes")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ FAILED - Exit code: {result.returncode}")
            print("Error:", result.stderr[-500:])
        
        return result.returncode == 0
    
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT - Exceeded 2 hours")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    start_time = time.time()
    results = []
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          COMPREHENSIVE MODEL TRAINING MARATHON               ║
║                                                              ║
║  Training both discriminators and generators                 ║
║  Estimated total time: 4-6 hours                            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Training configurations
    configs = [
        # DISCRIMINATORS - Pitches (multi-label)
        {
            "type": "discriminator",
            "cmd": ["python", "training/train_discriminator.py", 
                   "--model_type", "mlp", "--label_mode", "pitches",
                   "--context_measures", "4", "--epochs", "10", "--batch_size", "64"],
            "desc": "Discriminator MLP (Pitches, 4-measure context)"
        },
        {
            "type": "discriminator",
            "cmd": ["python", "training/train_discriminator.py",
                   "--model_type", "lstm", "--label_mode", "pitches",
                   "--context_measures", "4", "--epochs", "15", "--batch_size", "32"],
            "desc": "Discriminator LSTM (Pitches, 4-measure context)"
        },
        {
            "type": "discriminator",
            "cmd": ["python", "training/train_discriminator.py",
                   "--model_type", "transformer", "--label_mode", "pitches",
                   "--context_measures", "4", "--epochs", "12", "--batch_size", "32"],
            "desc": "Discriminator Transformer (Pitches, 4-measure context)"
        },
        
        # DISCRIMINATORS - Extended context
        {
            "type": "discriminator",
            "cmd": ["python", "training/train_discriminator.py",
                   "--model_type", "lstm", "--label_mode", "pitches",
                   "--context_measures", "6", "--epochs", "12", "--batch_size", "32"],
            "desc": "Discriminator LSTM (Pitches, 6-measure context)"
        },
        
        # GENERATORS - Different architectures
        {
            "type": "generator",
            "cmd": ["python", "training/train_generator.py",
                   "--model_type", "lstm", "--epochs", "20", "--batch_size", "128"],
            "desc": "Generator LSTM (Naive tokenization)"
        },
        {
            "type": "generator",
            "cmd": ["python", "training/train_generator.py",
                   "--model_type", "gru", "--epochs", "20", "--batch_size", "128"],
            "desc": "Generator GRU (Naive tokenization)"
        },
        # Transformer generator is too slow (6.4M params vs 971K for LSTM)
        # {
        #     "type": "generator",
        #     "cmd": ["python", "training/train_generator.py",
        #            "--model_type", "transformer", "--epochs", "10", "--batch_size", "64",
        #            "--lr", "5e-4"],
        #     "desc": "Generator Transformer (Naive tokenization)"
        # },
    ]
    
    # Run each configuration
    for i, config in enumerate(configs, 1):
        print(f"\n\n📊 Configuration {i}/{len(configs)}")
        success = run_command(config["cmd"], config["desc"])
        results.append({
            "config": config["desc"],
            "success": success
        })
    
    # Print summary
    total_time = time.time() - start_time
    print(f"""
    
╔══════════════════════════════════════════════════════════════╗
║                    TRAINING COMPLETE                         ║
╚══════════════════════════════════════════════════════════════╝

Total time: {total_time/3600:.2f} hours

Results Summary:
""")
    
    for i, result in enumerate(results, 1):
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{i}. {status} - {result['config']}")
    
    success_count = sum(1 for r in results if r["success"])
    print(f"\n{'='*80}")
    print(f"Overall: {success_count}/{len(results)} configurations completed successfully")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()