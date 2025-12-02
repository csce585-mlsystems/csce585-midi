"""
Experiment 4: Neural Network Architecture Comparison
=====================================================

This experiment compares how different neural network architectures perform
for music generation using the same (naive) tokenization to avoid confounders.

Models compared:
1. LSTM (baseline) - Long Short-Term Memory network
2. GRU - Gated Recurrent Unit (simpler than LSTM)
3. Transformer (small) - Attention-based architecture

Hypothesis: On small datasets, simpler architectures like LSTM may outperform
transformers, which typically require more data to learn effective representations.
This aligns with scaling-law intuition.

Metrics evaluated:
- Training loss curves (loaded from training logs)
- Generated MIDI quality metrics (from evaluate.py)
- Discriminator scores (optional)
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import shutil
import glob
from datetime import datetime
import csv
import os

# Add project root to path and change to project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # Change to project root so relative paths work

from generate import generate
from evaluate import evaluate_midi


# Model configurations
MODELS = {
    "lstm": {
        "name": "LSTM (Baseline)",
        "model_type": "lstm",
        "model_path": "models/generators/checkpoints/nottingham-dataset-final_experiments_naive/lstm_20251129_200244.pth",
        "loss_file": "logs/generators/nottingham-dataset-final_experiments_naive/models/train_losses_lstm_20251129_200244.npy",
        "summary_file": "logs/generators/nottingham-dataset-final_experiments_naive/models/training_summary_lstm_20251129_200244.txt",
        "description": "Two-layer LSTM with 256 hidden units",
        "params": 941620,
    },
    "gru": {
        "name": "GRU",
        "model_type": "gru",
        "model_path": "models/generators/checkpoints/nottingham-dataset-final_experiments_naive/gru_20251201_171806.pth",
        "loss_file": "logs/generators/nottingham-dataset-final_experiments_naive/models/train_losses_gru_20251201_171806.npy",
        "summary_file": "logs/generators/nottingham-dataset-final_experiments_naive/models/training_summary_gru_20251201_171806.txt",
        "description": "Two-layer GRU with 256 hidden units (fewer params than LSTM)",
        "params": 711220,
    },
    "transformer": {
        "name": "Transformer (Small)",
        "model_type": "transformer",
        "model_path": "models/generators/checkpoints/nottingham-dataset-final_experiments_naive/transformer_20251201_114247.pth",
        "loss_file": "logs/generators/nottingham-dataset-final_experiments_naive/models/train_losses_transformer_20251201_114247.npy",
        "summary_file": "logs/generators/nottingham-dataset-final_experiments_naive/models/training_summary_transformer_20251201_114247.txt",
        "description": "Small transformer with 2 layers, 8 heads, d_model=256",
        "params": 2133556,
    }
}

# Common data directory for all models (naive tokenization)
DATA_DIR = "data/nottingham-dataset-final_experiments_naive"

# Generation settings to test
GENERATION_SETTINGS = [
    {"name": "conservative", "strategy": "top_p", "p": 0.85, "temperature": 0.8},
    {"name": "balanced", "strategy": "top_p", "p": 0.9, "temperature": 1.0},
    {"name": "creative", "strategy": "top_p", "p": 0.95, "temperature": 1.2},
]

MODEL_ORDER = ['lstm', 'gru', 'transformer']
SETTING_ORDER = ['conservative', 'balanced', 'creative']


def load_training_losses():
    """Load training loss curves from saved .npy files."""
    losses = {}
    for model_key, model_config in MODELS.items():
        loss_file = PROJECT_ROOT / model_config["loss_file"]
        if loss_file.exists():
            losses[model_key] = np.load(loss_file)
            print(f"Loaded {len(losses[model_key])} loss values for {model_key}")
        else:
            print(f"Warning: Loss file not found for {model_key}: {loss_file}")
            losses[model_key] = None
    return losses


def parse_training_summary(summary_file):
    """Parse training summary text file for key metrics."""
    summary = {}
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
            
        # Extract key metrics using simple parsing
        for line in content.split('\n'):
            if 'Total Parameters:' in line:
                summary['total_params'] = int(line.split(':')[1].strip().replace(',', ''))
            elif 'Final Training Loss:' in line:
                summary['final_loss'] = float(line.split(':')[1].strip())
            elif 'Best Training Loss:' in line:
                summary['best_loss'] = float(line.split(':')[1].strip())
            elif 'Total Time:' in line:
                summary['training_time'] = line.split(':')[1].strip()
            elif 'Epochs Completed:' in line:
                summary['epochs'] = line.split(':')[1].strip()
                
    except Exception as e:
        print(f"Error parsing summary file {summary_file}: {e}")
        
    return summary


def run_experiment(
    output_dir="experiments/experiment4/results",
    num_samples_per_setting=10,
    generate_length=200,
    seq_length=50,
):
    """
    Run the architecture comparison experiment.
    
    Args:
        output_dir: Directory to store results and generated MIDI files
        num_samples_per_setting: Number of samples to generate per model/setting combination
        generate_length: Number of tokens to generate per sample
        seq_length: Sequence length for the model (should match training)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training loss data first
    print("Loading training loss curves...")
    training_losses = load_training_losses()
    
    # Save training losses for plotting
    losses_dir = output_dir / "training_losses"
    losses_dir.mkdir(exist_ok=True)
    for model_key, losses in training_losses.items():
        if losses is not None:
            np.save(losses_dir / f"{model_key}_losses.npy", losses)
    
    # Load and save training summaries
    training_summaries = {}
    for model_key, model_config in MODELS.items():
        summary_file = PROJECT_ROOT / model_config["summary_file"]
        if summary_file.exists():
            training_summaries[model_key] = parse_training_summary(summary_file)
    
    # Store all results
    all_results = {
        "experiment": "architecture_comparison",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_samples_per_setting": num_samples_per_setting,
            "generate_length": generate_length,
            "seq_length": seq_length,
            "generation_settings": GENERATION_SETTINGS,
            "data_dir": DATA_DIR,
        },
        "models": MODELS,
        "training_summaries": training_summaries,
        "results": {}
    }
    
    # CSV for detailed results
    csv_file = output_dir / "detailed_results.csv"
    csv_fields = [
        "model_type", "setting", "sample_id",
        "num_notes", "duration", "note_density", "pitch_range",
        "avg_polyphony", "scale_consistency", "pitch_entropy", "pitch_class_entropy"
    ]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
    
    # Run experiments for each model
    for model_key in MODEL_ORDER:
        model_config = MODELS[model_key]
        print("\n" + "=" * 80)
        print(f"TESTING MODEL: {model_config['name']}")
        print(f"Description: {model_config['description']}")
        print(f"Parameters: {model_config['params']:,}")
        print(f"Model path: {model_config['model_path']}")
        print("=" * 80)
        
        model_results = {}
        model_output_dir = output_dir / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify model exists
        model_path = PROJECT_ROOT / model_config["model_path"]
        if not model_path.exists():
            print(f"ERROR: Model not found at {model_path}")
            continue
        
        # Test each generation setting
        for setting in GENERATION_SETTINGS:
            print(f"\n--- Setting: {setting['name']} ---")
            print(f"    Strategy: {setting['strategy']}, Temperature: {setting['temperature']}, P: {setting.get('p', 'N/A')}")
            
            setting_dir = model_output_dir / setting["name"]
            setting_dir.mkdir(parents=True, exist_ok=True)
            
            setting_results = []
            
            for i in range(num_samples_per_setting):
                print(f"  Generating sample {i+1}/{num_samples_per_setting}...", end=" ")
                
                try:
                    # Generate sample using the specific model type
                    generate(
                        model_type=model_config["model_type"],  # Use actual model type
                        data_dir=DATA_DIR,
                        strategy=setting["strategy"],
                        generate_length=generate_length,
                        seq_length=seq_length,
                        temperature=setting["temperature"],
                        k=5,
                        p=setting.get("p", 0.9),
                        model_path=str(model_path),
                        discriminator_path=None,
                        discriminator_type=None,
                        seed_style="random"
                    )
                    
                    # Find the generated file
                    dataset_name = DATA_DIR.split("/")[-1]
                    generated_dir = PROJECT_ROOT / f"outputs/generators/{dataset_name}/midi"
                    list_of_files = glob.glob(str(generated_dir / "generated_*.mid"))
                    
                    if list_of_files:
                        latest_file = max(list_of_files, key=lambda x: Path(x).stat().st_mtime)
                        output_file = setting_dir / f"sample_{i+1}.mid"
                        shutil.move(latest_file, output_file)
                        
                        # Evaluate
                        metrics = evaluate_midi(output_file)
                        if metrics:
                            metrics["sample_id"] = i + 1
                            metrics["model_type"] = model_key
                            metrics["setting"] = setting["name"]
                            setting_results.append(metrics)
                            
                            # Write to CSV
                            with open(csv_file, 'a', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=csv_fields)
                                writer.writerow({k: metrics.get(k, '') for k in csv_fields})
                            
                            print(f"Scale: {metrics['scale_consistency']:.3f}, Entropy: {metrics['pitch_entropy']:.3f}")
                        else:
                            print("Evaluation failed")
                    else:
                        print(f"No file generated (checked: {generated_dir})")
                        
                except Exception as e:
                    import traceback
                    print(f"Error: {e}")
                    traceback.print_exc()
            
            # Calculate statistics for this setting
            if setting_results:
                model_results[setting["name"]] = calculate_statistics(setting_results)
        
        all_results["results"][model_key] = model_results
    
    # Save full results
    results_file = output_dir / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print_summary(all_results)
    
    return all_results


def calculate_statistics(results_list):
    """Calculate mean, std, min, max for each metric."""
    metrics = ["num_notes", "duration", "note_density", "pitch_range",
               "avg_polyphony", "scale_consistency", "pitch_entropy", "pitch_class_entropy"]
    
    stats = {}
    for metric in metrics:
        values = [r[metric] for r in results_list if metric in r and r[metric] is not None]
        if values:
            stats[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values)
            }
    
    return stats


def print_summary(results):
    """Print a formatted summary of the results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY: Architecture Comparison")
    print("=" * 80)
    
    # Print training summary first
    print("\n--- Training Performance ---")
    print(f"{'Model':<20} {'Params':>12} {'Final Loss':>12} {'Training Time':>15}")
    print("-" * 60)
    
    for model_key in MODEL_ORDER:
        if model_key in results.get("training_summaries", {}):
            summary = results["training_summaries"][model_key]
            params = summary.get('total_params', MODELS[model_key]['params'])
            final_loss = summary.get('final_loss', 'N/A')
            train_time = summary.get('training_time', 'N/A')
            print(f"{MODELS[model_key]['name']:<20} {params:>12,} {final_loss:>12.4f} {train_time:>15}")
    
    # Key metrics to highlight
    key_metrics = ["scale_consistency", "pitch_entropy", "pitch_class_entropy", "note_density", "pitch_range"]
    
    print("\n--- Generation Metrics ---")
    for model_key in MODEL_ORDER:
        if model_key in results.get("results", {}):
            model_results = results["results"][model_key]
            model_name = MODELS.get(model_key, {}).get("name", model_key)
            print(f"\n{model_name}")
            print("-" * 60)
            
            for setting_name, setting_stats in model_results.items():
                print(f"\n  {setting_name.upper()}:")
                for metric in key_metrics:
                    if metric in setting_stats:
                        stats = setting_stats[metric]
                        print(f"    {metric:25s}: {stats['mean']:8.4f} ± {stats['std']:.4f}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE:")
    print("-" * 80)
    print("""
    Training Loss: Lower = better fit to training data
    
    Scale Consistency: Higher = notes fit better in traditional scales (0-1)
    Pitch Entropy: Higher = more diverse pitch usage 
    Pitch Class Entropy: Higher = more even distribution across 12 pitch classes
    Note Density: Notes per second
    Pitch Range: Range of pitches in semitones
    
    Expected patterns:
    - LSTM may achieve competitive or better results with fewer parameters
    - Transformer may overfit more on small dataset (high training loss gap)
    - GRU should be similar to LSTM but with fewer parameters
    - All models use the same tokenization (naive) to ensure fair comparison
    """)


def compare_models(results_file):
    """Load saved results and generate comparison analysis."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run architecture comparison experiment")
    parser.add_argument("--output_dir", type=str, default="experiments/experiment4/results",
                        help="Directory to store results")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples per model/setting combination")
    parser.add_argument("--generate_length", type=int, default=200,
                        help="Number of tokens to generate per sample")
    parser.add_argument("--seq_length", type=int, default=50,
                        help="Sequence length (should match training)")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Path to existing results file to analyze (skip generation)")
    
    args = parser.parse_args()
    
    if args.analyze:
        compare_models(args.analyze)
    else:
        run_experiment(
            output_dir=args.output_dir,
            num_samples_per_setting=args.num_samples,
            generate_length=args.generate_length,
            seq_length=args.seq_length
        )
