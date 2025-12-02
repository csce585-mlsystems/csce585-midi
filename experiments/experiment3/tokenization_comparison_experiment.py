"""
Experiment 3: Tokenization Strategy Comparison
===============================================

This experiment compares how different tokenization strategies affect
the quality of generated music using the same LSTM architecture.

Models compared:
1. Naive tokenization (one token per note/chord, simple representation)
2. MidiTok REMI tokenization (structured representation with duration, velocity, etc.)
3. MidiTok REMI + Data Augmentation (same tokenization but with transposed training data)

Hypothesis: Different tokenization strategies will produce different musical characteristics:
- Naive: May produce simpler rhythmic patterns but potentially more coherent melodies
- MidiTok: Should capture more nuanced musical information (dynamics, timing)
- MidiTok + Augmentation: Should generalize better across keys, less overfitting

Metrics evaluated (from evaluate.py):
- Note Density: notes per second
- Pitch Range: difference between highest and lowest pitch
- Average Polyphony: average number of simultaneous notes
- Scale Consistency: how well notes fit common scales
- Pitch Entropy: diversity of pitches used
- Pitch Class Entropy: diversity of pitch classes (ignoring octaves)
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
    "naive": {
        "name": "Naive Tokenization",
        "model_path": "models/generators/checkpoints/nottingham-dataset-final_experiments_naive/lstm_20251129_200244.pth",
        "data_dir": "data/nottingham-dataset-final_experiments_naive",
        "description": "Simple one-token-per-note representation"
    },
    "miditok": {
        "name": "MidiTok REMI",
        "model_path": "models/generators/checkpoints/nottingham-dataset-final_experiments_miditok/lstm_20251130_091025.pth",
        "data_dir": "data/nottingham-dataset-final_experiments_miditok",
        "description": "Structured REMI tokenization with duration, velocity, and timing"
    },
    "miditok_augmented": {
        "name": "MidiTok REMI + Augmentation",
        "model_path": "models/generators/checkpoints/nottingham-dataset-final_experiments_miditok_augmented/lstm_20251201_170956.pth",
        "data_dir": "data/nottingham-dataset-final_experiments_miditok_augmented",
        "description": "REMI tokenization with key transposition augmentation"
    }
}

# Generation settings to test
GENERATION_SETTINGS = [
    {"name": "conservative", "strategy": "top_p", "p": 0.85, "temperature": 0.8},
    {"name": "balanced", "strategy": "top_p", "p": 0.9, "temperature": 1.0},
    {"name": "creative", "strategy": "top_p", "p": 0.95, "temperature": 1.2},
]


def run_experiment(
    output_dir="experiments/experiment3/results",
    num_samples_per_setting=10,
    generate_length=200,
    seq_length=50,
):
    """
    Run the tokenization comparison experiment.
    
    Args:
        output_dir: Directory to store results and generated MIDI files
        num_samples_per_setting: Number of samples to generate per model/setting combination
        generate_length: Number of tokens to generate per sample
        seq_length: Sequence length for the model (should match training)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = {
        "experiment": "tokenization_comparison",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_samples_per_setting": num_samples_per_setting,
            "generate_length": generate_length,
            "seq_length": seq_length,
            "generation_settings": GENERATION_SETTINGS
        },
        "models": MODELS,
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
    for model_key, model_config in MODELS.items():
        print("\n" + "=" * 80)
        print(f"TESTING MODEL: {model_config['name']}")
        print(f"Description: {model_config['description']}")
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
                    # Generate sample - use relative path for data_dir to avoid nested directory issues
                    generate(
                        model_type="lstm",
                        data_dir=model_config["data_dir"],  # Use relative path
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
                    dataset_name = model_config["data_dir"].split("/")[-1]
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
    print("EXPERIMENT SUMMARY: Tokenization Strategy Comparison")
    print("=" * 80)
    
    # Key metrics to highlight
    key_metrics = ["scale_consistency", "pitch_entropy", "pitch_class_entropy", "note_density", "pitch_range"]
    
    for model_key, model_results in results.get("results", {}).items():
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
    Scale Consistency: Higher = notes fit better in traditional scales (0-1)
    Pitch Entropy: Higher = more diverse pitch usage 
    Pitch Class Entropy: Higher = more even distribution across 12 pitch classes
    Note Density: Notes per second
    Pitch Range: Range of pitches in semitones
    
    Expected patterns:
    - Augmented model should show more consistent scale usage across keys
    - MidiTok models may show more varied note density (captures timing better)
    - Naive model may have simpler, more repetitive patterns (lower entropy)
    """)


def compare_models(results_file):
    """Load saved results and generate comparison analysis."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tokenization comparison experiment")
    parser.add_argument("--output_dir", type=str, default="experiments/experiment3/results",
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
