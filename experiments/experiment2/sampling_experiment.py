"""
Experiment 2: Sampling Strategy Comparison
==========================================

This experiment compares different sampling strategies on a trained generator model:
1. Greedy (always pick highest probability token)
2. Top-k (k=5) - sample from top k most likely tokens
3. Top-p (p=0.9) - nucleus sampling
4. Temperature sampling (various temperatures)

Metrics evaluated:
- Note Density (notes per second)
- Pitch Range (max - min pitch)
- Average Polyphony (simultaneous notes)
- Scale Consistency (how well notes fit common scales)
- Pitch Entropy (diversity of pitches used)
- Pitch Class Entropy (diversity of pitch classes)

Expected findings:
- Greedy: repetitive, low diversity
- Top-p: best musical diversity
- Top-k: moderate diversity
- Temperature: higher temp = more diversity but potentially less coherent
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import shutil
import glob
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from generate import generate
from evaluate import evaluate_midi


def run_sampling_experiment(
    generator_model_path,
    data_dir,
    output_dir,
    num_samples=30,
    generate_length=200,
    seq_length=50
):
    """
    Run sampling strategy comparison experiment.
    
    Args:
        generator_model_path: Path to trained generator model
        data_dir: Path to dataset directory
        output_dir: Where to save results
        num_samples: Number of samples per condition
        generate_length: Length of each generated sequence
        seq_length: Seed sequence length
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define sampling strategies to test
    sampling_strategies = {
        "greedy": {
            "strategy": "greedy",
            "temperature": 1.0,
            "k": 5,
            "p": 0.9,
            "description": "Greedy sampling - always picks highest probability token"
        },
        "top_k_5": {
            "strategy": "top_k",
            "temperature": 1.0,
            "k": 5,
            "p": 0.9,
            "description": "Top-k sampling with k=5"
        },
        "top_k_10": {
            "strategy": "top_k",
            "temperature": 1.0,
            "k": 10,
            "p": 0.9,
            "description": "Top-k sampling with k=10"
        },
        "top_p_0.9": {
            "strategy": "top_p",
            "temperature": 1.0,
            "k": 5,
            "p": 0.9,
            "description": "Nucleus sampling with p=0.9"
        },
        "top_p_0.95": {
            "strategy": "top_p",
            "temperature": 1.0,
            "k": 5,
            "p": 0.95,
            "description": "Nucleus sampling with p=0.95"
        },
        "temp_0.5": {
            "strategy": "random",
            "temperature": 0.5,
            "k": 5,
            "p": 0.9,
            "description": "Temperature sampling with T=0.5 (more conservative)"
        },
        "temp_1.0": {
            "strategy": "random",
            "temperature": 1.0,
            "k": 5,
            "p": 0.9,
            "description": "Temperature sampling with T=1.0 (baseline)"
        },
        "temp_1.5": {
            "strategy": "random",
            "temperature": 1.5,
            "k": 5,
            "p": 0.9,
            "description": "Temperature sampling with T=1.5 (more creative)"
        }
    }
    
    results = {
        "experiment": "sampling_strategy_comparison",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator_model": generator_model_path,
        "data_dir": data_dir,
        "num_samples": num_samples,
        "generate_length": generate_length,
        "strategies": {}
    }
    
    # Run each sampling strategy
    for strategy_name, params in sampling_strategies.items():
        print("\n" + "=" * 80)
        print(f"STRATEGY: {strategy_name}")
        print(f"Description: {params['description']}")
        print("=" * 80)
        
        strategy_dir = output_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        strategy_results = []
        
        for i in range(num_samples):
            print(f"\nGenerating sample {i+1}/{num_samples}...")
            
            try:
                # Generate with current strategy
                generate(
                    model_type="lstm",
                    data_dir=data_dir,
                    strategy=params["strategy"],
                    generate_length=generate_length,
                    seq_length=seq_length,
                    temperature=params["temperature"],
                    k=params["k"],
                    p=params["p"],
                    model_path=generator_model_path,
                    discriminator_path=None,
                    discriminator_type=None,
                    seed_style="random"
                )
                
                # Find and move the generated file
                dataset_name = data_dir.split("/")[-1] if "/" in data_dir else data_dir
                generated_dir = Path(f"outputs/generators/{dataset_name}/midi")
                list_of_files = glob.glob(str(generated_dir / "generated_*.mid"))
                
                if list_of_files:
                    latest_file = max(list_of_files, key=lambda x: Path(x).stat().st_mtime)
                    output_file = strategy_dir / f"sample_{i+1}.mid"
                    shutil.move(latest_file, output_file)
                    
                    # Evaluate
                    metrics = evaluate_midi(output_file)
                    if metrics:
                        metrics["sample_id"] = i + 1
                        strategy_results.append(metrics)
                        print(f"   Note Density: {metrics['note_density']:.2f}")
                        print(f"   Pitch Range: {metrics['pitch_range']}")
                        print(f"   Polyphony: {metrics['avg_polyphony']:.2f}")
                else:
                    print("   Could not find generated file")
                    
            except Exception as e:
                print(f"   Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate summary statistics for this strategy
        if strategy_results:
            summary = calculate_summary(strategy_results)
            results["strategies"][strategy_name] = {
                "params": params,
                "results": strategy_results,
                "summary": summary
            }
            
            print(f"\n--- {strategy_name} Summary ---")
            print(f"   Mean Note Density: {summary['mean_note_density']:.2f} ± {summary['std_note_density']:.2f}")
            print(f"   Mean Pitch Range: {summary['mean_pitch_range']:.2f} ± {summary['std_pitch_range']:.2f}")
            print(f"   Mean Polyphony: {summary['mean_avg_polyphony']:.2f} ± {summary['std_avg_polyphony']:.2f}")
            print(f"   Mean Scale Consistency: {summary['mean_scale_consistency']:.4f}")
            print(f"   Mean Pitch Entropy: {summary['mean_pitch_entropy']:.4f}")
    
    # Save results
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {results_file}")
    
    # Print comparison table
    print_comparison_table(results)
    
    return results


def calculate_summary(results):
    """Calculate summary statistics from results list."""
    metric_keys = [
        "note_density", "pitch_range", "avg_polyphony",
        "scale_consistency", "pitch_entropy", "pitch_class_entropy",
        "num_notes", "duration"
    ]
    
    summary = {}
    for key in metric_keys:
        values = [d[key] for d in results if key in d and d[key] is not None]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"std_{key}"] = float(np.std(values))
            summary[f"min_{key}"] = float(np.min(values))
            summary[f"max_{key}"] = float(np.max(values))
    
    return summary


def print_comparison_table(results):
    """Print a formatted comparison table of all strategies."""
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    
    metrics = [
        ("Note Density", "mean_note_density", "std_note_density"),
        ("Pitch Range", "mean_pitch_range", "std_pitch_range"),
        ("Polyphony", "mean_avg_polyphony", "std_avg_polyphony"),
        ("Scale Consistency", "mean_scale_consistency", "std_scale_consistency"),
        ("Pitch Entropy", "mean_pitch_entropy", "std_pitch_entropy"),
    ]
    
    # Header
    strategies = list(results["strategies"].keys())
    header = f"{'Metric':<20}"
    for s in strategies:
        header += f" | {s:<15}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for metric_name, mean_key, std_key in metrics:
        row = f"{metric_name:<20}"
        for s in strategies:
            summary = results["strategies"][s]["summary"]
            mean = summary.get(mean_key, 0)
            std = summary.get(std_key, 0)
            row += f" | {mean:>6.2f}±{std:<5.2f}"
        print(row)
    
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 2: Sampling Strategy Comparison"
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        required=True,
        help="Path to trained generator model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/experiment2/results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of samples per strategy (default: 30)"
    )
    parser.add_argument(
        "--generate_length",
        type=int,
        default=200,
        help="Length of generated sequences in tokens (default: 200)"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=50,
        help="Seed sequence length (default: 50)"
    )
    
    args = parser.parse_args()
    
    run_sampling_experiment(
        generator_model_path=args.generator_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        generate_length=args.generate_length,
        seq_length=args.seq_length
    )
