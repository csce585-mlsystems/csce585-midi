"""
Experiment 5: Discriminator Architecture Comparison for Guided Generation
==========================================================================

This experiment compares how different discriminator architectures affect
generation quality when used for guided generation with a fixed baseline 
generator (LSTM trained on naive tokenization).

Discriminator Architectures Compared:
1. No Discriminator (Baseline) - Generator alone for reference
2. MLP Discriminator - Simple feedforward network (fast, fewer temporal patterns)
3. LSTM Discriminator - Recurrent network (captures sequential patterns)
4. Transformer Discriminator - Attention-based (captures long-range dependencies)

Hypothesis: Different discriminator architectures will provide different 
quality of harmonic guidance during generation:
- MLP: Fast but may miss temporal context, providing basic chord tone guidance
- LSTM: Better at sequential patterns, should provide smoother transitions
- Transformer: May capture complex harmonic relationships, but could overfit

All discriminators are trained on pitch/chord label prediction, providing
guidance about which notes fit harmonically in the current musical context.

Metrics evaluated:
- Scale Consistency (how well notes fit musical scales)
- Pitch Entropy (diversity of pitches)
- Pitch Class Entropy (distribution across 12 pitch classes)
- Note Density, Pitch Range, Average Polyphony
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


# Fixed generator (baseline naive LSTM)
GENERATOR_CONFIG = {
    "name": "LSTM Generator (Baseline)",
    "model_type": "lstm",
    "model_path": "models/generators/checkpoints/nottingham-dataset-final_experiments_naive/lstm_20251129_200244.pth",
    "description": "Two-layer LSTM generator trained on naive tokenization (Nottingham dataset)",
}

# Common data directory for generation (naive tokenization)
DATA_DIR = "data/nottingham-dataset-final_experiments_naive"

# Discriminator configurations - using best checkpoints for each architecture
# Selected based on training epochs and label type (pitches for better tonal guidance)
DISCRIMINATORS = {
    "none": {
        "name": "No Discriminator (Baseline)",
        "model_type": None,
        "model_path": None,
        "description": "Generator alone without discriminator guidance (control condition)",
    },
    "mlp": {
        "name": "MLP Discriminator",
        "model_type": "mlp",
        # Using later epoch with pitch labels for better harmonic guidance
        "model_path": "models/discriminators/checkpoints/mlp/mlp_labelpitches_ctx4_ep15.pt",
        "description": "Feedforward MLP with pitch label prediction (15 epochs)",
        "context_measures": 4,
    },
    "lstm": {
        "name": "LSTM Discriminator",
        "model_type": "lstm",
        # Using well-trained LSTM with pitch labels
        "model_path": "models/discriminators/checkpoints/lstm/lstm_labelpitches_ctx4_ep10.pt",
        "description": "Recurrent LSTM with pitch label prediction (10 epochs)",
        "context_measures": 4,
    },
    "transformer": {
        "name": "Transformer Discriminator",
        "model_type": "transformer",
        # Using transformer with chord labels (best available)
        "model_path": "models/discriminators/checkpoints/transformer/transformer_labelchords_ctx4_ep6.pt",
        "description": "Attention-based transformer with chord label prediction (6 epochs)",
        "context_measures": 4,
    },
}

# Generation settings to test - using balanced setting for fair comparison
GENERATION_SETTINGS = [
    {"name": "conservative", "strategy": "top_p", "p": 0.85, "temperature": 0.8},
    {"name": "balanced", "strategy": "top_p", "p": 0.9, "temperature": 1.0},
    {"name": "creative", "strategy": "top_p", "p": 0.95, "temperature": 1.2},
]

# Guidance strengths to test
GUIDANCE_STRENGTHS = [0.3, 0.5, 0.7]

DISCRIMINATOR_ORDER = ['none', 'mlp', 'lstm', 'transformer']
SETTING_ORDER = ['conservative', 'balanced', 'creative']


def run_experiment(
    output_dir="experiments/experiment5/results",
    num_samples_per_setting=10,
    generate_length=200,
    seq_length=50,
    guidance_strength=0.5,
    test_guidance_strengths=False,
):
    """
    Run the discriminator architecture comparison experiment.
    
    Args:
        output_dir: Directory to store results and generated MIDI files
        num_samples_per_setting: Number of samples to generate per discriminator/setting combination
        generate_length: Number of tokens to generate per sample
        seq_length: Sequence length for the model (should match training)
        guidance_strength: Strength of discriminator guidance (0-1)
        test_guidance_strengths: Whether to also vary guidance strength
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = {
        "experiment": "discriminator_architecture_comparison",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_samples_per_setting": num_samples_per_setting,
            "generate_length": generate_length,
            "seq_length": seq_length,
            "guidance_strength": guidance_strength,
            "generation_settings": GENERATION_SETTINGS,
            "data_dir": DATA_DIR,
        },
        "generator": GENERATOR_CONFIG,
        "discriminators": DISCRIMINATORS,
        "results": {}
    }
    
    # CSV for detailed results
    csv_file = output_dir / "detailed_results.csv"
    csv_fields = [
        "discriminator_type", "setting", "guidance_strength", "sample_id",
        "num_notes", "duration", "note_density", "pitch_range",
        "avg_polyphony", "scale_consistency", "pitch_entropy", "pitch_class_entropy"
    ]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
    
    # Verify generator exists
    generator_path = PROJECT_ROOT / GENERATOR_CONFIG["model_path"]
    if not generator_path.exists():
        print(f"ERROR: Generator not found at {generator_path}")
        return None
    
    # Run experiments for each discriminator
    for disc_key in DISCRIMINATOR_ORDER:
        disc_config = DISCRIMINATORS[disc_key]
        print("\n" + "=" * 80)
        print(f"TESTING DISCRIMINATOR: {disc_config['name']}")
        print(f"Description: {disc_config['description']}")
        if disc_config['model_path']:
            print(f"Model path: {disc_config['model_path']}")
        print("=" * 80)
        
        disc_results = {}
        disc_output_dir = output_dir / disc_key
        disc_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify discriminator exists (if not baseline)
        if disc_config['model_path']:
            disc_path = PROJECT_ROOT / disc_config['model_path']
            if not disc_path.exists():
                print(f"ERROR: Discriminator not found at {disc_path}")
                continue
        
        # Test each generation setting
        for setting in GENERATION_SETTINGS:
            print(f"\n--- Setting: {setting['name']} ---")
            print(f"    Strategy: {setting['strategy']}, Temperature: {setting['temperature']}, P: {setting.get('p', 'N/A')}")
            
            setting_dir = disc_output_dir / setting["name"]
            setting_dir.mkdir(parents=True, exist_ok=True)
            
            setting_results = []
            
            for i in range(num_samples_per_setting):
                print(f"  Generating sample {i+1}/{num_samples_per_setting}...", end=" ")
                
                try:
                    # Generate sample using the generator with discriminator guidance
                    generate(
                        model_type=GENERATOR_CONFIG["model_type"],
                        data_dir=DATA_DIR,
                        strategy=setting["strategy"],
                        generate_length=generate_length,
                        seq_length=seq_length,
                        temperature=setting["temperature"],
                        k=5,
                        p=setting.get("p", 0.9),
                        model_path=str(generator_path),
                        discriminator_path=str(PROJECT_ROOT / disc_config['model_path']) if disc_config['model_path'] else None,
                        discriminator_type=disc_config['model_type'],
                        guidance_strength=guidance_strength if disc_config['model_type'] else 0,
                        context_measures=disc_config.get('context_measures', 4),
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
                            metrics["discriminator_type"] = disc_key
                            metrics["setting"] = setting["name"]
                            metrics["guidance_strength"] = guidance_strength if disc_config['model_type'] else 0
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
                disc_results[setting["name"]] = calculate_statistics(setting_results)
        
        all_results["results"][disc_key] = disc_results
    
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
    print("EXPERIMENT SUMMARY: Discriminator Architecture Comparison")
    print("=" * 80)
    
    print(f"\nGenerator: {results['generator']['name']}")
    print(f"Generator Path: {results['generator']['model_path']}")
    print(f"Guidance Strength: {results['config']['guidance_strength']}")
    
    # Key metrics to highlight
    key_metrics = ["scale_consistency", "pitch_entropy", "pitch_class_entropy", "note_density", "pitch_range"]
    
    print("\n--- Generation Metrics by Discriminator ---")
    for disc_key in DISCRIMINATOR_ORDER:
        if disc_key in results.get("results", {}):
            disc_results = results["results"][disc_key]
            disc_name = DISCRIMINATORS.get(disc_key, {}).get("name", disc_key)
            print(f"\n{disc_name}")
            print("-" * 60)
            
            for setting_name in SETTING_ORDER:
                if setting_name in disc_results:
                    setting_stats = disc_results[setting_name]
                    print(f"\n  {setting_name.upper()}:")
                    for metric in key_metrics:
                        if metric in setting_stats:
                            stats = setting_stats[metric]
                            print(f"    {metric:25s}: {stats['mean']:8.4f} ± {stats['std']:.4f}")
    
    # Print comparison table for balanced setting
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (Balanced Setting)")
    print("=" * 80)
    
    header = f"{'Discriminator':<25} {'Scale Cons.':<12} {'Pitch Ent.':<12} {'PC Ent.':<12} {'Note Dens.':<12}"
    print(header)
    print("-" * 75)
    
    for disc_key in DISCRIMINATOR_ORDER:
        if disc_key in results.get("results", {}) and "balanced" in results["results"][disc_key]:
            stats = results["results"][disc_key]["balanced"]
            disc_name = DISCRIMINATORS.get(disc_key, {}).get("name", disc_key)[:24]
            
            sc = stats.get("scale_consistency", {}).get("mean", 0)
            pe = stats.get("pitch_entropy", {}).get("mean", 0)
            pce = stats.get("pitch_class_entropy", {}).get("mean", 0)
            nd = stats.get("note_density", {}).get("mean", 0)
            
            print(f"{disc_name:<25} {sc:<12.4f} {pe:<12.4f} {pce:<12.4f} {nd:<12.4f}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE:")
    print("-" * 80)
    print("""
    This experiment tests how different discriminator architectures affect 
    the quality of guided music generation.
    
    Scale Consistency: Higher = notes fit better in traditional scales (0-1)
                       Discriminator guidance should improve this metric.
    
    Pitch Entropy: Higher = more diverse pitch usage
                   Good discriminators maintain diversity while adding guidance.
    
    Pitch Class Entropy: Higher = more even distribution across 12 pitch classes
                         Guidance shouldn't overly constrain to few pitches.
    
    Expected patterns:
    - Baseline (no discriminator): Reference point for ungided generation
    - MLP: Fast, basic harmonic guidance, may miss temporal patterns
    - LSTM: Better sequential awareness, smoother harmonic transitions
    - Transformer: Complex pattern recognition, but may overfit on small data
    
    A good discriminator should improve scale consistency without 
    significantly reducing pitch diversity (entropy metrics).
    """)


def compare_discriminators(results_file):
    """Load saved results and generate comparison analysis."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run discriminator architecture comparison experiment")
    parser.add_argument("--output_dir", type=str, default="experiments/experiment5/results",
                        help="Directory to store results")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples per discriminator/setting combination")
    parser.add_argument("--generate_length", type=int, default=200,
                        help="Number of tokens to generate per sample")
    parser.add_argument("--seq_length", type=int, default=50,
                        help="Sequence length (should match training)")
    parser.add_argument("--guidance_strength", type=float, default=0.5,
                        help="Strength of discriminator guidance (0-1)")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Path to existing results file to analyze (skip generation)")
    
    args = parser.parse_args()
    
    if args.analyze:
        compare_discriminators(args.analyze)
    else:
        run_experiment(
            output_dir=args.output_dir,
            num_samples_per_setting=args.num_samples,
            generate_length=args.generate_length,
            seq_length=args.seq_length,
            guidance_strength=args.guidance_strength,
        )
