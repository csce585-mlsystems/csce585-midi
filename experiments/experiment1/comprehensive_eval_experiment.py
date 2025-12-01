"""
Experiment 1: Comprehensive Evaluation (Generator vs Generator + Discriminator)
===============================================================================

This experiment compares:
1. Generator alone (baseline)
2. Generator + Discriminator conditioning (guided generation)

Metrics evaluated (from evaluate.py):
- Note Density
- Pitch Range
- Average Polyphony
- Scale Consistency
- Pitch Entropy
- Pitch Class Entropy
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

def run_experiment(
    generator_model_path,
    discriminator_model_path,
    data_dir,
    output_dir,
    num_samples=50,
    generate_length=200,
    guidance_strength=0.5,
    context_measures=4
):
    # Setup directories
    output_dir = Path(output_dir)
    baseline_dir = output_dir / "baseline_generator_alone"
    guided_dir = output_dir / "guided_generator_discriminator"
    
    baseline_dir.mkdir(parents=True, exist_ok=True)
    guided_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "experiment": "comprehensive_evaluation",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator_model": generator_model_path,
        "discriminator_model": discriminator_model_path,
        "data_dir": data_dir,
        "num_samples": num_samples,
        "generate_length": generate_length,
        "guidance_strength": guidance_strength,
        "baseline_results": [],
        "guided_results": []
    }
    
    # Condition 1: Generator alone (baseline)
    print("-" * 80)
    print("CONDITION 1: Generator Alone (Baseline)")
    print("-" * 80)
    
    for i in range(num_samples):
        print(f"\nGenerating baseline sample {i+1}/{num_samples}...")
        
        try:
            # Generate without discriminator
            generate(
                model_type="lstm",
                data_dir=data_dir,
                strategy="top_p",
                generate_length=generate_length,
                seq_length=50,
                temperature=1.0,
                k=5,
                p=0.9,
                model_path=generator_model_path,
                discriminator_path=None,
                discriminator_type=None,
                guidance_strength=guidance_strength,
                context_measures=context_measures,
                seed_style="random"
            )
            
            # Find and move file
            dataset_name = data_dir.split("/")[-1] if "/" in data_dir else data_dir
            generated_dir = Path(f"outputs/generators/{dataset_name}/midi")
            list_of_files = glob.glob(str(generated_dir / "generated_*.mid"))
            
            if list_of_files:
                latest_file = max(list_of_files, key=lambda x: Path(x).stat().st_mtime)
                output_file = baseline_dir / f"sample_{i+1}.mid"
                shutil.move(latest_file, output_file)
                
                # Evaluate
                metrics = evaluate_midi(output_file)
                if metrics:
                    metrics["sample_id"] = i + 1
                    results["baseline_results"].append(metrics)
                    print(f"   Scale Consistency: {metrics['scale_consistency']:.4f}")
                    print(f"   Pitch Entropy: {metrics['pitch_entropy']:.4f}")
            else:
                print("   Could not find generated file")
                
        except Exception as e:
            print(f"   Error: {e}")

    # Condition 2: Generator + Discriminator
    print("\n" + "-" * 80)
    print("CONDITION 2: Generator + Discriminator (Guided)")
    print("-" * 80)
    
    for i in range(num_samples):
        print(f"\nGenerating guided sample {i+1}/{num_samples}...")
        
        try:
            # Generate with discriminator
            generate(
                model_type="lstm",
                data_dir=data_dir,
                strategy="top_p",
                generate_length=generate_length,
                seq_length=50,
                temperature=1.0,
                k=5,
                p=0.9,
                model_path=generator_model_path,
                discriminator_path=discriminator_model_path,
                discriminator_type="lstm",
                guidance_strength=guidance_strength,
                context_measures=context_measures,
                seed_style="random"
            )
            
            # Find and move file
            dataset_name = data_dir.split("/")[-1] if "/" in data_dir else data_dir
            generated_dir = Path(f"outputs/generators/{dataset_name}/midi")
            list_of_files = glob.glob(str(generated_dir / "generated_*.mid"))
            
            if list_of_files:
                latest_file = max(list_of_files, key=lambda x: Path(x).stat().st_mtime)
                output_file = guided_dir / f"sample_{i+1}.mid"
                shutil.move(latest_file, output_file)
                
                # Evaluate
                metrics = evaluate_midi(output_file)
                if metrics:
                    metrics["sample_id"] = i + 1
                    results["guided_results"].append(metrics)
                    print(f"   Scale Consistency: {metrics['scale_consistency']:.4f}")
                    print(f"   Pitch Entropy: {metrics['pitch_entropy']:.4f}")
            else:
                print("   Could not find generated file")
                
        except Exception as e:
            print(f"   Error: {e}")

    # Calculate summaries
    metric_keys = [
        "note_density", "pitch_range", "avg_polyphony", 
        "scale_consistency", "pitch_entropy", "pitch_class_entropy"
    ]
    
    for condition in ["baseline", "guided"]:
        summary = {}
        data = results[f"{condition}_results"]
        if not data:
            continue
            
        for key in metric_keys:
            values = [d[key] for d in data if key in d]
            if values:
                summary[f"mean_{key}"] = float(np.mean(values))
                summary[f"std_{key}"] = float(np.std(values))
        
        results[f"{condition}_summary"] = summary

    # Save results
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {results_file}")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Baseline':<20} | {'Guided':<20} | {'Change':<10}")
    print("-" * 80)
    
    for key in metric_keys:
        base_mean = results["baseline_summary"].get(f"mean_{key}", 0)
        base_std = results["baseline_summary"].get(f"std_{key}", 0)
        guide_mean = results["guided_summary"].get(f"mean_{key}", 0)
        guide_std = results["guided_summary"].get(f"std_{key}", 0)
        
        change = ((guide_mean - base_mean) / base_mean * 100) if base_mean != 0 else 0
        
        print(f"{key.replace('_', ' ').title():<25} | {base_mean:.3f} ± {base_std:.3f}   | {guide_mean:.3f} ± {guide_std:.3f}   | {change:+.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_model", required=True)
    parser.add_argument("--discriminator_model", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="experiments/results/comprehensive")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--generate_length", type=int, default=200)
    parser.add_argument("--guidance_strength", type=float, default=0.5)
    parser.add_argument("--context_measures", type=int, default=4)
    
    args = parser.parse_args()
    
    run_experiment(
        args.generator_model,
        args.discriminator_model,
        args.data_dir,
        args.output_dir,
        args.num_samples,
        args.generate_length,
        args.guidance_strength,
        args.context_measures
    )
