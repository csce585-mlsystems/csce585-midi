"""
Experiment 1: Generator Alone vs Generator + Discriminator Conditioning
========================================================================

This experiment compares:
1. Gendef run_experiment(
    generator_model_path,
    discriminator_model_path,
    data_dir,
    output_dir,
    num_samples=50,
    generate_length=200,
    guidance_stren            # Evaluate all metrics (harmonic + standard)
            metrics = calculate_all_metrics(output_file, context_measure    # Calculate improvement for all metrics
    if results["baseline_results"] and results["guided_results"]:
        improvements = {}
        
        # Harmonic metrics
        for metric in ["consistency", "stability", "smoothness", "chord_fit"]:
            baseline_val = results["baseline_summary"][f"mean_{metric}"]
            guided_val = results["guided_summary"][f"mean_{metric}"]
            improvement = guided_val - baseline_val
            improvement_pct = (improvement / baseline_val) * 100 if baseline_val != 0 else 0
            improvements[metric] = {
                "absolute": float(improvement),
                "percentage": float(improvement_pct)
            }
        
        # Standard metrics
        for metric in ["note_density", "pitch_range", "polyphony", "num_notes", "duration"]:
            baseline_val = results["baseline_summary"][f"mean_{metric}"]
            guided_val = results["guided_summary"][f"mean_{metric}"]
            improvement = guided_val - baseline_val
            improvement_pct = (improvement / baseline_val) * 100 if baseline_val != 0 else 0
            improvements[metric] = {
                "absolute": float(improvement),
                "percentage": float(improvement_pct)
            }
        
        results["improvements"] = improvements
        
        print("\n" + "=" * 80)
        print("IMPROVEMENT ANALYSIS")
        print("=" * 80)
        
        print("\nHarmonic Metrics:")
        for metric in ["consistency", "stability", "smoothness", "chord_fit"]:
            imp = improvements[metric]
            symbol = "" if imp["absolute"] > 0 else ""
            print(f"  {symbol} {metric.replace('_', ' ').title()}: {imp['absolute']:+.4f} ({imp['percentage']:+.2f}%)")
        
        print("\nStandard Metrics:")
        for metric in ["note_density", "pitch_range", "polyphony", "num_notes", "duration"]:
            imp = improvements[metric]
            symbol = "↑" if imp["absolute"] > 0 else "↓"
            print(f"  {symbol} {metric.replace('_', ' ').title()}: {imp['absolute']:+.3f} ({imp['percentage']:+.2f}%)")
        
        print("=" * 80)           if metrics:
                metrics["sample_id"] = i + 1
                metrics["output_file"] = str(output_file)
                results["guided_results"].append(metrics)
                
                print(f"   Overall consistency: {metrics['overall_consistency']:.4f}")
                print(f"    - Note density: {metrics['note_density']:.2f} notes/sec")
                print(f"    - Pitch range: {metrics['pitch_range']}")
                print(f"    - Polyphony: {metrics['avg_polyphony']:.2f}")
            else:
                print(f"   Failed to evaluate sample {i+1}")ontext_measures=4
):ne (baseline)
2. Generator + Discriminator conditioning (guided generation)

Measure: Harmonic consistency across measures

The harmonic consistency metric evaluates:
- Pitch class stability across consecutive measures
- Chord tone usage (notes that belong to predicted chords)
- Smooth transitions between measures
"""

import sys
import json
import pickle
import numpy as np
import argparse
from pathlib import Path
from collections import Counter
import torch
import pretty_midi
from datetime import datetime

# Add project root to path (go up two levels: experiment1/ -> experiments/ -> project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from generate import generate
from utils.measure_dataset import midi_to_measure_pitches
from evaluate import evaluate_midi


def calculate_all_metrics(midi_path, context_measures=4):
    """
    Calculate both harmonic consistency metrics AND standard evaluation metrics.
    
    Combines:
    - Harmonic metrics (stability, smoothness, chord fit, etc.)
    - Standard metrics (note density, pitch range, polyphony, etc.)
    
    Args:
        midi_path: Path to MIDI file
        context_measures: Number of measures to consider for context
        
    Returns:
        dict with all metrics combined
    """
    # Get standard evaluation metrics from evaluate.py
    standard_metrics = evaluate_midi(midi_path, None)
    if standard_metrics is None:
        return None
    
    # Get harmonic consistency metrics
    harmonic_metrics = calculate_harmonic_consistency(midi_path, context_measures)
    if harmonic_metrics is None:
        return None
    
    # Combine both sets of metrics
    all_metrics = {**standard_metrics, **harmonic_metrics}
    return all_metrics


def calculate_harmonic_consistency(midi_path, context_measures=4):
    """
    Calculate harmonic consistency metrics for a generated MIDI file.
    
    Metrics:
    1. Pitch class stability: How consistent pitch classes are across measures
    2. Transition smoothness: How smoothly measures transition (shared notes)
    3. Pitch class diversity: Variety of pitch classes used
    4. Chord tone concentration: How concentrated pitches are around key centers
    
    Args:
        midi_path: Path to MIDI file
        context_measures: Number of measures to consider for context
        
    Returns:
        dict with harmonic consistency metrics
    """
    try:
        # Extract measures (sets of pitches per measure)
        measures = midi_to_measure_pitches(midi_path)
        
        if not measures or len(measures) < 2:
            return None
            
        # Convert pitches to pitch classes (0-11, removing octave information)
        pitch_class_measures = []
        for measure in measures:
            pitch_classes = set([p % 12 for p in measure])
            pitch_class_measures.append(pitch_classes)
        
        # Metric 1: Pitch class stability (how consistent pitch classes are across measures)
        # Higher is more stable/consistent
        all_pitch_classes = [pc for measure in pitch_class_measures for pc in measure]
        if len(all_pitch_classes) == 0:
            return None
            
        pitch_class_counts = Counter(all_pitch_classes)
        # Calculate entropy (lower entropy = more consistent/stable)
        total_pcs = len(all_pitch_classes)
        entropy = -sum((count/total_pcs) * np.log2(count/total_pcs) 
                      for count in pitch_class_counts.values())
        # Normalize entropy (0-1 scale, where 1 is most random)
        max_entropy = np.log2(12)  # Maximum entropy with 12 pitch classes
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        stability_score = 1 - normalized_entropy  # Invert so higher = more stable
        
        # Metric 2: Transition smoothness (shared pitch classes between consecutive measures)
        # Higher is smoother (more shared notes)
        transition_scores = []
        for i in range(len(pitch_class_measures) - 1):
            curr_measure = pitch_class_measures[i]
            next_measure = pitch_class_measures[i + 1]
            
            if len(curr_measure) == 0 or len(next_measure) == 0:
                continue
                
            # Jaccard similarity: intersection / union
            intersection = len(curr_measure & next_measure)
            union = len(curr_measure | next_measure)
            similarity = intersection / union if union > 0 else 0
            transition_scores.append(similarity)
        
        avg_transition_smoothness = np.mean(transition_scores) if transition_scores else 0
        
        # Metric 3: Pitch class diversity (how many unique pitch classes used)
        unique_pitch_classes = len(pitch_class_counts)
        diversity_score = unique_pitch_classes / 12  # Normalized to 0-1
        
        # Metric 4: Chord tone concentration
        # Check if pitches cluster around major/minor triads
        # For each measure, find how well pitches fit into a single major or minor chord
        chord_fit_scores = []
        for pc_set in pitch_class_measures:
            if len(pc_set) == 0:
                continue
                
            best_fit = 0
            # Try all 12 major and minor chords
            for root in range(12):
                # Major triad: root, major 3rd, perfect 5th
                major_chord = {root, (root + 4) % 12, (root + 7) % 12}
                # Minor triad: root, minor 3rd, perfect 5th
                minor_chord = {root, (root + 3) % 12, (root + 7) % 12}
                
                # Calculate how many notes in the measure fit these chords
                major_fit = len(pc_set & major_chord) / len(pc_set)
                minor_fit = len(pc_set & minor_chord) / len(pc_set)
                
                best_fit = max(best_fit, major_fit, minor_fit)
            
            chord_fit_scores.append(best_fit)
        
        avg_chord_fit = np.mean(chord_fit_scores) if chord_fit_scores else 0
        
        # Overall harmonic consistency score (weighted average)
        # Higher weight on stability and smoothness for harmonic consistency
        overall_score = (
            0.3 * stability_score + 
            0.4 * avg_transition_smoothness + 
            0.1 * diversity_score +
            0.2 * avg_chord_fit
        )
        
        return {
            "stability_score": float(stability_score),
            "transition_smoothness": float(avg_transition_smoothness),
            "diversity_score": float(diversity_score),
            "chord_fit_score": float(avg_chord_fit),
            "overall_consistency": float(overall_score),
            "num_measures": len(measures),
            "unique_pitch_classes": unique_pitch_classes
        }
        
    except Exception as e:
        print(f"Error calculating harmonic consistency for {midi_path}: {e}")
        return None


def run_experiment(
    generator_model_path,
    discriminator_model_path,
    data_dir,
    output_dir,
    num_samples=300,
    generate_length=200,
    guidance_strength=0.5,
    context_measures=4
):
    """
    Run the experiment comparing generator alone vs generator + discriminator.
    
    Args:
        generator_model_path: Path to trained generator model
        discriminator_model_path: Path to trained discriminator model
        data_dir: Path to preprocessed data directory
        output_dir: Directory to save results
        num_samples: Number of MIDI samples to generate per condition
        generate_length: Number of notes to generate per sample
        guidance_strength: Strength of discriminator guidance
        context_measures: Number of measures for discriminator context
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each condition
    baseline_dir = output_dir / "baseline_generator_alone"
    guided_dir = output_dir / "guided_generator_discriminator"
    baseline_dir.mkdir(exist_ok=True)
    guided_dir.mkdir(exist_ok=True)
    
    results = {
        "experiment": "harmonic_consistency_comparison",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator_model": generator_model_path,
        "discriminator_model": discriminator_model_path,
        "data_dir": data_dir,
        "num_samples": num_samples,
        "generate_length": generate_length,
        "guidance_strength": guidance_strength,
        "context_measures": context_measures,
        "baseline_results": [],
        "guided_results": []
    }
    
    print("=" * 80)
    print("EXPERIMENT 1: Harmonic Consistency - Generator Alone vs Generator + Discriminator")
    print("=" * 80)
    print(f"Generator model: {generator_model_path}")
    print(f"Discriminator model: {discriminator_model_path}")
    print(f"Generating {num_samples} samples per condition...")
    print()
    
    # Condition 1: Generator alone (baseline)
    print("-" * 80)
    print("CONDITION 1: Generator Alone (Baseline)")
    print("-" * 80)
    
    for i in range(num_samples):
        print(f"\nGenerating baseline sample {i+1}/{num_samples}...")
        
        try:
            # Generate without discriminator
            # Note: generate() creates its own output file path
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
                discriminator_path=None,  # No discriminator
                discriminator_type=None,
                guidance_strength=guidance_strength,
                context_measures=context_measures,
                seed_style="random"
            )
            
            # Find the most recently generated file and move it
            import glob
            import shutil
            dataset_name = data_dir.split("/")[-1] if "/" in data_dir else data_dir
            generated_dir = Path(f"outputs/generators/{dataset_name}/midi")
            
            # Get most recent generated file
            list_of_files = glob.glob(str(generated_dir / "generated_*.mid"))
            if list_of_files:
                latest_file = max(list_of_files, key=lambda x: Path(x).stat().st_mtime)
                output_file = baseline_dir / f"sample_{i+1}.mid"
                shutil.move(latest_file, output_file)
            else:
                print(f"   Could not find generated file")
                continue
            
            # Evaluate all metrics (harmonic + standard)
            metrics = calculate_all_metrics(output_file, context_measures)
            
            if metrics:
                metrics["sample_id"] = i + 1
                metrics["output_file"] = str(output_file)
                results["baseline_results"].append(metrics)
                
                print(f"   Overall consistency: {metrics['overall_consistency']:.4f}")
                print(f"    - Note density: {metrics['note_density']:.2f} notes/sec")
                print(f"    - Pitch range: {metrics['pitch_range']}")
                print(f"    - Polyphony: {metrics['avg_polyphony']:.2f}")
            else:
                print(f"   Failed to evaluate sample {i+1}")
                
        except Exception as e:
            print(f"   Error generating baseline sample {i+1}: {e}")
    
    # Condition 2: Generator + Discriminator
    print("\n" + "-" * 80)
    print("CONDITION 2: Generator + Discriminator (Guided)")
    print("-" * 80)
    
    for i in range(num_samples):
        print(f"\nGenerating guided sample {i+1}/{num_samples}...")
        
        try:
            # Generate with discriminator guidance
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
                discriminator_path=discriminator_model_path,  # With discriminator
                discriminator_type="lstm",
                guidance_strength=guidance_strength,
                context_measures=context_measures,
                seed_style="random"
            )
            
            # Find the most recently generated file and move it
            import glob
            import shutil
            dataset_name = data_dir.split("/")[-1] if "/" in data_dir else data_dir
            generated_dir = Path(f"outputs/generators/{dataset_name}/midi")
            
            # Get most recent generated file
            list_of_files = glob.glob(str(generated_dir / "generated_*.mid"))
            if list_of_files:
                latest_file = max(list_of_files, key=lambda x: Path(x).stat().st_mtime)
                output_file = guided_dir / f"sample_{i+1}.mid"
                shutil.move(latest_file, output_file)
            else:
                print(f"   Could not find generated file")
                continue
            
            # Evaluate all metrics (harmonic + standard)
            metrics = calculate_all_metrics(output_file, context_measures)
            
            if metrics:
                metrics["sample_id"] = i + 1
                metrics["output_file"] = str(output_file)
                results["guided_results"].append(metrics)
                
                print(f"   Overall consistency: {metrics['overall_consistency']:.4f}")
                print(f"    - Note density: {metrics['note_density']:.2f} notes/sec")
                print(f"    - Pitch range: {metrics['pitch_range']}")
                print(f"    - Polyphony: {metrics['avg_polyphony']:.2f}")
            else:
                print(f"   Failed to evaluate sample {i+1}")
                
        except Exception as e:
            print(f"Error generating guided sample {i+1}: {e}")
    
    # Calculate aggregate statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    if results["baseline_results"]:
        baseline_consistency = [r["overall_consistency"] for r in results["baseline_results"]]
        baseline_stability = [r["stability_score"] for r in results["baseline_results"]]
        baseline_smoothness = [r["transition_smoothness"] for r in results["baseline_results"]]
        baseline_chord_fit = [r["chord_fit_score"] for r in results["baseline_results"]]
        baseline_note_density = [r["note_density"] for r in results["baseline_results"]]
        baseline_pitch_range = [r["pitch_range"] for r in results["baseline_results"]]
        baseline_polyphony = [r["avg_polyphony"] for r in results["baseline_results"]]
        baseline_num_notes = [r["num_notes"] for r in results["baseline_results"]]
        baseline_duration = [r["duration"] for r in results["baseline_results"]]
        
        results["baseline_summary"] = {
            # Harmonic metrics
            "mean_consistency": float(np.mean(baseline_consistency)),
            "std_consistency": float(np.std(baseline_consistency)),
            "mean_stability": float(np.mean(baseline_stability)),
            "std_stability": float(np.std(baseline_stability)),
            "mean_smoothness": float(np.mean(baseline_smoothness)),
            "std_smoothness": float(np.std(baseline_smoothness)),
            "mean_chord_fit": float(np.mean(baseline_chord_fit)),
            "std_chord_fit": float(np.std(baseline_chord_fit)),
            # Standard metrics
            "mean_note_density": float(np.mean(baseline_note_density)),
            "std_note_density": float(np.std(baseline_note_density)),
            "mean_pitch_range": float(np.mean(baseline_pitch_range)),
            "std_pitch_range": float(np.std(baseline_pitch_range)),
            "mean_polyphony": float(np.mean(baseline_polyphony)),
            "std_polyphony": float(np.std(baseline_polyphony)),
            "mean_num_notes": float(np.mean(baseline_num_notes)),
            "std_num_notes": float(np.std(baseline_num_notes)),
            "mean_duration": float(np.mean(baseline_duration)),
            "std_duration": float(np.std(baseline_duration))
        }
        
        print("\nBaseline (Generator Alone):")
        print(f"  Harmonic Metrics:")
        print(f"    Overall Consistency: {results['baseline_summary']['mean_consistency']:.4f} ± {results['baseline_summary']['std_consistency']:.4f}")
        print(f"    Stability:           {results['baseline_summary']['mean_stability']:.4f} ± {results['baseline_summary']['std_stability']:.4f}")
        print(f"    Smoothness:          {results['baseline_summary']['mean_smoothness']:.4f} ± {results['baseline_summary']['std_smoothness']:.4f}")
        print(f"    Chord Fit:           {results['baseline_summary']['mean_chord_fit']:.4f} ± {results['baseline_summary']['std_chord_fit']:.4f}")
        print(f"  Standard Metrics:")
        print(f"    Note Density:        {results['baseline_summary']['mean_note_density']:.2f} ± {results['baseline_summary']['std_note_density']:.2f} notes/sec")
        print(f"    Pitch Range:         {results['baseline_summary']['mean_pitch_range']:.1f} ± {results['baseline_summary']['std_pitch_range']:.1f} semitones")
        print(f"    Polyphony:           {results['baseline_summary']['mean_polyphony']:.2f} ± {results['baseline_summary']['std_polyphony']:.2f}")
        print(f"    Num Notes:           {results['baseline_summary']['mean_num_notes']:.1f} ± {results['baseline_summary']['std_num_notes']:.1f}")
        print(f"    Duration:            {results['baseline_summary']['mean_duration']:.2f} ± {results['baseline_summary']['std_duration']:.2f} sec")
    
    if results["guided_results"]:
        guided_consistency = [r["overall_consistency"] for r in results["guided_results"]]
        guided_stability = [r["stability_score"] for r in results["guided_results"]]
        guided_smoothness = [r["transition_smoothness"] for r in results["guided_results"]]
        guided_chord_fit = [r["chord_fit_score"] for r in results["guided_results"]]
        guided_note_density = [r["note_density"] for r in results["guided_results"]]
        guided_pitch_range = [r["pitch_range"] for r in results["guided_results"]]
        guided_polyphony = [r["avg_polyphony"] for r in results["guided_results"]]
        guided_num_notes = [r["num_notes"] for r in results["guided_results"]]
        guided_duration = [r["duration"] for r in results["guided_results"]]
        
        results["guided_summary"] = {
            # Harmonic metrics
            "mean_consistency": float(np.mean(guided_consistency)),
            "std_consistency": float(np.std(guided_consistency)),
            "mean_stability": float(np.mean(guided_stability)),
            "std_stability": float(np.std(guided_stability)),
            "mean_smoothness": float(np.mean(guided_smoothness)),
            "std_smoothness": float(np.std(guided_smoothness)),
            "mean_chord_fit": float(np.mean(guided_chord_fit)),
            "std_chord_fit": float(np.std(guided_chord_fit)),
            # Standard metrics
            "mean_note_density": float(np.mean(guided_note_density)),
            "std_note_density": float(np.std(guided_note_density)),
            "mean_pitch_range": float(np.mean(guided_pitch_range)),
            "std_pitch_range": float(np.std(guided_pitch_range)),
            "mean_polyphony": float(np.mean(guided_polyphony)),
            "std_polyphony": float(np.std(guided_polyphony)),
            "mean_num_notes": float(np.mean(guided_num_notes)),
            "std_num_notes": float(np.std(guided_num_notes)),
            "mean_duration": float(np.mean(guided_duration)),
            "std_duration": float(np.std(guided_duration))
        }
        
        print("\nGuided (Generator + Discriminator):")
        print(f"  Harmonic Metrics:")
        print(f"    Overall Consistency: {results['guided_summary']['mean_consistency']:.4f} ± {results['guided_summary']['std_consistency']:.4f}")
        print(f"    Stability:           {results['guided_summary']['mean_stability']:.4f} ± {results['guided_summary']['std_stability']:.4f}")
        print(f"    Smoothness:          {results['guided_summary']['mean_smoothness']:.4f} ± {results['guided_summary']['std_smoothness']:.4f}")
        print(f"    Chord Fit:           {results['guided_summary']['mean_chord_fit']:.4f} ± {results['guided_summary']['std_chord_fit']:.4f}")
        print(f"  Standard Metrics:")
        print(f"    Note Density:        {results['guided_summary']['mean_note_density']:.2f} ± {results['guided_summary']['std_note_density']:.2f} notes/sec")
        print(f"    Pitch Range:         {results['guided_summary']['mean_pitch_range']:.1f} ± {results['guided_summary']['std_pitch_range']:.1f} semitones")
        print(f"    Polyphony:           {results['guided_summary']['mean_polyphony']:.2f} ± {results['guided_summary']['std_polyphony']:.2f}")
        print(f"    Num Notes:           {results['guided_summary']['mean_num_notes']:.1f} ± {results['guided_summary']['std_num_notes']:.1f}")
        print(f"    Duration:            {results['guided_summary']['mean_duration']:.2f} ± {results['guided_summary']['std_duration']:.2f} sec")
    
    # Calculate improvement
    if results["baseline_results"] and results["guided_results"]:
        improvement = results["guided_summary"]["mean_consistency"] - results["baseline_summary"]["mean_consistency"]
        improvement_pct = (improvement / results["baseline_summary"]["mean_consistency"]) * 100
        
        results["improvement"] = {
            "absolute": float(improvement),
            "percentage": float(improvement_pct)
        }
        
        print("\n" + "-" * 80)
        print("IMPROVEMENT:")
        if improvement > 0:
            print(f"Discriminator guidance improved consistency by {improvement:.4f} ({improvement_pct:.2f}%)")
        else:
            print(f"Discriminator guidance decreased consistency by {abs(improvement):.4f} ({abs(improvement_pct):.2f}%)")
        print("-" * 80)
    
    # Save results
    results_file = output_dir / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1: Compare generator alone vs generator + discriminator conditioning"
    )
    
    parser.add_argument(
        "--generator_model",
        type=str,
        required=True,
        help="Path to trained generator model checkpoint"
    )
    
    parser.add_argument(
        "--discriminator_model",
        type=str,
        required=True,
        help="Path to trained discriminator model checkpoint"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to preprocessed data directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/results/harmonic_consistency",
        help="Directory to save experiment results"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to generate per condition (default: 50 for robust statistics)"
    )
    
    parser.add_argument(
        "--generate_length",
        type=int,
        default=200,
        help="Number of notes to generate per sample"
    )
    
    parser.add_argument(
        "--guidance_strength",
        type=float,
        default=0.5,
        help="Strength of discriminator guidance (0.0-1.0)"
    )
    
    parser.add_argument(
        "--context_measures",
        type=int,
        default=4,
        help="Number of measures for discriminator context"
    )
    
    args = parser.parse_args()
    
    run_experiment(
        generator_model_path=args.generator_model,
        discriminator_model_path=args.discriminator_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        generate_length=args.generate_length,
        guidance_strength=args.guidance_strength,
        context_measures=args.context_measures
    )
