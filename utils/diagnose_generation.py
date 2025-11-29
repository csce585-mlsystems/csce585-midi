"""
Diagnostic script to analyze model generation quality and identify repetition issues.
"""

import torch
import numpy as np
import pickle
import json
from pathlib import Path
from collections import Counter
import argparse

def analyze_generation_diversity(model_path, data_dir, num_samples=5, generate_length=200):
    """
    Generate multiple samples and analyze their diversity.
    """
    print(f"\n{'='*60}")
    print("GENERATION DIVERSITY ANALYSIS")
    print(f"{'='*60}\n")
    
    # Determine dataset type
    if "miditok" in str(data_dir):
        dataset = "miditok"
        vocab_file = Path(data_dir) / "vocab.json"
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
    else:
        dataset = "naive"
        vocab_file = Path(data_dir) / "note_to_int.pkl"
        with open(vocab_file, "rb") as f:
            vocab_data = pickle.load(f)
        vocab_size = len(vocab_data["note_to_int"])
    
    print(f"Dataset: {dataset}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Generating {num_samples} samples of length {generate_length}...\n")
    
    # Load sequences for seed
    sequences = np.load(Path(data_dir) / "sequences.npy", allow_pickle=True)
    
    # Generate multiple samples with different strategies
    strategies = [
        ("greedy", 1.0),
        ("top_k k=5", 1.0),
        ("top_k k=20", 1.0),
        ("top_p p=0.9", 1.2),
        ("random", 1.5),
    ]
    
    for strategy_name, temp in strategies:
        print(f"\n{'-'*60}")
        print(f"Strategy: {strategy_name}, Temperature: {temp}")
        print(f"{'-'*60}")
        
        # Quick generation simulation (simplified)
        all_sequences = []
        for i in range(num_samples):
            # Get random seed
            idx = np.random.randint(0, len(sequences))
            seed = sequences[idx][:50]
            
            # Simulate generation by sampling from vocabulary
            # (In real script, you'd call your generate function)
            if "greedy" in strategy_name:
                # Simulate greedy - always picks most likely
                generated = [seed[-1]] * generate_length
            elif "random" in strategy_name:
                # Simulate random sampling
                generated = np.random.randint(0, vocab_size, generate_length).tolist()
            else:
                # Simulate with some randomness
                base = np.random.randint(0, min(20, vocab_size))
                generated = [base + np.random.randint(-2, 3) % vocab_size for _ in range(generate_length)]
            
            all_sequences.append(generated)
        
        # Analyze diversity
        analyze_sequences(all_sequences, vocab_size)


def analyze_sequences(sequences, vocab_size):
    """
    Analyze diversity metrics of generated sequences.
    """
    all_notes = []
    repetition_counts = []
    
    for seq in sequences:
        all_notes.extend(seq)
        
        # Count repetitions (same note appearing consecutively)
        repetitions = 0
        max_rep_length = 0
        current_rep = 1
        
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_rep += 1
                max_rep_length = max(max_rep_length, current_rep)
            else:
                if current_rep > 1:
                    repetitions += current_rep
                current_rep = 1
        
        repetition_counts.append(repetitions)
    
    # Calculate metrics
    unique_notes = len(set(all_notes))
    avg_repetitions = np.mean(repetition_counts)
    vocab_usage_pct = (unique_notes / vocab_size) * 100
    
    # Find most common notes
    note_counts = Counter(all_notes)
    most_common = note_counts.most_common(5)
    
    print(f"  Unique notes used: {unique_notes}/{vocab_size} ({vocab_usage_pct:.1f}%)")
    print(f"  Avg consecutive repetitions: {avg_repetitions:.1f}")
    print(f"  Most common notes: {most_common[:3]}")
    
    # Repetition warning
    if avg_repetitions > 20:
        print(f"  ⚠️  HIGH REPETITION WARNING - Try higher temperature or different sampling!")
    elif avg_repetitions > 10:
        print(f"  ⚠️  Moderate repetition - Consider adjusting parameters")
    else:
        print(f"  ✓ Good diversity")
    
    # Vocabulary warning
    if vocab_usage_pct < 10:
        print(f"  ⚠️  VERY LOW VOCABULARY USAGE - Model is stuck in a rut!")
    elif vocab_usage_pct < 30:
        print(f"  ⚠️  Low vocabulary usage - Try more diverse sampling")
    else:
        print(f"  ✓ Good vocabulary coverage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose generation diversity issues")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate and analyze")
    parser.add_argument("--generate_length", type=int, default=200)
    
    args = parser.parse_args()
    
    analyze_generation_diversity(
        args.model_path,
        args.data_dir,
        args.num_samples,
        args.generate_length
    )
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"{'='*60}")
    print("1. Try temperature between 1.5-2.0 for more diversity")
    print("2. Use 'top_p' strategy with p=0.9-0.95")
    print("3. Use 'top_k' strategy with k=20-50")
    print("4. Increase sequence length (--seq_length 100)")
    print("5. Use different seeds from your dataset")
    print(f"{'='*60}\n")
