#!/usr/bin/env python3
"""
Compare original MIDITok vs Augmented MIDITok datasets and training results.
Shows how data augmentation helps reduce overfitting.
"""

import numpy as np
import json
from pathlib import Path
import sys

def analyze_dataset(data_dir, name):
    """Analyze a MIDITok dataset."""
    data_dir = Path(data_dir)
    
    print(f"\n{'=' * 60}")
    print(f"{name} Dataset Analysis")
    print(f"{'=' * 60}")
    
    # Load config
    config_path = data_dir / "config.json"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load sequences
    sequences_path = data_dir / "sequences.npy"
    if not sequences_path.exists():
        print(f"❌ Sequences file not found: {sequences_path}")
        return None
    
    sequences = np.load(sequences_path, allow_pickle=True)
    
    # Calculate statistics
    seq_length = config.get('seq_length', 100)
    vocab_size = config.get('vocab_size', 284)
    valid_seqs = [seq for seq in sequences if len(seq) >= seq_length]
    total_samples = sum(len(seq) - seq_length for seq in valid_seqs)
    total_tokens = sum(len(seq) for seq in sequences)
    
    print(f"\n📊 Basic Statistics:")
    print(f"  Total sequences: {len(sequences):,}")
    print(f"  Valid sequences (>= {seq_length}): {len(valid_seqs):,}")
    print(f"  Avg sequence length: {np.mean([len(s) for s in sequences]):.1f}")
    print(f"  Total tokens: {total_tokens:,}")
    
    print(f"\n🎯 Training Statistics:")
    print(f"  Sequence length: {seq_length}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Training samples: {total_samples:,}")
    print(f"  Tokens per vocab entry: {total_tokens / vocab_size:.1f}")
    print(f"  Samples per vocab entry: {total_samples / vocab_size:.1f}")
    
    # Check for augmentation info
    if 'augmentation' in config:
        aug = config['augmentation']
        print(f"\n🎵 Augmentation Info:")
        print(f"  Enabled: {aug.get('enabled', False)}")
        print(f"  Transpositions: {aug.get('transpositions', [])}")
        print(f"  Augmentation factor: {aug.get('augmentation_factor', 1)}x")
        
        if 'breakdown' in aug:
            print(f"\n  Breakdown by transposition:")
            for semitones, count in sorted(aug['breakdown'].items()):
                sign = "+" if int(semitones) > 0 else ""
                print(f"    {sign}{semitones} semitones: {count:,} sequences")
    
    return {
        'name': name,
        'num_sequences': len(sequences),
        'num_valid': len(valid_seqs),
        'total_tokens': total_tokens,
        'total_samples': total_samples,
        'vocab_size': vocab_size,
        'seq_length': seq_length,
        'augmented': 'augmentation' in config
    }


def compare_datasets(original_stats, augmented_stats):
    """Compare two datasets."""
    if not original_stats or not augmented_stats:
        print("\n❌ Cannot compare - missing dataset statistics")
        return
    
    print(f"\n{'=' * 60}")
    print("📈 Comparison: Original vs Augmented")
    print(f"{'=' * 60}")
    
    print(f"\n📊 Sequences:")
    print(f"  Original: {original_stats['num_sequences']:,}")
    print(f"  Augmented: {augmented_stats['num_sequences']:,}")
    print(f"  Increase: {augmented_stats['num_sequences'] / original_stats['num_sequences']:.1f}x")
    
    print(f"\n🎯 Training Samples:")
    print(f"  Original: {original_stats['total_samples']:,}")
    print(f"  Augmented: {augmented_stats['total_samples']:,}")
    print(f"  Increase: {augmented_stats['total_samples'] / original_stats['total_samples']:.1f}x")
    
    print(f"\n📚 Data Density:")
    orig_density = original_stats['total_samples'] / original_stats['vocab_size']
    aug_density = augmented_stats['total_samples'] / augmented_stats['vocab_size']
    print(f"  Original samples per vocab entry: {orig_density:.1f}")
    print(f"  Augmented samples per vocab entry: {aug_density:.1f}")
    print(f"  Improvement: {aug_density / orig_density:.1f}x")
    
    print(f"\n✅ Expected Benefits:")
    print(f"  • {augmented_stats['num_sequences'] / original_stats['num_sequences']:.0f}x more diverse training examples")
    print(f"  • Better generalization (pitch-invariant learning)")
    print(f"  • Reduced overfitting due to increased data variety")
    print(f"  • More tokens per vocab entry for robust learning")


def check_training_results(dataset_name):
    """Check if training results exist for a dataset."""
    log_dir = Path(f"logs/generators/{dataset_name}")
    
    if not log_dir.exists():
        return None
    
    log_file = log_dir / "models.csv"
    if not log_file.exists():
        return None
    
    import csv
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return None
    
    print(f"\n{'=' * 60}")
    print(f"📝 Training Results for {dataset_name}")
    print(f"{'=' * 60}")
    
    print(f"\nRecent runs: {len(rows)}")
    print(f"\nLatest 3 runs:")
    print(f"{'Model':<8} {'Epochs':<8} {'Final Loss':<12} {'Val Loss':<12} {'Gap':<10}")
    print("-" * 60)
    
    for row in rows[-3:]:
        model = row.get('model_type', 'N/A')
        epochs = row.get('epochs', 'N/A')
        final_loss = float(row['final_loss']) if row.get('final_loss') else 0
        
        val_loss = None
        if 'best_val_loss' in row and row['best_val_loss']:
            try:
                val_loss = float(row['best_val_loss'])
            except:
                pass
        
        if val_loss:
            gap = val_loss - final_loss
            print(f"{model:<8} {epochs:<8} {final_loss:<12.4f} {val_loss:<12.4f} {gap:>9.4f}")
        else:
            print(f"{model:<8} {epochs:<8} {final_loss:<12.4f} {'N/A':<12} {'N/A':<10}")
    
    return rows


def main():
    print("\n" + "=" * 60)
    print("🎵 MIDITok Data Augmentation Analysis")
    print("=" * 60)
    
    # Analyze datasets
    original_stats = analyze_dataset("data/miditok", "Original MIDITok")
    augmented_stats = analyze_dataset("data/miditok_augmented", "Augmented MIDITok")
    
    # Compare if both exist
    if original_stats and augmented_stats:
        compare_datasets(original_stats, augmented_stats)
    
    # Check training results
    if original_stats:
        check_training_results("miditok")
    
    if augmented_stats:
        check_training_results("miditok_augmented")
    
    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)
    
    if not augmented_stats:
        print("\n💡 To create augmented dataset:")
        print("   python utils/augment_miditok.py")
        print("\n💡 Or run the full pipeline:")
        print("   bash scripts/train_augmented.sh")
    
    print()


if __name__ == "__main__":
    main()
