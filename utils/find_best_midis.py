"""
Find and rank the best generated MIDI files for presentation demos.
"""

import pandas as pd
from pathlib import Path
import shutil
import argparse

def find_best_midis(output_dir="presentation/best_midis", top_n=10):
    """Find the best generated MIDI files based on quality metrics."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all generation and evaluation logs
    generation_files = list(Path("logs/generators").glob("*/midi/output_midis.csv"))
    evaluation_files = list(Path("logs/generators").glob("*/midi/evaluation_log.csv"))
    
    if not generation_files or not evaluation_files:
        print("No log files found!")
        return
    
    # Combine logs
    gen_dfs = []
    eval_dfs = []
    
    for file in generation_files:
        df = pd.read_csv(file, on_bad_lines='skip')
        df['dataset'] = file.parent.parent.name
        gen_dfs.append(df)
    
    for file in evaluation_files:
        df = pd.read_csv(file, on_bad_lines='skip')
        df['dataset'] = file.parent.parent.name
        eval_dfs.append(df)
    
    gen_df = pd.concat(gen_dfs, ignore_index=True)
    eval_df = pd.concat(eval_dfs, ignore_index=True)
    
    # Merge
    merged = pd.merge(gen_df, eval_df, on='output_file', suffixes=('_gen', '_eval'))
    
    if merged.empty:
        print("No matching generation and evaluation data found!")
        return
    
    print(f"Found {len(merged)} generated MIDI files with evaluations\n")
    
    # Calculate quality score
    # Higher pitch range = more interesting
    # Moderate note density = not too sparse or dense
    # Longer duration = more substantial piece
    
    merged['pitch_range_score'] = merged['pitch_range'] / merged['pitch_range'].max()
    merged['density_score'] = 1 - abs(merged['note_density'] - merged['note_density'].median()) / merged['note_density'].std()
    merged['duration_score'] = merged['duration'] / merged['duration'].max()
    
    # Penalize greedy sampling (tends to be repetitive)
    merged['strategy_bonus'] = merged['strategy'].apply(lambda x: 1.5 if x != 'greedy' else 1.0)
    
    # Bonus for higher temperature (more creative)
    merged['temp_bonus'] = merged['temperature'].apply(lambda x: 1.2 if x > 1.2 else 1.0)
    
    # Overall quality score
    merged['quality_score'] = (
        merged['pitch_range_score'] * 0.4 +
        merged['density_score'] * 0.3 +
        merged['duration_score'] * 0.3
    ) * merged['strategy_bonus'] * merged['temp_bonus']
    
    # Sort by quality
    best = merged.nlargest(top_n, 'quality_score')
    
    print("=" * 80)
    print(f"TOP {top_n} GENERATED MIDI FILES FOR PRESENTATION")
    print("=" * 80)
    print()
    
    for rank, (idx, row) in enumerate(best.iterrows(), 1):
        print(f"#{rank} - Quality Score: {row['quality_score']:.3f}")
        print(f"   File: {Path(row['output_file']).name}")
        print(f"   Strategy: {row['strategy']}, Temperature: {row['temperature']}")
        print(f"   Pitch Range: {row['pitch_range']:.0f} semitones")
        print(f"   Note Density: {row['note_density']:.2f} notes/sec")
        print(f"   Duration: {row['duration']:.1f}s")
        print(f"   Dataset: {row['dataset_gen']}")
        
        # Copy to best_midis directory
        src = Path(row['output_file'])
        if src.exists():
            dst = output_dir / f"rank{rank:02d}_{src.name}"
            shutil.copy2(src, dst)
            print(f"   ✓ Copied to {dst}")
        else:
            print(f"   ⚠️  File not found: {src}")
        print()
    
    # Save ranking to CSV
    ranking_file = output_dir / "midi_rankings.csv"
    best[['output_file', 'quality_score', 'strategy', 'temperature', 
          'pitch_range', 'note_density', 'duration', 'dataset_gen']].to_csv(
        ranking_file, index=False
    )
    print(f"✅ Rankings saved to {ranking_file}")
    
    # Print category winners
    print("\n" + "=" * 80)
    print("CATEGORY WINNERS")
    print("=" * 80)
    print()
    
    # Most diverse (highest pitch range)
    most_diverse = merged.nlargest(1, 'pitch_range').iloc[0]
    print(f"🎨 Most Diverse:")
    print(f"   {Path(most_diverse['output_file']).name}")
    print(f"   Pitch Range: {most_diverse['pitch_range']:.0f} semitones")
    print()
    
    # Longest
    longest = merged.nlargest(1, 'duration').iloc[0]
    print(f"⏱️  Longest:")
    print(f"   {Path(longest['output_file']).name}")
    print(f"   Duration: {longest['duration']:.1f}s")
    print()
    
    # Best from each strategy
    for strategy in merged['strategy'].unique():
        strategy_best = merged[merged['strategy'] == strategy].nlargest(1, 'quality_score')
        if not strategy_best.empty:
            row = strategy_best.iloc[0]
            print(f"🎯 Best {strategy.upper()}:")
            print(f"   {Path(row['output_file']).name}")
            print(f"   Quality: {row['quality_score']:.3f}")
            print()
    
    print(f"\n✅ All best MIDI files copied to {output_dir}/")
    print("These are ready to use in your presentation demo!")


def main():
    parser = argparse.ArgumentParser(description="Find best generated MIDI files for presentation")
    parser.add_argument("--output_dir", type=str, default="presentation/best_midis",
                        help="Directory to copy best MIDI files to")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top files to select")
    
    args = parser.parse_args()
    
    find_best_midis(args.output_dir, args.top_n)


if __name__ == "__main__":
    main()
