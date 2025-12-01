"""
Plotting script for Comprehensive Evaluation Experiment (Experiment 1).
Based on instructions for remaking plotting for experiment1.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import pretty_midi
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

COLORS = {
    'Baseline': '#95a5a6',  # Grayish
    'Guided': '#3498db',    # Blue
    'Improvement': '#2ecc71', # Green
    'Decline': '#e74c3c',      # Red
    'Neutral': '#7f8c8d'
}

METRICS = ['mean_pitch_range', 'mean_scale_consistency', 'mean_pitch_entropy', 'mean_pitch_class_entropy']
DISPLAY_NAMES = {
    'scale_consistency': 'Scale Consistency',
    'pitch_entropy': 'Pitch Entropy',
    'pitch_class_entropy': 'PC Entropy',
    'pitch_range': 'Pitch Range',
    'note_density': 'Note Density',
    'avg_polyphony': 'Avg Polyphony'
}

def load_data(json_path):
    """Load JSON results and return a combined DataFrame."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df_base = pd.DataFrame(data['baseline_results']).sort_values('sample_id').set_index('sample_id')
    df_guid = pd.DataFrame(data['guided_results']).sort_values('sample_id').set_index('sample_id')
    
    # Combine into a single DataFrame with suffixes
    df = pd.concat([df_base.add_suffix('_base'), df_guid.add_suffix('_guid')], axis=1)
    
    # Also create a long-form DataFrame for some plots
    df_base['Condition'] = 'Baseline'
    df_guid['Condition'] = 'Guided'
    df_long = pd.concat([df_base, df_guid]).reset_index()
    
    return df, df_long, data

def plot_headline_summary(df_long, output_dir):
    """(A) Headline summary plots."""
    print("Generating Group A: Headline summary plots...")
    
    # 1. Bar plot of summary metrics (baseline vs guided)
    metrics = ['pitch_range', 'scale_consistency', 'pitch_entropy', 'pitch_class_entropy']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    for i, metric in enumerate(metrics):
        sns.barplot(data=df_long, x='Condition', y=metric, ax=axes[i], 
                    palette=[COLORS['Baseline'], COLORS['Guided']], capsize=.1, errorbar='sd')
        axes[i].set_title(DISPLAY_NAMES.get(metric, metric))
        axes[i].set_xlabel('Condition')
        axes[i].set_ylabel('Value')
        
    plt.suptitle('Summary Metrics: Baseline vs Guided (Mean ± Std)', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'A_headline_summary_bars.png', bbox_inches='tight')
    plt.close()

def plot_paired_comparisons(df, output_dir):
    """(B) Paired / per-sample comparisons."""
    print("Generating Group B: Paired / per-sample comparisons...")
    
    metrics = ['scale_consistency', 'pitch_entropy', 'pitch_class_entropy', 'pitch_range']
    
    # 1. Paired-lines / slopeplot for scale_consistency
    plt.figure(figsize=(10, 6))
    
    # Prepare data for slope chart
    metric = 'scale_consistency'
    col_base = f'{metric}_base'
    col_guid = f'{metric}_guid'
    
    # Sort by improvement
    df['delta_sc'] = df[col_guid] - df[col_base]
    df_sorted = df.sort_values('delta_sc')
    
    # Plot lines
    for idx, row in df_sorted.iterrows():
        color = COLORS['Improvement'] if row[col_guid] > row[col_base] else COLORS['Decline']
        plt.plot([0, 1], [row[col_base], row[col_guid]], color=color, alpha=0.4, linewidth=1)
        
    # Plot means
    plt.plot([0, 1], [df[col_base].mean(), df[col_guid].mean()], color='black', linewidth=3, marker='o')
    
    plt.xticks([0, 1], ['Baseline', 'Guided'])
    plt.xlabel('Condition')
    plt.ylabel(DISPLAY_NAMES[metric])
    plt.title(f'Paired Change in {DISPLAY_NAMES[metric]} (Per Sample)')
    plt.savefig(output_dir / 'B_paired_slope_scale_consistency.png')
    plt.close()
    
    # 2. Paired scatter with identity line
    for metric in metrics:
        plt.figure(figsize=(6, 6))
        col_base = f'{metric}_base'
        col_guid = f'{metric}_guid'
        
        sns.scatterplot(data=df, x=col_base, y=col_guid, alpha=0.6)
        
        # Identity line
        m = min(df[col_base].min(), df[col_guid].min())
        M = max(df[col_base].max(), df[col_guid].max())
        plt.plot([m, M], [m, M], 'k--', linewidth=1)
        
        plt.xlabel(f'{DISPLAY_NAMES[metric]} (Baseline)')
        plt.ylabel(f'{DISPLAY_NAMES[metric]} (Guided)')
        plt.title(f'{DISPLAY_NAMES[metric]}: Baseline vs Guided')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(output_dir / f'B_paired_scatter_{metric}.png')
        plt.close()

    # 3. Histogram of per-sample deltas
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        col_base = f'{metric}_base'
        col_guid = f'{metric}_guid'
        delta = df[col_guid] - df[col_base]
        
        sns.histplot(delta, kde=True, ax=axes[i], color=COLORS['Guided'])
        axes[i].axvline(0, color='k', linestyle='--')
        axes[i].set_title(f'Delta {DISPLAY_NAMES[metric]}')
        axes[i].set_xlabel('Guided - Baseline')
        axes[i].set_ylabel('Count')
        
    plt.tight_layout()
    plt.savefig(output_dir / 'B_delta_histograms.png')
    plt.close()

def plot_distributions_relationships(df, df_long, output_dir):
    """(C) Distributions & relationships."""
    print("Generating Group C: Distributions & relationships...")
    
    metrics = ['scale_consistency', 'pitch_entropy', 'pitch_class_entropy', 'pitch_range']
    
    # 1. Violin Plots
    # Filter df_long to only include relevant metrics
    df_melted = df_long.melt(id_vars=['Condition', 'sample_id'], value_vars=metrics, var_name='Metric', value_name='Value')
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, metric in enumerate(metrics):
        sns.violinplot(data=df_long, x='Condition', y=metric, ax=axes[i], 
                       palette=[COLORS['Baseline'], COLORS['Guided']], inner='box')
        axes[i].set_title(DISPLAY_NAMES[metric])
        axes[i].set_xlabel('Condition')
        axes[i].set_ylabel('Value')
        
    plt.tight_layout()
    plt.savefig(output_dir / 'C_violin_distributions.png')
    plt.close()
    
    # 2. Scatter: pitch_entropy vs scale_consistency
    plt.figure(figsize=(8, 6))
    
    # Plot arrows
    for idx, row in df.iterrows():
        plt.arrow(row['pitch_entropy_base'], row['scale_consistency_base'],
                  row['pitch_entropy_guid'] - row['pitch_entropy_base'],
                  row['scale_consistency_guid'] - row['scale_consistency_base'],
                  color='gray', alpha=0.3, length_includes_head=True, head_width=0.01)
                  
    # Plot points
    plt.scatter(df['pitch_entropy_base'], df['scale_consistency_base'], 
                c=COLORS['Baseline'], label='Baseline', alpha=0.6)
    plt.scatter(df['pitch_entropy_guid'], df['scale_consistency_guid'], 
                c=COLORS['Guided'], label='Guided', alpha=0.8)
                
    plt.xlabel('Pitch Entropy')
    plt.ylabel('Scale Consistency')
    plt.title('Entropy vs Scale Consistency Shift')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'C_scatter_entropy_vs_consistency.png')
    plt.close()

    # 3. CDFs
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, metric in enumerate(['scale_consistency', 'pitch_entropy']):
        sns.ecdfplot(data=df_long, x=metric, hue='Condition', ax=axes[i], 
                     palette=[COLORS['Baseline'], COLORS['Guided']])
        axes[i].set_title(f'CDF of {DISPLAY_NAMES[metric]}')
        axes[i].set_xlabel(DISPLAY_NAMES[metric])
        axes[i].set_ylabel('Proportion')
        
    plt.tight_layout()
    plt.savefig(output_dir / 'C_cdfs.png')
    plt.close()

def plot_musical_visualizations(df, output_dir):
    """(D) Musical / sequence visualizations."""
    print("Generating Group D: Musical / sequence visualizations...")
    
    # Select samples: Best improvement in scale consistency, Median, Worst
    df['delta_sc'] = df['scale_consistency_guid'] - df['scale_consistency_base']
    df_sorted = df.sort_values('delta_sc', ascending=False)
    
    top_sample = df_sorted.iloc[0]
    median_sample = df_sorted.iloc[len(df)//2]
    worst_sample = df_sorted.iloc[-1]
    
    samples_to_plot = [
        ('Best Improvement', top_sample),
        ('Median Change', median_sample),
        ('Worst/Negative Change', worst_sample)
    ]
    
    for label, sample in samples_to_plot:
        try:
            base_midi_path = sample['output_file_base']
            guid_midi_path = sample['output_file_guid']
            
            # Load MIDIs
            pm_base = pretty_midi.PrettyMIDI(base_midi_path)
            pm_guid = pretty_midi.PrettyMIDI(guid_midi_path)
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot Piano Rolls
            plot_piano_roll(pm_base, axes[0], f"{label} - Baseline (Scale Consistency={sample['scale_consistency_base']:.2f})")
            plot_piano_roll(pm_guid, axes[1], f"{label} - Guided (Scale Consistency={sample['scale_consistency_guid']:.2f})")
            
            plt.tight_layout()
            safe_label = label.replace(' ', '_').replace('/', '_')
            plt.savefig(output_dir / f'D_pianoroll_{safe_label}.png')
            plt.close()
            
            # Pitch Class Histogram
            plot_pitch_class_comparison(pm_base, pm_guid, label, output_dir)
            
        except Exception as e:
            print(f"Could not process MIDI for {label}: {e}")

def plot_piano_roll(pm, ax, title):
    """Helper to plot piano roll."""
    # Get piano roll (pitch x time)
    pr = pm.get_piano_roll(fs=100)
    # Crop to active range
    pitch_indices = np.where(np.any(pr > 0, axis=1))[0]
    if len(pitch_indices) > 0:
        min_pitch, max_pitch = pitch_indices[0], pitch_indices[-1]
        pr = pr[max(0, min_pitch-2):min(128, max_pitch+3), :]
        extent = [0, pm.get_end_time(), max(0, min_pitch-2), min(128, max_pitch+3)]
    else:
        extent = [0, pm.get_end_time(), 0, 128]
        
    ax.imshow(pr, aspect='auto', origin='lower', cmap='magma', interpolation='nearest', extent=extent)
    ax.set_title(title)
    ax.set_ylabel('Pitch')
    ax.set_xlabel('Time (s)')

def plot_pitch_class_comparison(pm_base, pm_guid, label, output_dir):
    """Helper to plot pitch class histograms."""
    def get_pc_hist(pm):
        chroma = pm.get_chroma()
        return np.sum(chroma, axis=1)
    
    hist_base = get_pc_hist(pm_base)
    hist_guid = get_pc_hist(pm_guid)
    
    # Normalize
    hist_base /= (hist_base.sum() + 1e-6)
    hist_guid /= (hist_guid.sum() + 1e-6)
    
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    x = np.arange(12)
    
    plt.figure(figsize=(10, 5))
    width = 0.35
    plt.bar(x - width/2, hist_base, width, label='Baseline', color=COLORS['Baseline'], alpha=0.7)
    plt.bar(x + width/2, hist_guid, width, label='Guided', color=COLORS['Guided'], alpha=0.7)
    
    plt.xticks(x, pitch_classes)
    plt.title(f'Pitch Class Distribution: {label}')
    plt.xlabel('Pitch Class')
    plt.ylabel('Proportion')
    plt.legend()
    plt.tight_layout()
    safe_label = label.replace(' ', '_').replace('/', '_')
    plt.savefig(output_dir / f'D_pc_hist_{safe_label}.png')
    plt.close()

def plot_ranking_selection(df, output_dir):
    """(E) Ranking / selection."""
    print("Generating Group E: Ranking / selection...")
    
    # Ranked waterfall of change magnitude (Scale Consistency)
    df['delta_sc'] = df['scale_consistency_guid'] - df['scale_consistency_base']
    df_sorted = df.sort_values('delta_sc', ascending=True).reset_index()
    
    plt.figure(figsize=(12, 6))
    colors = [COLORS['Improvement'] if x >= 0 else COLORS['Decline'] for x in df_sorted['delta_sc']]
    
    plt.bar(range(len(df_sorted)), df_sorted['delta_sc'], color=colors)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel('Sample Rank')
    plt.ylabel('Change in Scale Consistency')
    plt.title('Waterfall of Scale Consistency Improvement')
    plt.tight_layout()
    plt.savefig(output_dir / 'E_waterfall_scale_consistency.png')
    plt.close()

def plot_combined_visual_stats(df, output_dir):
    """(F) Compact reproducible recipe & statistical rigor."""
    print("Generating Group F: Combined visual + stats...")
    
    metrics = ['scale_consistency', 'pitch_entropy']
    
    for metric in metrics:
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
        
        col_base = f'{metric}_base'
        col_guid = f'{metric}_guid'
        delta = df[col_guid] - df[col_base]
        
        # 1. Top: Box plot
        ax0 = fig.add_subplot(gs[0])
        data_melt = pd.melt(df[[col_base, col_guid]].rename(columns={col_base: 'Baseline', col_guid: 'Guided'}))
        sns.boxplot(data=data_melt, x='variable', y='value', ax=ax0, palette=[COLORS['Baseline'], COLORS['Guided']])
        ax0.set_title(f'{DISPLAY_NAMES[metric]}: Distribution')
        ax0.set_xlabel('Condition')
        ax0.set_ylabel('Value')
        
        # 2. Middle: Paired Lines (Sorted)
        ax1 = fig.add_subplot(gs[1])
        df_sorted = df.sort_values(by=col_base) # Sort by baseline value for clarity
        # Or sort by delta? Instructions say "sorted by delta"
        df_sorted = df.sort_values(by=f'delta_{"sc" if metric=="scale_consistency" else "pe"}', ascending=True)
        # Actually let's just sort by index for now or baseline
        
        # Let's stick to sorting by delta for the waterfall effect, but for paired lines, 
        # maybe sorting by baseline value makes the "fan" effect visible?
        # The prompt says "paired lines for samples sorted by delta".
        
        # We need to iterate carefully
        x = np.arange(len(df))
        # This is tricky to plot as "paired lines" on a 2D plot if we want to show magnitude.
        # Usually paired lines means x=Condition.
        # If "sorted by delta", maybe they mean the slope plot we did in B?
        # Let's do the slope plot again but cleaner.
        
        for i, (idx, row) in enumerate(df.iterrows()):
            c = COLORS['Improvement'] if row[col_guid] > row[col_base] else COLORS['Decline']
            if metric == 'pitch_entropy': # Lower is usually better for entropy if we want structure? Or higher?
                # Assuming we want lower entropy for structure, or higher for diversity.
                # Let's stick to: Green if Guided > Baseline (Increase)
                pass
            
            ax1.plot([0, 1], [row[col_base], row[col_guid]], color=c, alpha=0.3)
            
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Baseline', 'Guided'])
        ax1.set_title('Paired Changes')
        ax1.set_xlabel('Condition')
        ax1.set_ylabel('Value')
        
        # 3. Bottom: Histogram of Deltas with Stats
        ax2 = fig.add_subplot(gs[2])
        sns.histplot(delta, kde=True, ax=ax2, color=COLORS['Guided'])
        ax2.axvline(0, color='k', linestyle='--')
        
        # Stats
        t_stat, p_val = stats.ttest_rel(df[col_guid], df[col_base])
        cohen_d = delta.mean() / delta.std(ddof=1)
        
        stats_text = f"Paired t-test p = {p_val:.2e}\nCohen's d = {cohen_d:.2f}\nMean Delta = {delta.mean():.3f}"
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, va='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_title('Distribution of Differences (Guided - Baseline)')
        ax2.set_xlabel('Difference')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'F_combined_{metric}.png')
        plt.close()

def main():
    results_file = Path("experiments/experiment1/results_miditok/comprehensive/experiment_results.json")
    #results_file = Path("experiments/experiment1/results_naive/comprehensive/experiment_results.json")
    plot_dir = Path("experiments/experiment1/results_miditok/comprehensive/plots_v2")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
        
    print(f"Loading results from {results_file}...")
    df, df_long, data = load_data(results_file)
    
    # Add delta columns if not present
    if 'delta_sc' not in df.columns:
        df['delta_sc'] = df['scale_consistency_guid'] - df['scale_consistency_base']
    if 'delta_pe' not in df.columns:
        df['delta_pe'] = df['pitch_entropy_guid'] - df['pitch_entropy_base']
    
    plot_headline_summary(df_long, plot_dir)
    plot_paired_comparisons(df, plot_dir)
    plot_distributions_relationships(df, df_long, plot_dir)
    plot_musical_visualizations(df, plot_dir)
    plot_ranking_selection(df, plot_dir)
    plot_combined_visual_stats(df, plot_dir)
    
    print(f"All plots saved to {plot_dir}")

if __name__ == "__main__":
    main()
