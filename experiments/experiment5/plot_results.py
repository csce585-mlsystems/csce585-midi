"""
Plotting for Experiment 5: Discriminator Architecture Comparison
=================================================================
Creates publication-quality plots comparing how different discriminator
architectures affect guided music generation quality.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np
from scipy import stats

# Set up publication-quality styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme - distinct colors for each discriminator type
COLORS = {
    'none': '#95a5a6',            # Gray - baseline (no discriminator)
    'mlp': '#e74c3c',             # Red - MLP
    'lstm': '#2ecc71',            # Green - LSTM
    'transformer': '#3498db'      # Blue - Transformer
}

# Display labels
LABELS = {
    'none': 'No Discriminator',
    'mlp': 'MLP',
    'lstm': 'LSTM',
    'transformer': 'Transformer'
}

FULL_LABELS = {
    'none': 'Baseline (No Discriminator)',
    'mlp': 'MLP Discriminator',
    'lstm': 'LSTM Discriminator',
    'transformer': 'Transformer Discriminator'
}

SETTING_ORDER = ['conservative', 'balanced', 'creative']
DISCRIMINATOR_ORDER = ['none', 'mlp', 'lstm', 'transformer']


def load_results(results_dir):
    """Load detailed CSV results."""
    csv_file = Path(results_dir) / "detailed_results.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        # Ensure categorical ordering
        df['setting'] = pd.Categorical(df['setting'], categories=SETTING_ORDER, ordered=True)
        df['discriminator_type'] = pd.Categorical(df['discriminator_type'], categories=DISCRIMINATOR_ORDER, ordered=True)
        return df
    else:
        raise FileNotFoundError(f"Results file not found: {csv_file}")


def load_experiment_results(results_dir):
    """Load the full experiment results JSON."""
    json_file = Path(results_dir) / "experiment_results.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            return json.load(f)
    return None


def plot_grouped_bar_comparison(df, metric, output_dir, show=False):
    """
    Create a grouped bar chart comparing a metric across all discriminators and settings.
    Each setting group has 4 bars (one per discriminator type).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by discriminator and setting, calculate mean and std
    grouped = df.groupby(['setting', 'discriminator_type'], observed=True)[metric].agg(['mean', 'std']).reset_index()
    
    # Bar positions
    settings = SETTING_ORDER
    n_settings = len(settings)
    n_discs = len(DISCRIMINATOR_ORDER)
    bar_width = 0.2
    x = np.arange(n_settings)
    
    # Plot bars for each discriminator
    for i, disc in enumerate(DISCRIMINATOR_ORDER):
        disc_data = grouped[grouped['discriminator_type'] == disc]
        if disc_data.empty:
            continue
        # Reorder to match settings order
        disc_data = disc_data.set_index('setting').reindex(settings).reset_index()
        
        offset = (i - 1.5) * bar_width  # Center the groups
        bars = ax.bar(x + offset, disc_data['mean'], bar_width,
                     yerr=disc_data['std'], capsize=4,
                     label=FULL_LABELS[disc],
                     color=COLORS[disc],
                     edgecolor='white', linewidth=0.5)
    
    # Formatting
    metric_title = metric.replace('_', ' ').title()
    ax.set_xlabel('Generation Setting', fontweight='bold')
    ax.set_ylabel(metric_title, fontweight='bold')
    ax.set_title(f'{metric_title} by Discriminator Architecture', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in settings])
    ax.legend(title='Discriminator', loc='best', framealpha=0.9)
    
    # Add grid for y-axis only
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"{metric}_grouped_bars.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_discriminator_comparison_boxplot(df, metric, output_dir, show=False):
    """
    Create side-by-side boxplots comparing discriminators, with separate boxes for each setting.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create a combined column for x-axis
    df_plot = df.copy()
    # Convert to string to avoid categorical mapping issues
    df_plot['discriminator_type'] = df_plot['discriminator_type'].astype(str)
    df_plot['disc_label'] = df_plot['discriminator_type'].map(LABELS)
    
    # Get discriminators actually present in the data
    present_discs = df_plot['discriminator_type'].dropna().unique().tolist()
    present_disc_labels = [LABELS[d] for d in DISCRIMINATOR_ORDER if d in present_discs]
    
    palette = {LABELS[d]: COLORS[d] for d in DISCRIMINATOR_ORDER if d in present_discs}
    
    sns.boxplot(data=df_plot, x='setting', y=metric, hue='disc_label',
                palette=palette, ax=ax, order=SETTING_ORDER,
                hue_order=present_disc_labels)
    
    # Add individual points
    sns.stripplot(data=df_plot, x='setting', y=metric, hue='disc_label',
                  palette=palette, ax=ax, order=SETTING_ORDER,
                  hue_order=present_disc_labels,
                  dodge=True, alpha=0.6, size=4, legend=False)
    
    metric_title = metric.replace('_', ' ').title()
    ax.set_xlabel('Generation Setting', fontweight='bold')
    ax.set_ylabel(metric_title, fontweight='bold')
    ax.set_title(f'Distribution of {metric_title} by Discriminator', fontsize=14, fontweight='bold')
    ax.set_xticklabels([s.title() for s in SETTING_ORDER])
    ax.legend(title='Discriminator', loc='best')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"{metric}_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_improvement_over_baseline(df, output_dir, show=False):
    """
    Plot the improvement (or degradation) of each discriminator over baseline.
    Shows the delta from baseline for key metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['scale_consistency', 'pitch_entropy', 'pitch_class_entropy']
    metric_titles = ['Scale Consistency', 'Pitch Entropy', 'Pitch Class Entropy']
    
    # Get balanced setting only for cleaner comparison
    balanced_df = df[df['setting'] == 'balanced']
    
    # Calculate baseline means
    baseline = balanced_df[balanced_df['discriminator_type'] == 'none']
    if baseline.empty:
        print("Warning: No baseline data found")
        return
    
    for ax, metric, title in zip(axes, metrics, metric_titles):
        baseline_mean = baseline[metric].mean()
        
        improvements = []
        errors = []
        disc_labels = []
        
        for disc in ['mlp', 'lstm', 'transformer']:
            disc_data = balanced_df[balanced_df['discriminator_type'] == disc]
            if not disc_data.empty:
                disc_mean = disc_data[metric].mean()
                disc_std = disc_data[metric].std()
                improvement = ((disc_mean - baseline_mean) / baseline_mean) * 100
                error = (disc_std / baseline_mean) * 100
                
                improvements.append(improvement)
                errors.append(error)
                disc_labels.append(LABELS[disc])
        
        if improvements:
            # Get colors for discriminators that have data
            disc_keys = [d for d in ['mlp', 'lstm', 'transformer'] 
                        if LABELS[d] in disc_labels]
            colors = [COLORS[d] for d in disc_keys]
            
            bars = ax.bar(disc_labels, improvements, yerr=errors, capsize=5,
                         color=colors, edgecolor='white', linewidth=0.5)
            
            # Add horizontal line at 0
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            # Color bars based on improvement direction
            for bar, imp in zip(bars, improvements):
                if imp < 0:
                    bar.set_hatch('//')
            
            ax.set_ylabel('% Change from Baseline', fontweight='bold')
            ax.set_title(f'{title}', fontweight='bold')
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle('Improvement over Baseline (No Discriminator) - Balanced Setting',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "improvement_over_baseline.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_discriminator_overview(df, results, output_dir, show=False):
    """
    Create a comprehensive overview plot with multiple panels showing
    discriminator effects on generation quality.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # Get balanced setting for cleaner comparison
    balanced_df = df[df['setting'] == 'balanced']
    
    # Panel 1: Scale Consistency comparison
    ax1 = fig.add_subplot(gs[0, 0])
    scale_means = balanced_df.groupby('discriminator_type')['scale_consistency'].mean()
    scale_stds = balanced_df.groupby('discriminator_type')['scale_consistency'].std()
    
    discs_present = [d for d in DISCRIMINATOR_ORDER if d in scale_means.index]
    bars = ax1.bar(discs_present, [scale_means.get(d, 0) for d in discs_present],
                   yerr=[scale_stds.get(d, 0) for d in discs_present],
                   color=[COLORS[d] for d in discs_present], capsize=5,
                   edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Scale Consistency', fontweight='bold')
    ax1.set_title('Harmonic Quality', fontweight='bold')
    ax1.set_xticklabels([LABELS[d] for d in discs_present], rotation=15)
    ax1.set_ylim(0, 1.1)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Panel 2: Pitch Entropy comparison
    ax2 = fig.add_subplot(gs[0, 1])
    entropy_means = balanced_df.groupby('discriminator_type')['pitch_entropy'].mean()
    entropy_stds = balanced_df.groupby('discriminator_type')['pitch_entropy'].std()
    
    bars = ax2.bar(discs_present, [entropy_means.get(d, 0) for d in discs_present],
                   yerr=[entropy_stds.get(d, 0) for d in discs_present],
                   color=[COLORS[d] for d in discs_present], capsize=5,
                   edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Pitch Entropy', fontweight='bold')
    ax2.set_title('Pitch Diversity', fontweight='bold')
    ax2.set_xticklabels([LABELS[d] for d in discs_present], rotation=15)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Panel 3: Pitch Class Entropy comparison
    ax3 = fig.add_subplot(gs[0, 2])
    pc_entropy_means = balanced_df.groupby('discriminator_type')['pitch_class_entropy'].mean()
    pc_entropy_stds = balanced_df.groupby('discriminator_type')['pitch_class_entropy'].std()
    
    bars = ax3.bar(discs_present, [pc_entropy_means.get(d, 0) for d in discs_present],
                   yerr=[pc_entropy_stds.get(d, 0) for d in discs_present],
                   color=[COLORS[d] for d in discs_present], capsize=5,
                   edgecolor='white', linewidth=0.5)
    ax3.set_ylabel('Pitch Class Entropy', fontweight='bold')
    ax3.set_title('Chromatic Distribution', fontweight='bold')
    ax3.set_xticklabels([LABELS[d] for d in discs_present], rotation=15)
    ax3.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Panel 4: Note Density comparison
    ax4 = fig.add_subplot(gs[1, 0])
    density_means = balanced_df.groupby('discriminator_type')['note_density'].mean()
    density_stds = balanced_df.groupby('discriminator_type')['note_density'].std()
    
    bars = ax4.bar(discs_present, [density_means.get(d, 0) for d in discs_present],
                   yerr=[density_stds.get(d, 0) for d in discs_present],
                   color=[COLORS[d] for d in discs_present], capsize=5,
                   edgecolor='white', linewidth=0.5)
    ax4.set_ylabel('Notes per Second', fontweight='bold')
    ax4.set_title('Note Density', fontweight='bold')
    ax4.set_xticklabels([LABELS[d] for d in discs_present], rotation=15)
    ax4.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Panel 5: Pitch Range comparison
    ax5 = fig.add_subplot(gs[1, 1])
    range_means = balanced_df.groupby('discriminator_type')['pitch_range'].mean()
    range_stds = balanced_df.groupby('discriminator_type')['pitch_range'].std()
    
    bars = ax5.bar(discs_present, [range_means.get(d, 0) for d in discs_present],
                   yerr=[range_stds.get(d, 0) for d in discs_present],
                   color=[COLORS[d] for d in discs_present], capsize=5,
                   edgecolor='white', linewidth=0.5)
    ax5.set_ylabel('Pitch Range (semitones)', fontweight='bold')
    ax5.set_title('Melodic Range', fontweight='bold')
    ax5.set_xticklabels([LABELS[d] for d in discs_present], rotation=15)
    ax5.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Panel 6: Summary radar chart
    ax6 = fig.add_subplot(gs[1, 2], projection='polar')
    metrics = ['scale_consistency', 'pitch_entropy', 'note_density', 'pitch_range']
    
    # Normalize metrics for radar
    balanced_stats = {}
    for d in discs_present:
        disc_data = balanced_df[balanced_df['discriminator_type'] == d]
        balanced_stats[d] = [disc_data[metric].mean() for metric in metrics]
    
    # Normalize each metric to 0-1
    for i in range(len(metrics)):
        values = [balanced_stats[d][i] for d in discs_present if d in balanced_stats]
        if values:
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                for d in discs_present:
                    if d in balanced_stats:
                        balanced_stats[d][i] = (balanced_stats[d][i] - min_val) / (max_val - min_val)
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    for disc in discs_present:
        if disc in balanced_stats:
            values = balanced_stats[disc] + [balanced_stats[disc][0]]
            ax6.plot(angles, values, 'o-', color=COLORS[disc], 
                    label=LABELS[disc], linewidth=2, markersize=4)
            ax6.fill(angles, values, color=COLORS[disc], alpha=0.1)
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels([m.replace('_', '\n').title() for m in metrics], size=8)
    ax6.set_title('Overall Profile', fontweight='bold', y=1.1)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
    
    fig.suptitle('Experiment 5: Discriminator Architecture Comparison Overview', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "discriminator_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_setting_effect_by_discriminator(df, output_dir, show=False):
    """
    Line plot showing how generation setting affects metrics for each discriminator.
    """
    metrics = ['scale_consistency', 'pitch_entropy', 'pitch_range']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric in zip(axes, metrics):
        for disc in DISCRIMINATOR_ORDER:
            disc_data = df[df['discriminator_type'] == disc]
            if disc_data.empty:
                continue
            disc_grouped = disc_data.groupby('setting')[metric].agg(['mean', 'std'])
            disc_grouped = disc_grouped.reindex(SETTING_ORDER)
            
            x = range(len(SETTING_ORDER))
            ax.errorbar(x, disc_grouped['mean'], yerr=disc_grouped['std'],
                       fmt='o-', color=COLORS[disc], label=LABELS[disc],
                       linewidth=2, markersize=8, capsize=4)
        
        metric_title = metric.replace('_', ' ').title()
        ax.set_xlabel('Generation Setting', fontweight='bold')
        ax.set_ylabel(metric_title, fontweight='bold')
        ax.set_title(f'{metric_title} by Setting', fontweight='bold')
        ax.set_xticks(range(len(SETTING_ORDER)))
        ax.set_xticklabels([s.title() for s in SETTING_ORDER])
        ax.legend(loc='best')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle('Effect of Generation Setting on Metrics by Discriminator',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "setting_effect_by_discriminator.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_pairwise_comparison(df, output_dir, show=False):
    """
    Create scatter plots comparing pairs of metrics, colored by discriminator.
    Helps visualize tradeoffs between different quality dimensions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    balanced_df = df[df['setting'] == 'balanced']
    
    pairs = [
        ('scale_consistency', 'pitch_entropy'),
        ('scale_consistency', 'pitch_range'),
        ('pitch_entropy', 'note_density')
    ]
    
    for ax, (x_metric, y_metric) in zip(axes, pairs):
        for disc in DISCRIMINATOR_ORDER:
            disc_data = balanced_df[balanced_df['discriminator_type'] == disc]
            if not disc_data.empty:
                ax.scatter(disc_data[x_metric], disc_data[y_metric],
                          c=COLORS[disc], label=LABELS[disc],
                          s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        x_title = x_metric.replace('_', ' ').title()
        y_title = y_metric.replace('_', ' ').title()
        ax.set_xlabel(x_title, fontweight='bold')
        ax.set_ylabel(y_title, fontweight='bold')
        ax.set_title(f'{x_title} vs {y_title}', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)
    
    fig.suptitle('Metric Tradeoffs by Discriminator (Balanced Setting)',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "pairwise_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def print_statistical_summary(df, output_dir):
    """Print statistical summary and save to file."""
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("EXPERIMENT 5: STATISTICAL SUMMARY")
    summary_lines.append("=" * 80)
    
    balanced_df = df[df['setting'] == 'balanced']
    metrics = ['scale_consistency', 'pitch_entropy', 'pitch_class_entropy', 
               'note_density', 'pitch_range']
    
    # Get baseline stats
    baseline = balanced_df[balanced_df['discriminator_type'] == 'none']
    
    summary_lines.append("\n--- ANOVA Tests (Balanced Setting) ---")
    
    for metric in metrics:
        groups = []
        group_names = []
        for disc in DISCRIMINATOR_ORDER:
            disc_data = balanced_df[balanced_df['discriminator_type'] == disc][metric].dropna()
            if len(disc_data) > 0:
                groups.append(disc_data.values)
                group_names.append(disc)
        
        if len(groups) >= 2:
            # One-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            summary_lines.append(f"\n{metric}:")
            summary_lines.append(f"  F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                summary_lines.append(f"  Result: Significant difference between discriminators (p < 0.05)")
            else:
                summary_lines.append(f"  Result: No significant difference (p >= 0.05)")
            
            # Print means for each discriminator
            for disc in group_names:
                disc_data = balanced_df[balanced_df['discriminator_type'] == disc][metric]
                summary_lines.append(f"    {LABELS[disc]}: {disc_data.mean():.4f} ± {disc_data.std():.4f}")
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save to file
    with open(Path(output_dir) / "statistical_summary.txt", 'w') as f:
        f.write(summary_text)
    print(f"\nSaved: {Path(output_dir) / 'statistical_summary.txt'}")


def generate_all_plots(results_dir, show=False):
    """Generate all plots for the experiment."""
    results_dir = Path(results_dir)
    output_dir = results_dir
    
    print("Loading results...")
    df = load_results(results_dir)
    results = load_experiment_results(results_dir)
    
    print(f"Loaded {len(df)} samples")
    print(f"Discriminators: {df['discriminator_type'].unique().tolist()}")
    print(f"Settings: {df['setting'].unique().tolist()}")
    
    print("\nGenerating plots...")
    
    # Key metrics to plot
    key_metrics = ['scale_consistency', 'pitch_entropy', 'pitch_class_entropy', 
                   'note_density', 'pitch_range']
    
    # Generate individual metric plots
    for metric in key_metrics:
        plot_grouped_bar_comparison(df, metric, output_dir, show=show)
        plot_discriminator_comparison_boxplot(df, metric, output_dir, show=show)
    
    # Generate overview and comparison plots
    plot_discriminator_overview(df, results, output_dir, show=show)
    plot_improvement_over_baseline(df, output_dir, show=show)
    plot_setting_effect_by_discriminator(df, output_dir, show=show)
    plot_pairwise_comparison(df, output_dir, show=show)
    
    # Statistical summary
    print_statistical_summary(df, output_dir)
    
    print("\nAll plots generated!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for discriminator comparison experiment")
    parser.add_argument("--results_dir", type=str, default="experiments/experiment5/results",
                        help="Directory containing experiment results")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively")
    
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir, show=args.show)
