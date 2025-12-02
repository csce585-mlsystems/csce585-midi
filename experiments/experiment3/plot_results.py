"""
Enhanced Plotting for Experiment 3: Tokenization Comparison
============================================================
Creates publication-quality plots comparing the three tokenization strategies.
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

# Color scheme - distinct colors for each model
COLORS = {
    'naive': '#2ecc71',           # Green - simple/natural
    'miditok': '#3498db',         # Blue - structured
    'miditok_augmented': '#9b59b6'  # Purple - enhanced
}

# Display labels
LABELS = {
    'naive': 'Naive',
    'miditok': 'MidiTok',
    'miditok_augmented': 'MidiTok+Aug'
}

FULL_LABELS = {
    'naive': 'Naive Tokenization',
    'miditok': 'MidiTok REMI',
    'miditok_augmented': 'MidiTok REMI + Augmentation'
}

SETTING_ORDER = ['conservative', 'balanced', 'creative']
MODEL_ORDER = ['naive', 'miditok', 'miditok_augmented']


def load_results(results_dir):
    """Load detailed CSV results."""
    csv_file = Path(results_dir) / "detailed_results.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        # Ensure categorical ordering
        df['setting'] = pd.Categorical(df['setting'], categories=SETTING_ORDER, ordered=True)
        df['model_type'] = pd.Categorical(df['model_type'], categories=MODEL_ORDER, ordered=True)
        return df
    else:
        raise FileNotFoundError(f"Results file not found: {csv_file}")


def plot_grouped_bar_comparison(df, metric, output_dir, show=False):
    """
    Create a grouped bar chart comparing a metric across all models and settings.
    Each setting group has 3 bars (one per model).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by model and setting, calculate mean and std
    grouped = df.groupby(['setting', 'model_type'])[metric].agg(['mean', 'std']).reset_index()
    
    # Bar positions
    settings = SETTING_ORDER
    n_settings = len(settings)
    n_models = len(MODEL_ORDER)
    bar_width = 0.25
    x = np.arange(n_settings)
    
    # Plot bars for each model
    for i, model in enumerate(MODEL_ORDER):
        model_data = grouped[grouped['model_type'] == model]
        # Reorder to match settings order
        model_data = model_data.set_index('setting').loc[settings].reset_index()
        
        offset = (i - 1) * bar_width  # Center the groups
        bars = ax.bar(x + offset, model_data['mean'], bar_width,
                     yerr=model_data['std'], capsize=4,
                     label=FULL_LABELS[model],
                     color=COLORS[model],
                     edgecolor='white', linewidth=0.5)
    
    # Formatting
    metric_title = metric.replace('_', ' ').title()
    ax.set_xlabel('Generation Setting', fontweight='bold')
    ax.set_ylabel(metric_title, fontweight='bold')
    ax.set_title(f'{metric_title} by Tokenization Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in settings])
    ax.legend(title='Tokenization', loc='best', framealpha=0.9)
    
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


def plot_model_comparison_boxplot(df, metric, output_dir, show=False):
    """
    Create side-by-side boxplots comparing models, with separate boxes for each setting.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a combined column for x-axis
    df_plot = df.copy()
    df_plot['model_label'] = df_plot['model_type'].map(LABELS)
    
    palette = {LABELS[m]: COLORS[m] for m in MODEL_ORDER}
    
    sns.boxplot(data=df_plot, x='setting', y=metric, hue='model_label',
                palette=palette, ax=ax, order=SETTING_ORDER,
                hue_order=[LABELS[m] for m in MODEL_ORDER])
    
    # Add individual points
    sns.stripplot(data=df_plot, x='setting', y=metric, hue='model_label',
                  palette=palette, ax=ax, order=SETTING_ORDER,
                  hue_order=[LABELS[m] for m in MODEL_ORDER],
                  dodge=True, alpha=0.6, size=4, legend=False)
    
    metric_title = metric.replace('_', ' ').title()
    ax.set_xlabel('Generation Setting', fontweight='bold')
    ax.set_ylabel(metric_title, fontweight='bold')
    ax.set_title(f'Distribution of {metric_title} Across Models', fontsize=14, fontweight='bold')
    ax.set_xticklabels([s.title() for s in SETTING_ORDER])
    ax.legend(title='Model', loc='best')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"{metric}_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_entropy_vs_scale_consistency(df, output_dir, show=False):
    """
    Scatter plot showing the relationship between pitch entropy and scale consistency.
    This reveals the tradeoff between diversity and musicality.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model in MODEL_ORDER:
        model_data = df[df['model_type'] == model]
        
        for setting in SETTING_ORDER:
            setting_data = model_data[model_data['setting'] == setting]
            marker = {'conservative': 'o', 'balanced': 's', 'creative': '^'}[setting]
            alpha = {'conservative': 0.6, 'balanced': 0.8, 'creative': 1.0}[setting]
            
            ax.scatter(setting_data['pitch_entropy'], setting_data['scale_consistency'],
                      c=COLORS[model], marker=marker, s=100, alpha=alpha,
                      edgecolor='white', linewidth=0.5)
    
    # Create custom legend
    model_handles = [mpatches.Patch(color=COLORS[m], label=FULL_LABELS[m]) for m in MODEL_ORDER]
    setting_handles = [
        plt.Line2D([0], [0], marker='o', color='gray', label='Conservative', markersize=8, linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='gray', label='Balanced', markersize=8, linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='gray', label='Creative', markersize=8, linestyle='None'),
    ]
    
    legend1 = ax.legend(handles=model_handles, title='Model', loc='lower left')
    ax.add_artist(legend1)
    ax.legend(handles=setting_handles, title='Setting', loc='lower right')
    
    ax.set_xlabel('Pitch Entropy (higher = more diverse)', fontweight='bold')
    ax.set_ylabel('Scale Consistency (higher = more tonal)', fontweight='bold')
    ax.set_title('Diversity vs. Musicality Tradeoff', fontsize=14, fontweight='bold')
    
    # Add reference lines
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Good tonality threshold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "entropy_vs_scale_scatter.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_temperature_effect(df, output_dir, show=False):
    """
    Line plot showing how temperature (via settings) affects key metrics for each model.
    """
    metrics = ['scale_consistency', 'pitch_entropy', 'pitch_range']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric in zip(axes, metrics):
        for model in MODEL_ORDER:
            model_data = df[df['model_type'] == model].groupby('setting')[metric].agg(['mean', 'std'])
            model_data = model_data.loc[SETTING_ORDER]
            
            x = range(len(SETTING_ORDER))
            ax.plot(x, model_data['mean'], 'o-', color=COLORS[model], 
                   label=LABELS[model], linewidth=2, markersize=8)
            ax.fill_between(x, 
                           model_data['mean'] - model_data['std'],
                           model_data['mean'] + model_data['std'],
                           color=COLORS[model], alpha=0.2)
        
        metric_title = metric.replace('_', ' ').title()
        ax.set_xlabel('Generation Setting', fontweight='bold')
        ax.set_ylabel(metric_title, fontweight='bold')
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(SETTING_ORDER)))
        ax.set_xticklabels([s.title() for s in SETTING_ORDER])
        ax.legend()
    
    fig.suptitle('Effect of Temperature on Generation Quality', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "temperature_effect_lines.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_model_profiles_radar(df, output_dir, setting='balanced', show=False):
    """
    Create a radar chart showing the profile of each model across all metrics.
    """
    df_setting = df[df['setting'] == setting]
    
    metrics = ['note_density', 'pitch_range', 'avg_polyphony', 
               'scale_consistency', 'pitch_entropy', 'pitch_class_entropy']
    
    # Calculate means
    summary = df_setting.groupby('model_type')[metrics].mean()
    
    # Normalize to 0-1 for visualization
    summary_norm = (summary - summary.min()) / (summary.max() - summary.min() + 1e-10)
    
    # Number of variables
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for model in MODEL_ORDER:
        if model in summary_norm.index:
            values = summary_norm.loc[model].values.flatten().tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2.5, 
                   label=FULL_LABELS[model], color=COLORS[model], markersize=6)
            ax.fill(angles, values, alpha=0.15, color=COLORS[model])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics], size=10)
    ax.set_ylim(0, 1)
    
    ax.set_title(f'Model Characteristics ({setting.title()} Setting)', 
                fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"radar_profiles_{setting}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_summary_heatmap(df, output_dir, show=False):
    """
    Create a comprehensive heatmap showing all metrics for all model/setting combinations.
    """
    metrics = ['scale_consistency', 'pitch_entropy', 'pitch_class_entropy',
               'note_density', 'pitch_range', 'avg_polyphony']
    
    # Create pivot table with multi-level index
    summary = df.groupby(['model_type', 'setting'])[metrics].mean()
    
    # Reshape for heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create formatted data
    data_list = []
    row_labels = []
    for model in MODEL_ORDER:
        for setting in SETTING_ORDER:
            if (model, setting) in summary.index:
                data_list.append(summary.loc[(model, setting)].values)
                row_labels.append(f"{LABELS[model]} - {setting.title()}")
    
    data_array = np.array(data_list)
    
    # Normalize columns for color mapping
    data_norm = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0) + 1e-10)
    
    # Create heatmap
    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto')
    
    # Add text annotations with actual values
    for i in range(len(row_labels)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data_array[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Score', fontweight='bold')
    
    ax.set_title('Comprehensive Metric Comparison\n(Values shown, colors normalized by column)', 
                fontsize=14, fontweight='bold')
    
    # Add horizontal lines to separate models
    for i in [3, 6]:
        ax.axhline(y=i-0.5, color='white', linewidth=2)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "comprehensive_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_pairwise_metric_comparison(df, output_dir, show=False):
    """
    Create a pairplot showing relationships between key metrics, colored by model.
    """
    metrics = ['scale_consistency', 'pitch_entropy', 'note_density', 'pitch_range']
    
    df_plot = df.copy()
    df_plot['Model'] = df_plot['model_type'].map(FULL_LABELS)
    
    palette = {FULL_LABELS[m]: COLORS[m] for m in MODEL_ORDER}
    
    g = sns.pairplot(df_plot, vars=metrics, hue='Model', palette=palette,
                     diag_kind='kde', plot_kws={'alpha': 0.7, 's': 60},
                     corner=True)
    
    g.fig.suptitle('Pairwise Metric Relationships by Model', y=1.02, fontsize=14, fontweight='bold')
    
    output_path = Path(output_dir) / "pairwise_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def create_summary_statistics_table(df, output_dir):
    """
    Create a summary statistics table as CSV and formatted text.
    """
    metrics = ['scale_consistency', 'pitch_entropy', 'pitch_class_entropy',
               'note_density', 'pitch_range', 'avg_polyphony']
    
    # Calculate statistics
    summary = df.groupby(['model_type', 'setting'])[metrics].agg(['mean', 'std', 'count'])
    
    # Save to CSV
    csv_path = Path(output_dir) / "summary_statistics.csv"
    summary.to_csv(csv_path)
    print(f"Saved: {csv_path}")
    
    # Also create a formatted markdown table
    md_path = Path(output_dir) / "summary_statistics.md"
    with open(md_path, 'w') as f:
        f.write("# Experiment 3: Summary Statistics\n\n")
        
        for model in MODEL_ORDER:
            f.write(f"\n## {FULL_LABELS[model]}\n\n")
            f.write("| Setting | Scale Consistency | Pitch Entropy | Note Density | Pitch Range |\n")
            f.write("|---------|-------------------|---------------|--------------|-------------|\n")
            
            for setting in SETTING_ORDER:
                row = df[(df['model_type'] == model) & (df['setting'] == setting)]
                if len(row) > 0:
                    sc = f"{row['scale_consistency'].mean():.3f} ± {row['scale_consistency'].std():.3f}"
                    pe = f"{row['pitch_entropy'].mean():.3f} ± {row['pitch_entropy'].std():.3f}"
                    nd = f"{row['note_density'].mean():.2f} ± {row['note_density'].std():.2f}"
                    pr = f"{row['pitch_range'].mean():.1f} ± {row['pitch_range'].std():.1f}"
                    f.write(f"| {setting.title()} | {sc} | {pe} | {nd} | {pr} |\n")
    
    print(f"Saved: {md_path}")


def generate_all_plots(results_dir, show=False):
    """Generate all plots for the experiment."""
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading results from: {results_dir}")
    df = load_results(results_dir)
    
    print(f"Loaded {len(df)} samples")
    print(f"Models: {list(df['model_type'].unique())}")
    print(f"Settings: {list(df['setting'].unique())}")
    
    # Generate all plots
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    # 1. Grouped bar charts for key metrics
    print("\n1. Grouped bar charts...")
    for metric in ['scale_consistency', 'pitch_entropy', 'pitch_class_entropy', 
                   'note_density', 'pitch_range', 'avg_polyphony']:
        plot_grouped_bar_comparison(df, metric, plots_dir, show)
    
    # 2. Box plots
    print("\n2. Box plots...")
    for metric in ['scale_consistency', 'pitch_entropy']:
        plot_model_comparison_boxplot(df, metric, plots_dir, show)
    
    # 3. Scatter plot: Entropy vs Scale Consistency
    print("\n3. Entropy vs Scale Consistency scatter...")
    plot_entropy_vs_scale_consistency(df, plots_dir, show)
    
    # 4. Temperature effect lines
    print("\n4. Temperature effect line plots...")
    plot_temperature_effect(df, plots_dir, show)
    
    # 5. Radar charts for each setting
    print("\n5. Radar charts...")
    for setting in SETTING_ORDER:
        plot_model_profiles_radar(df, plots_dir, setting, show)
    
    # 6. Comprehensive heatmap
    print("\n6. Comprehensive heatmap...")
    plot_summary_heatmap(df, plots_dir, show)
    
    # 7. Pairwise metric comparison
    print("\n7. Pairwise metric comparison...")
    plot_pairwise_metric_comparison(df, plots_dir, show)
    
    # 8. Summary statistics table
    print("\n8. Summary statistics...")
    create_summary_statistics_table(df, plots_dir)
    
    print("\n" + "="*60)
    print(f"All plots saved to: {plots_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for tokenization comparison experiment")
    parser.add_argument("--results_dir", type=str, default="experiments/experiment3/results",
                        help="Directory containing experiment results")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    
    args = parser.parse_args()
    generate_all_plots(args.results_dir, args.show)
