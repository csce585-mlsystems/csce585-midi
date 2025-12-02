"""
Enhanced Plotting for Experiment 4: Architecture Comparison
============================================================
Creates publication-quality plots comparing LSTM, GRU, and Transformer architectures.
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

# Color scheme - distinct colors for each architecture
COLORS = {
    'lstm': '#e74c3c',           # Red - baseline
    'gru': '#2ecc71',            # Green - simpler variant
    'transformer': '#3498db'      # Blue - attention-based
}

# Display labels
LABELS = {
    'lstm': 'LSTM',
    'gru': 'GRU',
    'transformer': 'Transformer'
}

FULL_LABELS = {
    'lstm': 'LSTM (Baseline)',
    'gru': 'GRU',
    'transformer': 'Transformer (Small)'
}

# Model parameter counts
PARAMS = {
    'lstm': 941620,
    'gru': 711220,
    'transformer': 2133556
}

SETTING_ORDER = ['conservative', 'balanced', 'creative']
MODEL_ORDER = ['lstm', 'gru', 'transformer']


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


def load_training_losses(results_dir):
    """Load training loss curves."""
    losses_dir = Path(results_dir) / "training_losses"
    losses = {}
    
    for model in MODEL_ORDER:
        loss_file = losses_dir / f"{model}_losses.npy"
        if loss_file.exists():
            losses[model] = np.load(loss_file)
        else:
            print(f"Warning: No loss file found for {model}")
    
    return losses


def load_experiment_results(results_dir):
    """Load the full experiment results JSON."""
    json_file = Path(results_dir) / "experiment_results.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            return json.load(f)
    return None


def plot_training_loss_curves(losses, output_dir, show=False):
    """
    Plot training loss curves for all three architectures.
    This is a key plot showing convergence behavior.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in MODEL_ORDER:
        if model in losses and losses[model] is not None:
            loss_data = losses[model]
            epochs = np.arange(1, len(loss_data) + 1)
            ax.plot(epochs, loss_data, 'o-', color=COLORS[model], 
                   label=f"{FULL_LABELS[model]} ({PARAMS[model]:,} params)",
                   linewidth=2, markersize=6)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('Training Loss Curves by Architecture', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation about transformer behavior
    ax.annotate('Lower loss ≠ better generation\n(potential overfitting)', 
                xy=(0.95, 0.05), xycoords='axes fraction',
                fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "training_loss_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_loss_vs_params(results, output_dir, show=False):
    """
    Scatter plot showing final training loss vs parameter count.
    Illustrates the efficiency of different architectures.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    summaries = results.get('training_summaries', {})
    
    for model in MODEL_ORDER:
        if model in summaries:
            params = PARAMS[model]
            final_loss = summaries[model].get('final_loss', None)
            if final_loss:
                ax.scatter(params / 1e6, final_loss, s=200, c=COLORS[model], 
                          label=FULL_LABELS[model], edgecolor='white', linewidth=2)
                ax.annotate(LABELS[model], (params / 1e6, final_loss), 
                           textcoords="offset points", xytext=(10, 5), fontsize=10)
    
    ax.set_xlabel('Parameters (Millions)', fontweight='bold')
    ax.set_ylabel('Final Training Loss', fontweight='bold')
    ax.set_title('Model Efficiency: Loss vs Parameters', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "loss_vs_params.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_grouped_bar_comparison(df, metric, output_dir, show=False):
    """
    Create a grouped bar chart comparing a metric across all models and settings.
    Each setting group has 3 bars (one per architecture).
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
    ax.set_title(f'{metric_title} by Architecture', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in settings])
    ax.legend(title='Architecture', loc='best', framealpha=0.9)
    
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
    ax.set_title(f'Distribution of {metric_title} Across Architectures', fontsize=14, fontweight='bold')
    ax.set_xticklabels([s.title() for s in SETTING_ORDER])
    ax.legend(title='Architecture', loc='best')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"{metric}_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_architecture_overview(df, results, output_dir, show=False):
    """
    Create a comprehensive overview plot with multiple panels showing
    training loss, parameter count, and key generation metrics.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: Parameter comparison (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    params = [PARAMS[m] / 1e6 for m in MODEL_ORDER]
    bars = ax1.bar(MODEL_ORDER, params, color=[COLORS[m] for m in MODEL_ORDER])
    ax1.set_ylabel('Parameters (Millions)', fontweight='bold')
    ax1.set_title('Model Size', fontweight='bold')
    ax1.set_xticklabels([LABELS[m] for m in MODEL_ORDER])
    
    # Add value labels on bars
    for bar, val in zip(bars, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}M', ha='center', va='bottom', fontsize=10)
    
    # Panel 2: Final training loss comparison
    ax2 = fig.add_subplot(gs[0, 1])
    summaries = results.get('training_summaries', {})
    losses = [summaries.get(m, {}).get('final_loss', 0) for m in MODEL_ORDER]
    bars = ax2.bar(MODEL_ORDER, losses, color=[COLORS[m] for m in MODEL_ORDER])
    ax2.set_ylabel('Final Training Loss', fontweight='bold')
    ax2.set_title('Training Performance', fontweight='bold')
    ax2.set_xticklabels([LABELS[m] for m in MODEL_ORDER])
    
    for bar, val in zip(bars, losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Panel 3: Scale consistency (balanced setting)
    ax3 = fig.add_subplot(gs[0, 2])
    balanced_df = df[df['setting'] == 'balanced']
    scale_means = balanced_df.groupby('model_type')['scale_consistency'].mean()
    scale_stds = balanced_df.groupby('model_type')['scale_consistency'].std()
    
    bars = ax3.bar(MODEL_ORDER, [scale_means.get(m, 0) for m in MODEL_ORDER],
                   yerr=[scale_stds.get(m, 0) for m in MODEL_ORDER],
                   color=[COLORS[m] for m in MODEL_ORDER], capsize=5)
    ax3.set_ylabel('Scale Consistency', fontweight='bold')
    ax3.set_title('Musical Quality (Balanced)', fontweight='bold')
    ax3.set_xticklabels([LABELS[m] for m in MODEL_ORDER])
    ax3.set_ylim(0, 1.1)
    
    # Panel 4: Pitch entropy (balanced setting)
    ax4 = fig.add_subplot(gs[1, 0])
    entropy_means = balanced_df.groupby('model_type')['pitch_entropy'].mean()
    entropy_stds = balanced_df.groupby('model_type')['pitch_entropy'].std()
    
    bars = ax4.bar(MODEL_ORDER, [entropy_means.get(m, 0) for m in MODEL_ORDER],
                   yerr=[entropy_stds.get(m, 0) for m in MODEL_ORDER],
                   color=[COLORS[m] for m in MODEL_ORDER], capsize=5)
    ax4.set_ylabel('Pitch Entropy', fontweight='bold')
    ax4.set_title('Pitch Diversity (Balanced)', fontweight='bold')
    ax4.set_xticklabels([LABELS[m] for m in MODEL_ORDER])
    
    # Panel 5: Note density (balanced setting)
    ax5 = fig.add_subplot(gs[1, 1])
    density_means = balanced_df.groupby('model_type')['note_density'].mean()
    density_stds = balanced_df.groupby('model_type')['note_density'].std()
    
    bars = ax5.bar(MODEL_ORDER, [density_means.get(m, 0) for m in MODEL_ORDER],
                   yerr=[density_stds.get(m, 0) for m in MODEL_ORDER],
                   color=[COLORS[m] for m in MODEL_ORDER], capsize=5)
    ax5.set_ylabel('Notes per Second', fontweight='bold')
    ax5.set_title('Note Density (Balanced)', fontweight='bold')
    ax5.set_xticklabels([LABELS[m] for m in MODEL_ORDER])
    
    # Panel 6: Summary radar chart
    ax6 = fig.add_subplot(gs[1, 2], projection='polar')
    metrics = ['scale_consistency', 'pitch_entropy', 'note_density', 'pitch_range']
    
    # Normalize metrics for radar
    balanced_stats = {}
    for m in MODEL_ORDER:
        model_data = balanced_df[balanced_df['model_type'] == m]
        balanced_stats[m] = [model_data[metric].mean() for metric in metrics]
    
    # Normalize each metric to 0-1
    for i in range(len(metrics)):
        values = [balanced_stats[m][i] for m in MODEL_ORDER]
        min_val, max_val = min(values), max(values)
        if max_val > min_val:
            for m in MODEL_ORDER:
                balanced_stats[m][i] = (balanced_stats[m][i] - min_val) / (max_val - min_val)
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    for model in MODEL_ORDER:
        values = balanced_stats[model] + [balanced_stats[model][0]]
        ax6.plot(angles, values, 'o-', color=COLORS[model], 
                label=LABELS[model], linewidth=2, markersize=4)
        ax6.fill(angles, values, color=COLORS[model], alpha=0.1)
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels([m.replace('_', '\n').title() for m in metrics], size=8)
    ax6.set_title('Overall Profile', fontweight='bold', y=1.1)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    fig.suptitle('Experiment 4: Architecture Comparison Overview', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "architecture_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_temperature_effect(df, output_dir, show=False):
    """
    Line plot showing how temperature (via settings) affects key metrics for each architecture.
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
    
    fig.suptitle('Effect of Temperature on Generation Quality by Architecture', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "temperature_effect_lines.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_entropy_vs_scale_consistency(df, output_dir, show=False):
    """
    Scatter plot showing the relationship between pitch entropy and scale consistency.
    This reveals how each architecture balances diversity and musicality.
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
    
    legend1 = ax.legend(handles=model_handles, title='Architecture', loc='lower left')
    ax.add_artist(legend1)
    ax.legend(handles=setting_handles, title='Setting', loc='lower right')
    
    ax.set_xlabel('Pitch Entropy (higher = more diverse)', fontweight='bold')
    ax.set_ylabel('Scale Consistency (higher = more tonal)', fontweight='bold')
    ax.set_title('Diversity vs. Musicality by Architecture', fontsize=14, fontweight='bold')
    
    # Add reference line
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Good tonality threshold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "entropy_vs_scale_scatter.png"
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
    
    ax.set_title('Comprehensive Metric Comparison by Architecture\n(Values shown, colors normalized by column)', 
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


def create_summary_statistics_table(df, results, output_dir):
    """
    Create a summary statistics table as CSV and formatted markdown.
    """
    metrics = ['scale_consistency', 'pitch_entropy', 'pitch_class_entropy',
               'note_density', 'pitch_range', 'avg_polyphony']
    
    # Calculate statistics
    summary = df.groupby(['model_type', 'setting'])[metrics].agg(['mean', 'std', 'count'])
    
    # Save to CSV
    csv_path = Path(output_dir) / "summary_statistics.csv"
    summary.to_csv(csv_path)
    print(f"Saved: {csv_path}")
    
    # Create a formatted markdown table
    md_path = Path(output_dir) / "summary_statistics.md"
    with open(md_path, 'w') as f:
        f.write("# Experiment 4: Architecture Comparison - Summary Statistics\n\n")
        
        # Training summary section
        f.write("## Training Performance\n\n")
        f.write("| Architecture | Parameters | Final Loss | Training Time |\n")
        f.write("|--------------|------------|------------|---------------|\n")
        
        summaries = results.get('training_summaries', {})
        for model in MODEL_ORDER:
            if model in summaries:
                s = summaries[model]
                params = f"{PARAMS[model]:,}"
                loss = f"{s.get('final_loss', 'N/A'):.4f}" if isinstance(s.get('final_loss'), (int, float)) else 'N/A'
                time = s.get('training_time', 'N/A')
                f.write(f"| {FULL_LABELS[model]} | {params} | {loss} | {time} |\n")
        
        f.write("\n## Generation Metrics\n\n")
        
        for model in MODEL_ORDER:
            f.write(f"\n### {FULL_LABELS[model]}\n\n")
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
        
        f.write("\n## Key Findings\n\n")
        f.write("- **Parameter Efficiency**: Compare how each architecture uses its parameters\n")
        f.write("- **Training Loss**: Lower training loss doesn't always mean better generation\n")
        f.write("- **Scale Consistency**: Higher values indicate more musically coherent output\n")
        f.write("- **Pitch Entropy**: Balance between diversity and repetition\n")
    
    print(f"Saved: {md_path}")


def generate_all_plots(results_dir, show=False):
    """Generate all plots for the experiment."""
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading results from: {results_dir}")
    
    # Load data
    try:
        df = load_results(results_dir)
        print(f"Loaded {len(df)} samples")
        print(f"Models: {list(df['model_type'].unique())}")
        print(f"Settings: {list(df['setting'].unique())}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        df = None
    
    losses = load_training_losses(results_dir)
    results = load_experiment_results(results_dir)
    
    # Generate all plots
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    # 1. Training loss curves (always possible if we have loss files)
    print("\n1. Training loss curves...")
    if losses:
        plot_training_loss_curves(losses, plots_dir, show)
    
    # 2. Loss vs parameters
    if results:
        print("\n2. Loss vs parameters...")
        plot_loss_vs_params(results, plots_dir, show)
    
    if df is not None:
        # 3. Grouped bar charts for key metrics
        print("\n3. Grouped bar charts...")
        for metric in ['scale_consistency', 'pitch_entropy', 'pitch_class_entropy', 
                       'note_density', 'pitch_range', 'avg_polyphony']:
            plot_grouped_bar_comparison(df, metric, plots_dir, show)
        
        # 4. Box plots
        print("\n4. Box plots...")
        for metric in ['scale_consistency', 'pitch_entropy']:
            plot_model_comparison_boxplot(df, metric, plots_dir, show)
        
        # 5. Architecture overview
        if results:
            print("\n5. Architecture overview...")
            plot_architecture_overview(df, results, plots_dir, show)
        
        # 6. Temperature effect lines
        print("\n6. Temperature effect line plots...")
        plot_temperature_effect(df, plots_dir, show)
        
        # 7. Entropy vs Scale Consistency scatter
        print("\n7. Entropy vs Scale Consistency scatter...")
        plot_entropy_vs_scale_consistency(df, plots_dir, show)
        
        # 8. Comprehensive heatmap
        print("\n8. Comprehensive heatmap...")
        plot_summary_heatmap(df, plots_dir, show)
        
        # 9. Summary statistics table
        if results:
            print("\n9. Summary statistics...")
            create_summary_statistics_table(df, results, plots_dir)
    
    print("\n" + "="*60)
    print(f"All plots saved to: {plots_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for architecture comparison experiment")
    parser.add_argument("--results_dir", type=str, default="experiments/experiment4/results",
                        help="Directory containing experiment results")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    
    args = parser.parse_args()
    generate_all_plots(args.results_dir, args.show)
