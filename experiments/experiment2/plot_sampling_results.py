"""
Plot Sampling Strategy Experiment Results
==========================================

Creates visualizations comparing different sampling strategies:
- Box plots for each metric
- Bar charts with error bars
- Radar chart for overall comparison
- Distribution plots
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_results(results_path):
    """Load experiment results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_box_plots(results, output_dir):
    """Create box plots comparing distributions across strategies."""
    
    strategies = list(results["strategies"].keys())
    
    metrics = [
        ("note_density", "Note Density", "Notes per Second"),
        ("pitch_range", "Pitch Range", "Semitones"),
        ("avg_polyphony", "Average Polyphony", "Simultaneous Notes"),
        ("scale_consistency", "Scale Consistency", "Proportion"),
        ("pitch_entropy", "Pitch Entropy", "Bits"),
        ("pitch_class_entropy", "Pitch Class Entropy", "Bits")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (metric_key, title, ylabel) in enumerate(metrics):
        ax = axes[idx]
        
        data = []
        labels = []
        for strategy in strategies:
            values = [
                r[metric_key] 
                for r in results["strategies"][strategy]["results"] 
                if metric_key in r and r[metric_key] is not None
            ]
            if values:
                data.append(values)
                # Shorten label for display
                label = strategy.replace("_", "\n")
                labels.append(label)
        
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle("Sampling Strategy Comparison - Distribution Analysis", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "box_plots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_bar_charts(results, output_dir):
    """Create bar charts with error bars for each metric."""
    
    strategies = list(results["strategies"].keys())
    
    # Clean up strategy names for display
    display_names = {
        "greedy": "Greedy",
        "top_k_5": "Top-k\n(k=5)",
        "top_k_10": "Top-k\n(k=10)",
        "top_p_0.9": "Top-p\n(p=0.9)",
        "top_p_0.95": "Top-p\n(p=0.95)",
        "temp_0.5": "Temp\n(T=0.5)",
        "temp_1.0": "Temp\n(T=1.0)",
        "temp_1.5": "Temp\n(T=1.5)"
    }
    
    metrics = [
        ("note_density", "Note Density (notes/sec)"),
        ("pitch_range", "Pitch Range (semitones)"),
        ("avg_polyphony", "Average Polyphony"),
        ("scale_consistency", "Scale Consistency"),
        ("pitch_entropy", "Pitch Entropy (bits)"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    x = np.arange(len(strategies))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(strategies)))
    
    for idx, (metric_key, title) in enumerate(metrics):
        ax = axes[idx]
        
        means = []
        stds = []
        for strategy in strategies:
            summary = results["strategies"][strategy]["summary"]
            means.append(summary.get(f"mean_{metric_key}", 0))
            stds.append(summary.get(f"std_{metric_key}", 0))
        
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, 
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([display_names.get(s, s) for s in strategies], 
                          fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel("Value")
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{mean:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
    
    # Hide last subplot if odd number of metrics
    if len(metrics) < len(axes):
        axes[-1].set_visible(False)
    
    plt.suptitle("Sampling Strategy Comparison - Mean ± Std Dev", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "bar_charts.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_radar_chart(results, output_dir):
    """Create radar chart comparing strategies across all metrics."""
    
    strategies = list(results["strategies"].keys())
    
    # Metrics to include (normalized to 0-1 scale for radar chart)
    metrics = [
        "note_density",
        "pitch_range", 
        "avg_polyphony",
        "scale_consistency",
        "pitch_entropy"
    ]
    
    metric_labels = [
        "Note Density",
        "Pitch Range",
        "Polyphony",
        "Scale Consistency",
        "Pitch Entropy"
    ]
    
    # Collect data and normalize
    data = {}
    for strategy in strategies:
        summary = results["strategies"][strategy]["summary"]
        data[strategy] = [summary.get(f"mean_{m}", 0) for m in metrics]
    
    # Normalize each metric to 0-1 range
    normalized_data = {}
    for i, metric in enumerate(metrics):
        values = [data[s][i] for s in strategies]
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        for strategy in strategies:
            if strategy not in normalized_data:
                normalized_data[strategy] = []
            normalized_data[strategy].append(
                (data[strategy][i] - min_val) / range_val
            )
    
    # Create radar chart
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    for idx, strategy in enumerate(strategies):
        values = normalized_data[strategy]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=strategy, 
               color=colors[idx], alpha=0.8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_title("Sampling Strategy Comparison\n(Normalized 0-1 Scale)", 
                fontsize=14, fontweight='bold', pad=20)
    
    # Position legend outside
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    output_path = output_dir / "radar_chart.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_diversity_plot(results, output_dir):
    """Create a focused plot on diversity metrics (key for this experiment)."""
    
    strategies = list(results["strategies"].keys())
    
    display_names = {
        "greedy": "Greedy",
        "top_k_5": "Top-k (k=5)",
        "top_k_10": "Top-k (k=10)",
        "top_p_0.9": "Top-p (p=0.9)",
        "top_p_0.95": "Top-p (p=0.95)",
        "temp_0.5": "Temp (T=0.5)",
        "temp_1.0": "Temp (T=1.0)",
        "temp_1.5": "Temp (T=1.5)"
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Pitch Entropy vs Scale Consistency (diversity vs quality tradeoff)
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    for idx, strategy in enumerate(strategies):
        summary = results["strategies"][strategy]["summary"]
        entropy = summary.get("mean_pitch_entropy", 0)
        consistency = summary.get("mean_scale_consistency", 0)
        entropy_std = summary.get("std_pitch_entropy", 0)
        consistency_std = summary.get("std_scale_consistency", 0)
        
        ax1.errorbar(entropy, consistency, 
                    xerr=entropy_std, yerr=consistency_std,
                    fmt='o', markersize=12, capsize=4,
                    color=colors[idx], label=display_names.get(strategy, strategy),
                    alpha=0.8)
    
    ax1.set_xlabel("Pitch Entropy (Diversity)", fontsize=12)
    ax1.set_ylabel("Scale Consistency (Quality)", fontsize=12)
    ax1.set_title("Diversity vs Quality Tradeoff", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Pitch Range comparison
    ax2 = axes[1]
    
    x = np.arange(len(strategies))
    pitch_ranges = [
        results["strategies"][s]["summary"].get("mean_pitch_range", 0)
        for s in strategies
    ]
    pitch_range_stds = [
        results["strategies"][s]["summary"].get("std_pitch_range", 0)
        for s in strategies
    ]
    
    bars = ax2.bar(x, pitch_ranges, yerr=pitch_range_stds, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([display_names.get(s, s) for s in strategies],
                        rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel("Pitch Range (semitones)", fontsize=12)
    ax2.set_title("Pitch Range by Strategy", fontsize=13, fontweight='bold')
    
    plt.suptitle("Sampling Strategy - Musical Diversity Analysis", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "diversity_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_heatmap(results, output_dir):
    """Create a heatmap summary of all metrics."""
    
    strategies = list(results["strategies"].keys())
    metrics = [
        ("mean_note_density", "Note Density"),
        ("mean_pitch_range", "Pitch Range"),
        ("mean_avg_polyphony", "Polyphony"),
        ("mean_scale_consistency", "Scale Consistency"),
        ("mean_pitch_entropy", "Pitch Entropy"),
    ]
    
    # Build data matrix
    data_matrix = []
    for strategy in strategies:
        row = []
        summary = results["strategies"][strategy]["summary"]
        for metric_key, _ in metrics:
            row.append(summary.get(metric_key, 0))
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Normalize columns to 0-1
    normalized = np.zeros_like(data_matrix)
    for j in range(data_matrix.shape[1]):
        col = data_matrix[:, j]
        min_val, max_val = col.min(), col.max()
        if max_val != min_val:
            normalized[:, j] = (col - min_val) / (max_val - min_val)
        else:
            normalized[:, j] = 0.5
    
    # Clean up strategy names for display
    display_names = [s.replace("_", " ").replace(".", "") for s in strategies]
    metric_names = [m[1] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(normalized, cmap='YlGnBu', aspect='auto')
    
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(display_names)))
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_yticklabels(display_names, fontsize=10)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(strategies)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f"{data_matrix[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title("Sampling Strategy Comparison\n(Cell values are raw metrics, colors show relative ranking)",
                fontsize=13, fontweight='bold')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Relative Scale (0=lowest, 1=highest)", rotation=-90, va="bottom")
    
    plt.tight_layout()
    
    output_path = output_dir / "summary_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_all_plots(results_path, output_dir=None):
    """Create all visualization plots."""
    
    results = load_results(results_path)
    
    if output_dir is None:
        output_dir = Path(results_path).parent / "plots"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating plots in {output_dir}")
    print("-" * 50)
    
    create_box_plots(results, output_dir)
    create_bar_charts(results, output_dir)
    create_radar_chart(results, output_dir)
    create_diversity_plot(results, output_dir)
    create_summary_heatmap(results, output_dir)
    
    print("-" * 50)
    print("All plots created successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots for sampling experiment results"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="experiments/experiment2/results/experiment_results.json",
        help="Path to experiment results JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: results_dir/plots)"
    )
    
    args = parser.parse_args()
    create_all_plots(args.results_path, args.output_dir)
