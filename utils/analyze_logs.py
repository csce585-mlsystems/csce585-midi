"""
Comprehensive log analysis tool for final presentation.
Combines all training, generation, and evaluation logs to provide insights.

Usage:
    python analyze_logs.py --output analysis_report.md
    python analyze_logs.py --plot  # Creates visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class LogAnalyzer:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.generator_logs = pd.DataFrame()
        self.discriminator_logs = pd.DataFrame()
        self.generation_logs = pd.DataFrame()
        self.evaluation_logs = pd.DataFrame()
        
    def load_all_logs(self):
        """Load all log files into dataframes."""
        print("Loading logs...")
        
        # Load generator training logs
        generator_model_files = list(self.logs_dir.glob("generators/*/models/models.csv"))
        if generator_model_files:
            dfs = []
            for file in generator_model_files:
                try:
                    df = pd.read_csv(file, on_bad_lines='skip')
                    df['dataset_name'] = file.parent.parent.name
                    dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
            self.generator_logs = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            print(f"Loaded {len(self.generator_logs)} generator training runs")
        
        # Load generation output logs
        generation_files = list(self.logs_dir.glob("generators/*/midi/output_midis.csv"))
        if generation_files:
            dfs = []
            for file in generation_files:
                try:
                    df = pd.read_csv(file, on_bad_lines='skip')
                    df['dataset_name'] = file.parent.parent.name
                    dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
            self.generation_logs = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            print(f"Loaded {len(self.generation_logs)} generation runs")
        
        # Load evaluation logs
        evaluation_files = list(self.logs_dir.glob("generators/*/midi/evaluation_log.csv"))
        if evaluation_files:
            dfs = []
            for file in evaluation_files:
                try:
                    df = pd.read_csv(file, on_bad_lines='skip')
                    df['dataset_name'] = file.parent.parent.name
                    dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
            self.evaluation_logs = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            print(f"Loaded {len(self.evaluation_logs)} evaluations")
        
        # Load discriminator logs
        disc_file = self.logs_dir / "discriminators" / "train_summary.csv"
        if disc_file.exists():
            self.discriminator_logs = pd.read_csv(disc_file)
            print(f"Loaded {len(self.discriminator_logs)} discriminator training epochs")
        
        print("Loading complete!\n")
    
    def analyze_training_performance(self):
        """Analyze generator training performance."""
        if self.generator_logs.empty:
            return "No generator training logs found."
        
        df = self.generator_logs
        
        report = []
        report.append("=" * 70)
        report.append("GENERATOR TRAINING PERFORMANCE")
        report.append("=" * 70)
        report.append("")
        
        # Overall statistics
        report.append(f"Total models trained: {len(df)}")
        report.append(f"Datasets used: {df['dataset_name'].nunique()}")
        report.append(f"Model types: {', '.join(str(x) for x in df['model_type'].unique())}")
        report.append("")
        
        # Performance by model type
        report.append("Performance by Model Type:")
        report.append("-" * 70)
        for model_type in df['model_type'].unique():
            model_df = df[df['model_type'] == model_type]
            model_name = str(model_type).upper() if model_type else "UNKNOWN"
            report.append(f"\n{model_name}:")
            report.append(f"  Models trained: {len(model_df)}")
            report.append(f"  Avg final loss: {model_df['final_loss'].mean():.4f}")
            report.append(f"  Min loss achieved: {model_df['min_loss'].min():.4f}")
            report.append(f"  Avg training time: {model_df['train_time_sec'].mean():.1f}s")
            report.append(f"  Avg parameters: {model_df['num_params'].mean():,.0f}")
        
        report.append("")
        
        # Performance by dataset
        report.append("Performance by Dataset:")
        report.append("-" * 70)
        for dataset in df['dataset_name'].unique():
            dataset_df = df[df['dataset_name'] == dataset]
            report.append(f"\n{dataset}:")
            report.append(f"  Models trained: {len(dataset_df)}")
            report.append(f"  Avg final loss: {dataset_df['final_loss'].mean():.4f}")
            report.append(f"  Best model: {dataset_df.loc[dataset_df['min_loss'].idxmin(), 'model_type']}")
            report.append(f"  Best loss: {dataset_df['min_loss'].min():.4f}")
        
        report.append("")
        
        # Best models overall
        report.append("Top 5 Best Models (by min_loss):")
        report.append("-" * 70)
        best_models = df.nsmallest(5, 'min_loss')
        for idx, row in best_models.iterrows():
            report.append(f"  {row['model_type']:12s} | {row['dataset_name']:30s} | Loss: {row['min_loss']:.4f} | Params: {row['num_params']:,}")
        
        report.append("")
        return "\n".join(report)
    
    def analyze_generation_quality(self):
        """Analyze generated MIDI quality."""
        if self.generation_logs.empty or self.evaluation_logs.empty:
            return "No generation or evaluation logs found."
        
        # Merge generation and evaluation logs
        merged = pd.merge(
            self.generation_logs, 
            self.evaluation_logs, 
            left_on='output_file', 
            right_on='output_file',
            suffixes=('_gen', '_eval')
        )
        
        report = []
        report.append("=" * 70)
        report.append("GENERATION QUALITY ANALYSIS")
        report.append("=" * 70)
        report.append("")
        
        report.append(f"Total generations analyzed: {len(merged)}")
        report.append("")
        
        # Quality by sampling strategy
        report.append("Quality by Sampling Strategy:")
        report.append("-" * 70)
        for strategy in merged['strategy'].unique():
            strategy_df = merged[merged['strategy'] == strategy]
            report.append(f"\n{strategy.upper()}:")
            report.append(f"  Samples: {len(strategy_df)}")
            report.append(f"  Avg note density: {strategy_df['note_density'].mean():.2f} notes/sec")
            report.append(f"  Avg pitch range: {strategy_df['pitch_range'].mean():.1f} semitones")
            report.append(f"  Avg duration: {strategy_df['duration'].mean():.1f}s")
            report.append(f"  Avg polyphony: {strategy_df['avg_polyphony'].mean():.1f}")
        
        report.append("")
        
        # Temperature effects
        if 'temperature' in merged.columns:
            report.append("Temperature Effects:")
            report.append("-" * 70)
            temp_groups = merged.groupby(pd.cut(merged['temperature'], bins=[0, 1.0, 1.5, 2.0, 3.0]))
            for temp_range, group in temp_groups:
                if len(group) > 0:
                    report.append(f"\nTemperature {temp_range}:")
                    report.append(f"  Samples: {len(group)}")
                    report.append(f"  Avg pitch range: {group['pitch_range'].mean():.1f}")
                    report.append(f"  Avg note density: {group['note_density'].mean():.2f}")
        
        report.append("")
        
        # Dataset comparison
        if 'dataset_name_gen' in merged.columns:
            report.append("Quality by Dataset:")
            report.append("-" * 70)
            for dataset in merged['dataset_name_gen'].unique():
                dataset_df = merged[merged['dataset_name_gen'] == dataset]
                report.append(f"\n{dataset}:")
                report.append(f"  Generations: {len(dataset_df)}")
                report.append(f"  Avg pitch range: {dataset_df['pitch_range'].mean():.1f}")
                report.append(f"  Avg duration: {dataset_df['duration'].mean():.1f}s")
        
        report.append("")
        return "\n".join(report)
    
    def analyze_discriminator_performance(self):
        """Analyze discriminator training."""
        if self.discriminator_logs.empty:
            return "No discriminator logs found."
        
        df = self.discriminator_logs
        
        report = []
        report.append("=" * 70)
        report.append("DISCRIMINATOR PERFORMANCE")
        report.append("=" * 70)
        report.append("")
        
        # Performance by model type
        report.append("Performance by Model Type:")
        report.append("-" * 70)
        for model_type in df['model_type'].unique():
            model_df = df[df['model_type'] == model_type]
            best_epoch = model_df.loc[model_df['micro_f1'].idxmax()]
            report.append(f"\n{model_type.upper()}:")
            report.append(f"  Epochs trained: {model_df['epoch'].max()}")
            report.append(f"  Best F1 score: {best_epoch['micro_f1']:.4f}")
            report.append(f"  Best precision: {best_epoch['micro_precision']:.4f}")
            report.append(f"  Best recall: {best_epoch['micro_recall']:.4f}")
            report.append(f"  Final loss: {model_df.iloc[-1]['train_loss']:.4f}")
        
        report.append("")
        
        # Context size effects
        if 'context' in df.columns:
            report.append("Context Size Effects:")
            report.append("-" * 70)
            for context in sorted(df['context'].unique()):
                context_df = df[df['context'] == context]
                best = context_df.loc[context_df['micro_f1'].idxmax()]
                report.append(f"\nContext: {context} measures")
                report.append(f"  Best F1: {best['micro_f1']:.4f}")
                report.append(f"  Model: {best['model_type']}")
        
        report.append("")
        return "\n".join(report)
    
    def generate_insights(self):
        """Generate key insights for presentation."""
        insights = []
        insights.append("=" * 70)
        insights.append("KEY INSIGHTS FOR PRESENTATION")
        insights.append("=" * 70)
        insights.append("")
        
        # Generator insights
        if not self.generator_logs.empty:
            df = self.generator_logs
            best_model = df.loc[df['min_loss'].idxmin()]
            insights.append("TRAINING FINDINGS:")
            insights.append(f"  Best performing model: {best_model['model_type'].upper()}")
            insights.append(f"  Achieved loss: {best_model['min_loss']:.4f}")
            insights.append(f"  Dataset: {best_model['dataset_name']}")
            insights.append(f"  Total training time: {df['train_time_sec'].sum()/60:.1f} minutes")
            
            # Model comparison
            lstm_avg = df[df['model_type'] == 'lstm']['min_loss'].mean()
            transformer_avg = df[df['model_type'] == 'transformer']['min_loss'].mean()
            if not np.isnan(lstm_avg) and not np.isnan(transformer_avg):
                better = "LSTM" if lstm_avg < transformer_avg else "Transformer"
                diff = abs(lstm_avg - transformer_avg)
                insights.append(f"  {better} performed {diff:.4f} better on average")
            insights.append("")
        
        # Generation insights
        if not self.generation_logs.empty and not self.evaluation_logs.empty:
            merged = pd.merge(self.generation_logs, self.evaluation_logs, on='output_file', how='inner')
            insights.append("GENERATION FINDINGS:")
            insights.append(f"  • Total generations: {len(merged)}")
            insights.append(f"  • Avg pitch range: {merged['pitch_range'].mean():.1f} semitones")
            insights.append(f"  • Avg note density: {merged['note_density'].mean():.2f} notes/sec")
            
            # Strategy comparison
            if 'strategy' in merged.columns and len(merged) > 1:
                best_strategy = merged.groupby('strategy')['pitch_range'].mean().idxmax()
                insights.append(f"  Most diverse strategy: {best_strategy}")
            insights.append("")
        
        # Discriminator insights
        if not self.discriminator_logs.empty:
            df = self.discriminator_logs
            best = df.loc[df['micro_f1'].idxmax()]
            insights.append("DISCRIMINATOR FINDINGS:")
            insights.append(f"  Best F1 score: {best['micro_f1']:.4f}")
            insights.append(f"  Best model: {best['model_type'].upper()}")
            insights.append(f"  Training epochs: {df['epoch'].max()}")
            insights.append("")
        
        insights.append("RECOMMENDATIONS:")
        if not self.generator_logs.empty:
            best_type = self.generator_logs.groupby('model_type')['min_loss'].mean().idxmin()
            insights.append(f"  Use {best_type.upper()} architecture for best results")
        if not self.generation_logs.empty:
            insights.append("  Experiment with temperature 1.5-2.0 for diversity")
            insights.append("  Use nucleus sampling (top_p) for musical coherence")
        
        insights.append("")
        return "\n".join(insights)
    
    def create_visualizations(self, output_dir="presentation/figures"):
        """Create visualizations for presentation."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating visualizations in {output_dir}...")
        
        # 1. Training loss comparison
        if not self.generator_logs.empty:
            plt.figure(figsize=(10, 6))
            for model_type in self.generator_logs['model_type'].unique():
                model_df = self.generator_logs[self.generator_logs['model_type'] == model_type]
                model_name = str(model_type).upper() if model_type else "UNKNOWN"
                plt.scatter(range(len(model_df)), model_df['final_loss'], 
                           label=model_name, alpha=0.6, s=100)
            plt.xlabel('Training Run', fontsize=12)
            plt.ylabel('Final Loss', fontsize=12)
            plt.title('Generator Training Performance by Architecture', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'training_loss_comparison.png', dpi=300)
            print(f"Saved training_loss_comparison.png")
            plt.close()
        
        # 2. Model size vs performance
        if not self.generator_logs.empty and 'num_params' in self.generator_logs.columns:
            plt.figure(figsize=(10, 6))
            for model_type in self.generator_logs['model_type'].unique():
                model_df = self.generator_logs[self.generator_logs['model_type'] == model_type]
                model_name = str(model_type).upper() if model_type else "UNKNOWN"
                plt.scatter(model_df['num_params'], model_df['min_loss'], 
                           label=model_name, alpha=0.6, s=100)
            plt.xlabel('Number of Parameters', fontsize=12)
            plt.ylabel('Minimum Loss', fontsize=12)
            plt.title('Model Size vs Performance', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.xscale('log')
            plt.tight_layout()
            plt.savefig(output_dir / 'model_size_vs_performance.png', dpi=300)
            print(f"Saved model_size_vs_performance.png")
            plt.close()
        
        # 3. Generation quality metrics
        if not self.evaluation_logs.empty:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Note density
            axes[0, 0].hist(self.evaluation_logs['note_density'], bins=20, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Note Density (notes/sec)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Note Density')
            
            # Pitch range
            axes[0, 1].hist(self.evaluation_logs['pitch_range'], bins=20, color='lightcoral', edgecolor='black')
            axes[0, 1].set_xlabel('Pitch Range (semitones)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Pitch Range')
            
            # Duration
            axes[1, 0].hist(self.evaluation_logs['duration'], bins=20, color='lightgreen', edgecolor='black')
            axes[1, 0].set_xlabel('Duration (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Duration')
            
            # Polyphony
            axes[1, 1].hist(self.evaluation_logs['avg_polyphony'], bins=20, color='plum', edgecolor='black')
            axes[1, 1].set_xlabel('Average Polyphony')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Polyphony')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'generation_quality_metrics.png', dpi=300)
            print(f"Saved generation_quality_metrics.png")
            plt.close()
        
        # 4. Discriminator learning curves
        if not self.discriminator_logs.empty:
            plt.figure(figsize=(12, 6))
            for model_type in self.discriminator_logs['model_type'].unique():
                model_df = self.discriminator_logs[self.discriminator_logs['model_type'] == model_type]
                model_name = str(model_type).upper() if model_type else "UNKNOWN"
                plt.plot(model_df['epoch'], model_df['micro_f1'], 
                        marker='o', label=f'{model_name} F1', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Micro F1 Score', fontsize=12)
            plt.title('Discriminator Learning Curves', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'discriminator_learning_curves.png', dpi=300)
            print(f"  ✓ Saved discriminator_learning_curves.png")
            plt.close()
        
        # 5. Dataset comparison
        if not self.generator_logs.empty and 'dataset_name' in self.generator_logs.columns:
            dataset_performance = self.generator_logs.groupby('dataset_name')['min_loss'].mean().sort_values()
            
            plt.figure(figsize=(12, 6))
            plt.barh(range(len(dataset_performance)), dataset_performance.values, color='steelblue')
            plt.yticks(range(len(dataset_performance)), dataset_performance.index)
            plt.xlabel('Average Minimum Loss', fontsize=12)
            plt.title('Performance by Dataset', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'dataset_comparison.png', dpi=300)
            print(f"Saved dataset_comparison.png")
            plt.close()
        
        print(f"\nAll visualizations saved to {output_dir}/")
    
    def export_summary_tables(self, output_dir="presentation/tables"):
        """Export summary tables as CSV for easy inclusion in presentation."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Exporting summary tables to {output_dir}...")
        
        # Generator summary
        if not self.generator_logs.empty:
            summary = self.generator_logs.groupby('model_type').agg({
                'final_loss': ['mean', 'std', 'min'],
                'num_params': 'mean',
                'train_time_sec': 'mean'
            }).round(4)
            summary.to_csv(output_dir / 'generator_summary.csv')
            print(f"  ✓ Saved generator_summary.csv")
        
        # Generation quality summary
        if not self.evaluation_logs.empty:
            summary = self.evaluation_logs.agg({
                'num_notes': ['mean', 'std'],
                'duration': ['mean', 'std'],
                'note_density': ['mean', 'std'],
                'pitch_range': ['mean', 'std'],
                'avg_polyphony': ['mean', 'std']
            }).round(2)
            summary.to_csv(output_dir / 'generation_quality_summary.csv')
            print(f"  ✓ Saved generation_quality_summary.csv")
        
        # Discriminator summary
        if not self.discriminator_logs.empty:
            summary = self.discriminator_logs.groupby('model_type').agg({
                'micro_f1': 'max',
                'micro_precision': 'max',
                'micro_recall': 'max',
                'train_loss': 'min'
            }).round(4)
            summary.to_csv(output_dir / 'discriminator_summary.csv')
            print(f"Saved discriminator_summary.csv")
        
        print(f"\nAll tables saved to {output_dir}/")
    
    def generate_full_report(self, output_file="presentation/analysis_report.md"):
        """Generate a comprehensive markdown report."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("# MIDI Generation Project - Comprehensive Analysis Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n")
        
        report.append(self.analyze_training_performance())
        report.append("\n\n")
        report.append(self.analyze_generation_quality())
        report.append("\n\n")
        report.append(self.analyze_discriminator_performance())
        report.append("\n\n")
        report.append(self.generate_insights())
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nFull report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training and generation logs")
    parser.add_argument("output_dir", nargs="?", default="presentation/analysis",
                        help="Directory to store all analysis outputs")
    parser.add_argument("--plot", action="store_true",
                        help="Generate visualization plots")
    parser.add_argument("--tables", action="store_true",
                        help="Export summary tables")
    parser.add_argument("--report", action="store_true",
                        help="Generate markdown report")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If no specific action flags are set, do everything
    do_all = not (args.plot or args.tables or args.report)
    
    # Initialize analyzer
    analyzer = LogAnalyzer()
    analyzer.load_all_logs()
    
    # Generate outputs based on arguments
    if do_all or args.report:
        analyzer.generate_full_report(output_dir / "analysis_report.md")
    
    if do_all or args.plot:
        analyzer.create_visualizations(output_dir / "figures")
    
    if do_all or args.tables:
        analyzer.export_summary_tables(output_dir / "tables")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Outputs saved to: {output_dir}")
    print("\nNext steps for your presentation:")
    print("  1. Review the markdown report for key findings")
    print("  2. Use the generated figures in your slides")
    print("  3. Reference the summary tables for specific metrics")
    print("  4. Highlight the key insights section")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
