import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def create_model_comparison():
    """Create clean model comparison plots for presentation"""
    
    # Model data based on your analysis
    models = ['Naive', 'MidiTok']
    avg_notes = [392, 73]
    min_notes = [18, 3] 
    max_notes = [1441, 278]
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MIDI Generation: Naive vs MidiTok Tokenization', fontsize=16, fontweight='bold')
    
    colors = ['#2E8B57', '#CD5C5C']  # Sea green, Indian red
    
    # Average notes comparison
    bars1 = ax1.bar(models, avg_notes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_title('Average Notes Generated', fontweight='bold')
    ax1.set_ylabel('Number of Notes')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, avg_notes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(value), ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Min notes comparison
    bars2 = ax2.bar(models, min_notes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_title('Minimum Notes Generated', fontweight='bold')
    ax2.set_ylabel('Number of Notes')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars2, min_notes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(value), ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Max notes comparison
    bars3 = ax3.bar(models, max_notes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_title('Maximum Notes Generated', fontweight='bold')
    ax3.set_ylabel('Number of Notes')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars3, max_notes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 25,
                str(value), ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Performance ratio
    ratios = [avg_notes[0]/avg_notes[1], min_notes[0]/min_notes[1], max_notes[0]/max_notes[1]]
    metrics = ['Avg Notes', 'Min Notes', 'Max Notes']
    
    bars4 = ax4.bar(metrics, ratios, color='#4169E1', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_title('Naive vs MidiTok Performance Ratio', fontweight='bold')
    ax4.set_ylabel('Ratio (Naive/MidiTok)')
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
    ax4.legend()
    
    for bar, value in zip(bars4, ratios):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path('../outputs/model_comparison_presentation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Model comparison saved to: {output_path}")
    
    return output_path

def create_summary_table():
    """Create a summary statistics table"""
    
    data = {
        'Metric': ['Average Notes', 'Minimum Notes', 'Maximum Notes', 'Performance Ratio'],
        'Naive': ['392', '18', '1441', 'Baseline'],
        'MidiTok': ['73', '3', '278', '5.4x Lower'],
        'Winner': ['🏆 Naive', '🏆 Naive', '🏆 Naive', '🏆 Naive']
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=[data[col] for col in data.keys()][1:],
                    rowLabels=data['Metric'],
                    colLabels=['Naive', 'MidiTok', 'Winner'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(data['Metric'])):
        table[(i+1, 2)].set_facecolor('#90EE90')  # Light green for winner column
    
    # Header styling
    for j in range(3):
        table[(0, j)].set_facecolor('#B0C4DE')
        table[(0, j)].set_text_props(weight='bold')
    
    plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    output_path = Path('../outputs/model_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Summary table saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    print("Creating model comparison visualizations...")
    
    # Create comparison plot
    comparison_path = create_model_comparison()
    
    # Create summary table
    table_path = create_summary_table()
    
    print(f"\n🎉 Presentation materials created!")
    print(f"📊 Model comparison: {comparison_path}")
    print(f"📋 Summary table: {table_path}")
    print(f"\n💡 Use these for your presentation to show:")
    print(f"   • Naive tokenization significantly outperforms MidiTok")
    print(f"   • 5.4x improvement in average note generation")
    print(f"   • Consistent performance advantage across all metrics")