import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def create_comparison_plots():
    """Generate presentation-ready comparison plots"""
    
    # Read training logs
    naive_logs = pd.read_csv("logs/naive/models.csv")
    miditok_logs = pd.read_csv("logs/miditok/models.csv")
    
    # subplots (a subplot is a single plot within a larger figure)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("MIDI Generation Model Comparison", fontsize=16, fontweight='bold')
    
    # Plot 1: Final Loss Comparison
    # mean of final losses for each dataset
    datasets = ['Naive', 'MidiTok']
    final_losses = [naive_logs['final_loss'].mean(), miditok_logs['final_loss'].mean()]
    
    # bar plot
    ax1.bar(datasets, final_losses, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    ax1.set_title('Final Training Loss Comparison')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Vocabulary Size Impact
    # mean of vocabulary sizes for each dataset
    vocab_sizes = [naive_logs['vocab_size'].iloc[0], miditok_logs['vocab_size'].iloc[0]]
    ax2.bar(datasets, vocab_sizes, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    ax2.set_title('Vocabulary Size Comparison')
    ax2.set_ylabel('Number of Tokens')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Time Efficiency
    # mean of training times for each dataset
    train_times = [naive_logs['train_time_sec'].mean(), miditok_logs['train_time_sec'].mean()]
    ax3.bar(datasets, train_times, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    ax3.set_title('Training Time Comparison')
    ax3.set_ylabel('Time (seconds)')
    ax3.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('outputs/presentation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved comparison plot to outputs/presentation_comparison.png")

if __name__ == "__main__":
    create_comparison_plots()