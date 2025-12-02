# Experiment 4: Neural Network Architecture Comparison

## Overview

This experiment compares how different neural network architectures perform for music generation, using the same tokenization strategy (naive) to avoid confounders.

## Hypothesis

On small datasets, simpler recurrent architectures like LSTM may outperform transformers, which typically require more data to learn effective attention patterns. This aligns with scaling-law intuition—transformers excel with large datasets but may overfit on small ones.

## Models Compared

| Architecture | Parameters | Description |
|--------------|------------|-------------|
| **LSTM (Baseline)** | 941,620 | Two-layer LSTM with 256 hidden units |
| **GRU** | 711,220 | Two-layer GRU with 256 hidden units (simpler than LSTM) |
| **Transformer (Small)** | 2,133,556 | 2 layers, 8 attention heads, d_model=256 |

All models trained on the same dataset:
- `data/nottingham-dataset-final_experiments_naive` (naive tokenization)
- 10 epochs, batch size 32, learning rate 0.001

## Checkpoints Used

```
models/generators/checkpoints/nottingham-dataset-final_experiments_naive/
├── lstm_20251129_200244.pth        # LSTM baseline
├── gru_20251201_171806.pth         # GRU
└── transformer_20251201_114247.pth # Small Transformer
```

## Evaluation Metrics

### Training Performance
- **Loss Curves**: Convergence speed and final loss value
- **Training Time**: Efficiency of each architecture

### Generation Quality (from evaluate.py)
- **Scale Consistency**: How well notes fit common musical scales (0-1)
- **Pitch Entropy**: Diversity of pitches used
- **Pitch Class Entropy**: Distribution across 12 pitch classes
- **Note Density**: Notes per second
- **Pitch Range**: Difference between highest and lowest pitch
- **Average Polyphony**: Simultaneous notes

### Optional
- **Discriminator Score**: How "real" the generated music sounds to a trained discriminator

## Running the Experiment

### Full Experiment
```bash
./experiments/experiment4/run_experiment4.sh
```

### Python Script
```bash
# Generate samples and evaluate
python experiments/experiment4/architecture_comparison_experiment.py \
    --output_dir experiments/experiment4/results \
    --num_samples 10 \
    --generate_length 200

# Analyze existing results
python experiments/experiment4/architecture_comparison_experiment.py \
    --analyze experiments/experiment4/results/experiment_results.json
```

### Generate Plots
```bash
python experiments/experiment4/plot_results.py \
    --results_dir experiments/experiment4/results
```

## Expected Findings

1. **LSTM Performance**: May achieve competitive or better generation quality despite having fewer parameters than the transformer.

2. **Transformer Behavior**: May show lower training loss (fits training data well) but not necessarily better generation—potential overfitting on small dataset.

3. **GRU Efficiency**: Similar performance to LSTM with ~25% fewer parameters, demonstrating parameter efficiency.

4. **Scaling Law Intuition**: Results may confirm that simpler architectures are more appropriate for small datasets, while transformers benefit from scale.

## Output Structure

```
experiments/experiment4/results/
├── experiment_results.json      # Full experiment results
├── detailed_results.csv         # Per-sample metrics
├── training_losses/             # Copied loss curves
│   ├── lstm_losses.npy
│   ├── gru_losses.npy
│   └── transformer_losses.npy
├── lstm/                        # Generated MIDI files
│   ├── conservative/
│   ├── balanced/
│   └── creative/
├── gru/
├── transformer/
└── plots/                       # Generated visualizations
    ├── training_loss_curves.png
    ├── loss_vs_params.png
    ├── architecture_overview.png
    ├── scale_consistency_grouped_bars.png
    ├── ...
    └── summary_statistics.md
```

## Key Plots

1. **Training Loss Curves** (`training_loss_curves.png`): Shows convergence behavior of each architecture.

2. **Loss vs Parameters** (`loss_vs_params.png`): Efficiency comparison—loss achieved per parameter.

3. **Architecture Overview** (`architecture_overview.png`): Comprehensive 6-panel comparison.

4. **Grouped Bar Charts**: Per-metric comparison across all architectures and temperature settings.

5. **Heatmap** (`comprehensive_heatmap.png`): All metrics at a glance.

## Interpretation Guide

| Metric | Higher = | Notes |
|--------|----------|-------|
| Scale Consistency | Better | Musically coherent output |
| Pitch Entropy | More diverse | But too high may be chaotic |
| Note Density | More notes | Style-dependent |
| Training Loss | Worse (usually) | But lower doesn't mean better generation |

## Files in This Directory

- `architecture_comparison_experiment.py` - Main experiment script
- `plot_results.py` - Visualization script
- `run_experiment4.sh` - Shell script to run full experiment
- `README.md` - This documentation
