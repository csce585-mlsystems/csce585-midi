# Experiment 4: Architecture Comparison - Summary Statistics

## Training Performance

| Architecture | Parameters | Final Loss | Training Time |
|--------------|------------|------------|---------------|
| LSTM (Baseline) | 941,620 | 1.4186 | 51.80 minutes |
| GRU | 711,220 | 1.2103 | 335.25 minutes |
| Transformer (Small) | 2,133,556 | 0.8497 | 109.05 minutes |

## Generation Metrics


### LSTM (Baseline)

| Setting | Scale Consistency | Pitch Entropy | Note Density | Pitch Range |
|---------|-------------------|---------------|--------------|-------------|
| Conservative | 0.989 ± 0.013 | 3.192 ± 0.194 | 4.00 ± 0.00 | 19.1 ± 2.5 |
| Balanced | 0.958 ± 0.031 | 3.536 ± 0.201 | 4.00 ± 0.00 | 38.6 ± 7.7 |
| Creative | 0.817 ± 0.047 | 4.598 ± 0.275 | 4.00 ± 0.00 | 51.2 ± 1.1 |

### GRU

| Setting | Scale Consistency | Pitch Entropy | Note Density | Pitch Range |
|---------|-------------------|---------------|--------------|-------------|
| Conservative | 0.984 ± 0.024 | 3.108 ± 0.258 | 4.00 ± 0.00 | 19.1 ± 5.4 |
| Balanced | 0.958 ± 0.037 | 3.425 ± 0.286 | 4.00 ± 0.00 | 33.2 ± 13.3 |
| Creative | 0.799 ± 0.050 | 4.685 ± 0.326 | 4.00 ± 0.00 | 50.9 ± 1.3 |

### Transformer (Small)

| Setting | Scale Consistency | Pitch Entropy | Note Density | Pitch Range |
|---------|-------------------|---------------|--------------|-------------|
| Conservative | 0.994 ± 0.020 | 2.216 ± 0.549 | 4.00 ± 0.00 | 14.7 ± 2.4 |
| Balanced | 0.980 ± 0.025 | 2.920 ± 0.467 | 4.00 ± 0.00 | 19.5 ± 5.8 |
| Creative | 0.903 ± 0.032 | 3.942 ± 0.282 | 4.00 ± 0.00 | 47.1 ± 3.5 |

## Key Findings

- **Parameter Efficiency**: Compare how each architecture uses its parameters
- **Training Loss**: Lower training loss doesn't always mean better generation
- **Scale Consistency**: Higher values indicate more musically coherent output
- **Pitch Entropy**: Balance between diversity and repetition
