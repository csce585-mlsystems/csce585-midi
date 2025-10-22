# Scripts Directory

This directory contains automation scripts for training and evaluation.

## Available Scripts

### `train_all_discriminators.py`
Comprehensive training script that trains all three discriminator models (MLP, LSTM, Transformer) with multiple configurations.

**Usage:**
```bash
# From project root
python scripts/train_all_discriminators.py
```

**What it does:**
- Trains 9 different model configurations
- MLP: 3 configurations (baseline, extended context, long training)
- LSTM: 3 configurations (baseline, extended context, long training)  
- Transformer: 3 configurations (baseline, extended context, long training)
- Auto-detects best device (MPS/CUDA/CPU)
- Saves all models and logs automatically
- Provides progress updates and final summary

**Requirements:**
- Must have measure dataset ready (`data/measures/measure_sequences.npy`)
- Run `python utils/measure_dataset.py` first if needed

**Output:**
- Model files: `models/discriminators/{model_type}/`
- Training logs: `logs/discriminators/train_summary.csv`

### `train_discriminators.py`
Individual discriminator training script (if exists).

## Running Scripts

Always run scripts from the project root directory:
```bash
cd /Users/cadestocker/Desktop/csce585-midi
python scripts/train_all_discriminators.py
```

The scripts automatically handle directory navigation and file paths.