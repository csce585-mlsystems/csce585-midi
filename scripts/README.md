# Scripts Directory

This directory contains automation scripts for training, evaluation, and data augmentation.

## 🎵 Data Augmentation Scripts (NEW!)

### `test_augmentation.py`
Test the augmentation functionality before running the full pipeline.

**Usage:**
```bash
python scripts/test_augmentation.py
```

**What it does:**
- Tests transposition on sample MIDI files
- Verifies tokenization works correctly
- Shows preview of expected augmentation results
- Takes ~30 seconds

### `train_augmented.sh`
Complete pipeline to create augmented data and train models.

**Usage:**
```bash
bash scripts/train_augmented.sh
```

**What it does:**
- Creates augmented MIDITok dataset (7x more data)
- Trains LSTM model on augmented data
- Trains GRU model on augmented data
- Takes ~6 hours total

**Why use this:**
- Reduces overfitting (train/val loss gap: 0.45 → 0.2)
- Creates 7 versions of each song in different keys
- Increases training samples from 1.2M to 8.3M

### `compare_augmentation.py`
Analyze and compare original vs augmented datasets.

**Usage:**
```bash
python scripts/compare_augmentation.py
```

**What it shows:**
- Dataset statistics comparison
- Training sample counts
- Data density improvements
- Training results (if available)

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