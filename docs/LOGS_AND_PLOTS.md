# Training Logs and Plots - Complete Guide

## Overview

The training system now automatically saves comprehensive logs and plots to Google Drive (or local directory) so you never lose your training history, visualizations, or analysis data.

## What Gets Saved

### 1. Model Checkpoints (`models/`)

**File format**: `{model_type}_{timestamp}.pth`

Example: `lstm_20251111_143022.pth`

**Contents**: PyTorch state_dict with trained model weights

**Size**: 60-120 MB per model

**When saved**: At end of training (best model preserved)

### 2. Training Plots (`outputs/plots/`)

**File format**: `loss_curve_{model_type}_{timestamp}.png`

Example: `loss_curve_lstm_20251111_143022.png`

**Contents**: 
- Training loss curve (blue line)
- Validation loss curve (orange line)
- Grid lines for easy reading
- Clear axis labels and title

**Specifications**:
- Resolution: 150 DPI (high quality for presentations)
- Size: ~400-500 KB per plot
- Format: PNG with transparent background option

**When saved**: At end of training

### 3. Training Summaries (`logs/`)

**File format**: `training_summary_{model_type}_{timestamp}.txt`

Example: `training_summary_lstm_20251111_143022.txt`

**Contents**:
```
================================================================================
TRAINING SUMMARY - LSTM
================================================================================

Timestamp: 20251111_143022
Dataset: miditok_augmented
Model Type: lstm
Device: cuda

Model Configuration:
  Vocab Size: 284
  Sequence Length: 100
  Embedding Size: 256
  Hidden Size: 1024
  Num Layers: 4
  Dropout: 0.4
  Total Parameters: 85,234,944

Training Configuration:
  Epochs: 20
  Batch Size: 256
  Learning Rate: 0.001
  Subsample Ratio: 1.0
  Validation Split: 0.1
  Early Stopping Patience: 5

Training Results:
  Total Time: 183.45 minutes
  Epochs Completed: 14/20
  Final Training Loss: 0.2234
  Best Training Loss: 0.2134
  Final Validation Loss: 0.3456
  Best Validation Loss: 0.3398
  Train/Val Gap: +0.1222

Model saved to: /path/to/models/lstm_20251111_143022.pth
Plot saved to: /path/to/outputs/plots/loss_curve_lstm_20251111_143022.png
================================================================================
```

**Size**: ~2-3 KB per summary

**When saved**: At end of training

### 4. Loss Arrays (`logs/`)

**File formats**: 
- `train_losses_{model_type}_{timestamp}.npy`
- `val_losses_{model_type}_{timestamp}.npy`

Example: 
- `train_losses_lstm_20251111_143022.npy`
- `val_losses_lstm_20251111_143022.npy`

**Contents**: NumPy arrays with loss value for each epoch

**Usage**: For custom analysis, plotting, or comparisons

**Size**: ~1 KB each

**When saved**: At end of training

### 5. Experiment Log (`logs/models.csv`)

**File format**: CSV with all hyperparameters and results

**Columns**:
- Timestamp
- Model type, dataset
- All hyperparameters (hidden_size, num_layers, dropout, etc.)
- Final loss, min loss, validation loss
- Training time, number of epochs
- Early stopping status

**Purpose**: Track all experiments for easy comparison

**When saved**: Appended after each training run

## Directory Structure

### With Google Drive (`--checkpoint_dir` specified):

```
/content/drive/MyDrive/csce585_model_checkpoints/
└── miditok_augmented/
    ├── models/
    │   ├── lstm_20251111_143022.pth          (85 MB)
    │   ├── gru_20251111_163045.pth           (64 MB)
    │   └── transformer_20251111_193012.pth   (120 MB)
    ├── outputs/
    │   └── plots/
    │       ├── loss_curve_lstm_20251111_143022.png       (450 KB)
    │       ├── loss_curve_gru_20251111_163045.png        (445 KB)
    │       └── loss_curve_transformer_20251111_193012.png (472 KB)
    └── logs/
        ├── models.csv                                     (5 KB)
        ├── training_summary_lstm_20251111_143022.txt      (2 KB)
        ├── training_summary_gru_20251111_163045.txt       (2 KB)
        ├── training_summary_transformer_20251111_193012.txt (2 KB)
        ├── train_losses_lstm_20251111_143022.npy          (1 KB)
        ├── val_losses_lstm_20251111_143022.npy            (1 KB)
        └── ... (similar for other models)
```

**Total storage for 3 models**: ~275 MB

### Without Google Drive (local):

```
models/generators/checkpoints/miditok_augmented/
outputs/generators/miditok_augmented/
logs/generators/miditok_augmented/
```

Same structure, different base path.

## How to Use

### 1. Training with Checkpoint Directory (Colab)

```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --checkpoint_dir /content/drive/MyDrive/csce585_model_checkpoints \
    # ... other args
```

**Result**: Everything saved to Google Drive!

### 2. Training Locally

```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    # ... other args (no checkpoint_dir)
```

**Result**: Everything saved to local project directories.

## Viewing Results

### View Training Summary

```python
from pathlib import Path

log_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/logs')

# Get most recent summary
summaries = sorted(log_dir.glob('training_summary_*.txt'))
if summaries:
    with open(summaries[-1], 'r') as f:
        print(f.read())
```

### Display Loss Curve

```python
from IPython.display import Image, display
from pathlib import Path

plots_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/outputs/plots')
plot = sorted(plots_dir.glob('loss_curve_lstm_*.png'))[-1]

display(Image(filename=str(plot), width=800))
```

### Load Loss Arrays for Analysis

```python
import numpy as np
from pathlib import Path

log_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/logs')

# Load losses
train_losses = np.load(log_dir / 'train_losses_lstm_20251111_143022.npy')
val_losses = np.load(log_dir / 'val_losses_lstm_20251111_143022.npy')

# Analyze
print(f"Training epochs: {len(train_losses)}")
print(f"Best training loss: {train_losses.min():.4f} (epoch {train_losses.argmin()+1})")
print(f"Best validation loss: {val_losses.min():.4f} (epoch {val_losses.argmin()+1})")
print(f"Final train/val gap: {val_losses[-1] - train_losses[-1]:.4f}")
```

### Compare Multiple Models

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

log_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/logs')

models = ['lstm', 'gru', 'transformer']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(12, 6))

for model, color in zip(models, colors):
    train_file = sorted(log_dir.glob(f'train_losses_{model}_*.npy'))[-1]
    val_file = sorted(log_dir.glob(f'val_losses_{model}_*.npy'))[-1]
    
    train_losses = np.load(train_file)
    val_losses = np.load(val_file)
    
    plt.plot(train_losses, f'{color}--', label=f'{model.upper()} Train', alpha=0.7)
    plt.plot(val_losses, f'{color}-', label=f'{model.upper()} Val', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Comparison - Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Read Experiment Log

```python
import pandas as pd
from pathlib import Path

log_file = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/logs/models.csv')

if log_file.exists():
    df = pd.read_csv(log_file)
    
    # Show most recent experiments
    print("Recent Experiments:")
    print(df[['timestamp', 'model_type', 'final_val_loss', 'best_val_loss', 'train_time_sec']].tail(5))
    
    # Find best model
    best_model = df.loc[df['best_val_loss'].idxmin()]
    print(f"\nBest Model: {best_model['model_type']} (val loss: {best_model['best_val_loss']:.4f})")
```

## Benefits

### 1. Never Lose Results
- ✅ Plots saved even if Colab disconnects
- ✅ Summaries preserved for documentation
- ✅ Loss arrays available for re-plotting

### 2. Easy Comparison
- ✅ Visual comparison via loss curves
- ✅ Numerical comparison via summaries
- ✅ Quantitative comparison via CSV log

### 3. Presentation Ready
- ✅ High-quality PNG plots (150 DPI)
- ✅ Professional formatting
- ✅ Clear labels and legends

### 4. Reproducibility
- ✅ Complete hyperparameter record
- ✅ Training time and epoch information
- ✅ Random seed tracking (if implemented)

### 5. Analysis Flexibility
- ✅ Raw loss arrays for custom plots
- ✅ CSV for pandas/Excel analysis
- ✅ Text summaries for quick review

## Tips

### 1. Organize by Experiment

Create subdirectories for different experiments:

```bash
python training/train_generator.py \
    --checkpoint_dir /content/drive/MyDrive/experiments/experiment_1 \
    # ... other args
```

### 2. Add Notes to Summaries

After training, add notes to your summary files:

```python
from pathlib import Path

summary_file = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/logs/training_summary_lstm_20251111_143022.txt')

with open(summary_file, 'a') as f:
    f.write("\n\nNOTES:\n")
    f.write("- This model showed good convergence\n")
    f.write("- Validation loss plateaued around epoch 12\n")
    f.write("- Consider increasing dropout for next run\n")
```

### 3. Create Comparison Notebooks

Save your comparison code in a Colab notebook for reuse:

1. Mount Drive
2. Load all loss arrays
3. Generate comparison plots
4. Save comparison to new file

### 4. Backup Important Results

Download critical results before deleting:

```python
from google.colab import files
import shutil

# Zip important results
shutil.make_archive('/content/best_results', 'zip', 
                   '/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented')

files.download('/content/best_results.zip')
```

## Troubleshooting

### Problem: Plots not appearing

**Solution**: Check if matplotlib saved correctly:

```python
from pathlib import Path

plots_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/outputs/plots')
plots = list(plots_dir.glob('*.png'))

if not plots:
    print("No plots found - check training completed successfully")
else:
    print(f"Found {len(plots)} plots")
```

### Problem: Summary file incomplete

**Solution**: Training may have been interrupted. Check the model was saved:

```python
from pathlib import Path

model_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/models')
models = list(model_dir.glob('*.pth'))

print(f"Found {len(models)} model files")
# If model exists but summary doesn't, training was interrupted during cleanup
```

### Problem: Can't load loss arrays

**Solution**: Verify file exists and use correct path:

```python
import numpy as np
from pathlib import Path

log_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/logs')

# List all loss array files
loss_files = list(log_dir.glob('*_losses_*.npy'))
print(f"Available loss files: {len(loss_files)}")
for f in loss_files:
    print(f"  {f.name}")
```

## See Also

- [CHECKPOINT_SYSTEM.md](CHECKPOINT_SYSTEM.md): Complete checkpoint documentation
- [COLAB_TRAINING.md](COLAB_TRAINING.md): Colab training guide
- [COLAB_COMMANDS.md](COLAB_COMMANDS.md): Quick reference commands
